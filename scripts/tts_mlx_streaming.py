# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "moshi_mlx==0.2.9",
#     "numpy",
#     "sounddevice",
# ]
# ///

import argparse
import json
import queue
import sys
import threading
import time
from typing import Optional, List
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sentencepiece
import sounddevice as sd
import sphn
from moshi_mlx import models
from moshi_mlx.client_utils import make_log
from moshi_mlx.models.tts import (
    DEFAULT_DSM_TTS_REPO,
    DEFAULT_DSM_TTS_VOICE_REPO,
    TTSModel,
    Entry,
    State,
)
from moshi_mlx.models.generate import LmGen
from moshi_mlx.utils.loaders import hf_get


def log(level: str, msg: str):
    print(make_log(level, msg))


@dataclass
class ClientState:
    """Tracks the state of a streaming TTS client."""
    is_complete: bool = False
    state: Optional[State] = None
    offset: int = 0
    text_queue: queue.Queue = None
    
    def __post_init__(self):
        self.text_queue = queue.Queue()
    
    def reset(self, state_machine):
        """Reset the client state."""
        self.is_complete = False
        self.offset = 0
        self.state = state_machine.new_state([])


class StreamingTTSService:
    """Implements word-by-word streaming TTS similar to the Rust server."""
    
    def __init__(self, tts_model: TTSModel, cfg_coef_conditioning, cfg_is_no_text: bool, cfg_is_no_prefix: bool):
        self.tts_model = tts_model
        self.cfg_coef_conditioning = cfg_coef_conditioning
        self.cfg_is_no_text = cfg_is_no_text
        self.cfg_is_no_prefix = cfg_is_no_prefix
        
        # Initialize components
        self.lm = tts_model.lm
        self.mimi = tts_model.mimi
        self.machine = tts_model.machine
        self.text_tokenizer = None  # Will be set from main
        
        # Setup client state
        self.client = ClientState()
        
        # Audio output queue
        self.wav_frames = queue.Queue()
        
        # Setup voice attributes
        self.attributes = None
        
        # Generation state
        self.lm_gen = None
        self.is_running = True
        self.generation_thread = None
        
    def prepare_voice(self, voice_path: Optional[str]):
        """Prepare voice attributes."""
        if self.tts_model.multi_speaker and voice_path:
            voices = [self.tts_model.get_voice_path(voice_path)]
        else:
            voices = []
        self.attributes = self.tts_model.make_condition_attributes(voices, self.cfg_coef_conditioning)
        
    def add_text(self, text: str):
        """Add text to the generation queue."""
        # Tokenize the text
        tokens = self.text_tokenizer.encode(text)
        if tokens:
            self.client.text_queue.put(tokens)
            log("info", f"Added {len(tokens)} tokens to queue")
            
    def signal_complete(self):
        """Signal that no more text will be added."""
        self.client.text_queue.put(None)
        self.client.is_complete = True
        
    def _on_frame(self, frame):
        """Process generated audio frame."""
        if (frame == -1).any():
            return
        _pcm = self.mimi.decode_step(frame[:, :, None])
        _pcm = np.array(mx.clip(_pcm[0, 0], -1, 1))
        self.wav_frames.put_nowait(_pcm)
        
    def _on_text_hook(self, text_tokens):
        """Hook called when text tokens are generated."""
        if self.client.state is None:
            return
            
        tokens = text_tokens.tolist()
        out_tokens = []
        
        for token in tokens:
            # Process token through state machine
            out_token, consumed_new_word = self.machine.process(
                self.client.offset, self.client.state, token
            )
            
            if consumed_new_word:
                log("info", "Consumed a word")
                
            out_tokens.append(out_token)
            
        text_tokens[:] = mx.array(out_tokens, dtype=mx.int32)
        
    def _on_audio_hook(self, audio_tokens):
        """Hook called when audio tokens are generated."""
        # Apply delay masking
        delays = getattr(self.lm_gen, 'delays', [0] * audio_tokens.shape[1])
        for i, delay in enumerate(delays[1:], 1):
            if self.client.offset < delay + self.tts_model.delay_steps:
                audio_tokens[:, i] = self.machine.token_ids.zero
                
    def start_generation(self):
        """Start the generation thread."""
        self.generation_thread = threading.Thread(target=self._generation_loop, daemon=True)
        self.generation_thread.start()
        
    def _generation_loop(self):
        """Main generation loop that processes text incrementally."""
        log("info", "Starting generation loop")
        
        # Initialize LmGen
        self.lm_gen = LmGen(
            self.lm,
            temp=self.tts_model.temp,
            temp_text=self.tts_model.temp,
            cfg_coef=self.tts_model.cfg_coef,
            condition_tensors=None,
            on_text_hook=self._on_text_hook,
            on_audio_hook=self._on_audio_hook,
            cfg_is_no_text=self.cfg_is_no_text,
        )
        
        # Initialize state
        self.client.reset(self.machine)
        
        # Prepare initial attributes
        all_entries = [[]]  # Start with empty entries
        all_attributes = [self.attributes]
        
        # Start streaming generation
        self.lm_gen.streaming_forever(1)
        
        # Initialize input tokens
        n_q = self.lm.n_q - self.lm.dep_q
        input_tokens = mx.full((1, n_q, 1), self.machine.token_ids.zero, dtype=mx.int32)
        
        last_token_time = time.time()
        
        while self.is_running:
            # Check for new tokens
            try:
                new_tokens = self.client.text_queue.get(timeout=0.01)
                if new_tokens is None:
                    log("info", "Received end of text signal")
                    self.client.state.entries.append(Entry([-2], '', padding=0))  # End token
                else:
                    # Add new tokens as entries
                    for token in new_tokens:
                        self.client.state.entries.append(Entry([token], '', padding=1))
                    log("info", f"Added {len(new_tokens)} tokens to state")
            except queue.Empty:
                pass
                
            # Check if we should generate
            should_generate = False
            
            if self.client.state is None:
                should_generate = False
            elif self.client.is_complete and not self.client.state.entries:
                # We're done
                break
            elif self.client.state.forced_padding > 0:
                # We have padding to generate
                should_generate = True
            elif len(self.client.state.entries) > self.machine.second_stream_ahead:
                # We have enough entries queued
                should_generate = True
            
            if should_generate:
                # Generate one step
                frame = self.lm_gen.step(input_tokens)
                if frame is not None:
                    # Process audio frame
                    audio_frame = frame[:, 1:]
                    audio_frame = mx.clip(audio_frame, 0, None)
                    
                    # Decode to PCM
                    pcm = self.mimi.decode_step(audio_frame)
                    if pcm is not None:
                        pcm = mx.clip(pcm, -0.99, 0.99)
                        self._on_frame(audio_frame)
                    
                    self.client.offset += 1
                    
                    # Check if generation is complete
                    if self.client.is_complete and self.client.state.end_step is not None:
                        real_end = (
                            self.client.state.end_step + 
                            self.tts_model.delay_steps + 
                            self.tts_model.final_padding
                        )
                        if self.client.offset >= real_end:
                            log("info", "Generation complete")
                            break
            else:
                # Wait a bit if we're not generating
                time.sleep(0.001)
                
        self.wav_frames.put(None)  # Signal end
        log("info", "Generation loop ended")
        
    def stop(self):
        """Stop the generation."""
        self.is_running = False
        if self.generation_thread:
            self.generation_thread.join()


class TextStreamReader:
    """Reads text from stdin and feeds it to the TTS service."""
    
    def __init__(self, tts_service: StreamingTTSService):
        self.tts_service = tts_service
        self.thread = None
        
    def start(self):
        """Start reading from stdin."""
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        
    def _read_loop(self):
        """Read text from stdin and feed to TTS."""
        try:
            if sys.stdin.isatty():
                print("Enter text to synthesize (press Enter to send each line, Ctrl+D to finish):")
            
            while True:
                line = sys.stdin.readline()
                if not line:  # EOF
                    self.tts_service.signal_complete()
                    break
                    
                # Send line to TTS service
                text = line.strip()
                if text:
                    log("info", f"Sending text: {text[:50]}...")
                    self.tts_service.add_text(text + " ")
                    
        except Exception as e:
            log("error", f"Error reading input: {e}")
            self.tts_service.signal_complete()


def main():
    parser = argparse.ArgumentParser(
        description="Run Kyutai TTS with word-by-word streaming using MLX"
    )
    parser.add_argument("inp", type=str, help="Input file, use - for stdin streaming")
    parser.add_argument(
        "out", type=str, help="Output file to generate, use - for playing the audio"
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=DEFAULT_DSM_TTS_REPO,
        help="HF repo in which to look for the pretrained models.",
    )
    parser.add_argument(
        "--voice-repo",
        default=DEFAULT_DSM_TTS_VOICE_REPO,
        help="HF repo in which to look for pre-computed voice embeddings.",
    )
    parser.add_argument(
        "--voice", default="expresso/ex03-ex01_happy_001_channel1_334s.wav"
    )
    parser.add_argument(
        "--quantize",
        type=int,
        help="The quantization to be applied, e.g. 8 for 8 bits.",
    )
    args = parser.parse_args()

    if args.inp != "-":
        log("error", "This script only supports streaming from stdin. Use - for input.")
        sys.exit(1)

    mx.random.seed(299792458)

    log("info", "retrieving checkpoints")

    raw_config = hf_get("config.json", args.hf_repo)
    with open(hf_get(raw_config), "r") as fobj:
        raw_config = json.load(fobj)

    mimi_weights = hf_get(raw_config["mimi_name"], args.hf_repo)
    moshi_name = raw_config.get("moshi_name", "model.safetensors")
    moshi_weights = hf_get(moshi_name, args.hf_repo)
    tokenizer = hf_get(raw_config["tokenizer_name"], args.hf_repo)
    lm_config = models.LmConfig.from_config_dict(raw_config)
    model = models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)

    log("info", f"loading model weights from {moshi_weights}")
    model.load_pytorch_weights(str(moshi_weights), lm_config, strict=True)

    if args.quantize is not None:
        log("info", f"quantizing model to {args.quantize} bits")
        nn.quantize(model.depformer, bits=args.quantize)
        for layer in model.transformer.layers:
            nn.quantize(layer.self_attn, bits=args.quantize)
            nn.quantize(layer.gating, bits=args.quantize)

    log("info", f"loading the text tokenizer from {tokenizer}")
    text_tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer))

    log("info", f"loading the audio tokenizer {mimi_weights}")
    generated_codebooks = lm_config.generated_codebooks
    audio_tokenizer = models.mimi.Mimi(models.mimi_202407(generated_codebooks))
    audio_tokenizer.load_pytorch_weights(str(mimi_weights), strict=True)

    cfg_coef_conditioning = None
    tts_model = TTSModel(
        model,
        audio_tokenizer,
        text_tokenizer,
        voice_repo=args.voice_repo,
        temp=0.6,
        cfg_coef=1,
        max_padding=8,
        initial_padding=2,
        final_padding=2,
        padding_bonus=0,
        raw_config=raw_config,
    )
    if tts_model.valid_cfg_conditionings:
        cfg_coef_conditioning = tts_model.cfg_coef
        tts_model.cfg_coef = 1.0
        cfg_is_no_text = False
        cfg_is_no_prefix = False
    else:
        cfg_is_no_text = True
        cfg_is_no_prefix = True

    # Create streaming service
    service = StreamingTTSService(tts_model, cfg_coef_conditioning, cfg_is_no_text, cfg_is_no_prefix)
    service.text_tokenizer = text_tokenizer  # Set the tokenizer
    service.prepare_voice(args.voice)
    
    # Start text reader
    reader = TextStreamReader(service)
    reader.start()
    
    # Start generation
    service.start_generation()

    if args.out == "-":
        # Play audio in real-time
        def audio_callback(outdata, _a, _b, _c):
            try:
                pcm_data = service.wav_frames.get(block=False)
                if pcm_data is not None:
                    outdata[:, 0] = pcm_data
                else:
                    outdata[:] = 0
            except queue.Empty:
                outdata[:] = 0

        with sd.OutputStream(
            samplerate=tts_model.mimi.sample_rate,
            blocksize=1920,
            channels=1,
            callback=audio_callback,
        ):
            # Wait for generation to complete
            service.generation_thread.join()
            
            # Keep playing until all audio is consumed
            log("info", "Waiting for audio playback to complete...")
            while True:
                if service.wav_frames.qsize() == 0:
                    time.sleep(0.5)
                    break
                time.sleep(0.1)
    else:
        # Save to file
        service.generation_thread.join()
        
        # Collect all audio frames
        frames = []
        while True:
            frame = service.wav_frames.get()
            if frame is None:
                break
            frames.append(frame)
            
        if frames:
            wav = np.concatenate(frames, -1)
            sphn.write_wav(args.out, wav, tts_model.mimi.sample_rate)
            log("info", f"Saved audio to {args.out}")
        else:
            log("warning", "No audio frames generated")
            
    service.stop()


if __name__ == "__main__":
    main()