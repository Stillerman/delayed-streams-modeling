# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "moshi_mlx",
#     "msgpack",
#     "numpy",
#     "rustymimi",
#     "sentencepiece",
#     "sounddevice",
#     "sphn",
#     "websockets",
#     "tqdm",
# ]
# ///

import argparse
import asyncio
import json
import queue
import threading
from urllib.parse import urlencode

import mlx.core as mx
import mlx.nn as nn
import msgpack
import numpy as np
import rustymimi
import sentencepiece
import sounddevice as sd
import sphn
import tqdm
import websockets
from huggingface_hub import hf_hub_download
from moshi_mlx import models, utils

SAMPLE_RATE = 24000
DEFAULT_DSM_TTS_VOICE_REPO = "kyutai/tts-voices"
AUTH_TOKEN = "public_token"


class STSPipeline:
    def __init__(self, args):
        self.args = args
        self.websocket = None
        self.output_queue = asyncio.Queue()
        self.text_queue = asyncio.Queue()
        self.audio_block_queue = queue.Queue()
        self.should_exit = False
        
        # Initialize STT model
        self.init_stt_model()
        
    def init_stt_model(self):
        """Initialize the STT model components"""
        print("Initializing STT model...")
        
        lm_config = hf_hub_download(self.args.hf_repo, "config.json")
        with open(lm_config, "r") as fobj:
            lm_config = json.load(fobj)
        mimi_weights = hf_hub_download(self.args.hf_repo, lm_config["mimi_name"])
        moshi_name = lm_config.get("moshi_name", "model.safetensors")
        moshi_weights = hf_hub_download(self.args.hf_repo, moshi_name)
        tokenizer = hf_hub_download(self.args.hf_repo, lm_config["tokenizer_name"])

        lm_config = models.LmConfig.from_config_dict(lm_config)
        self.model = models.Lm(lm_config)
        self.model.set_dtype(mx.bfloat16)
        if moshi_weights.endswith(".q4.safetensors"):
            nn.quantize(self.model, bits=4, group_size=32)
        elif moshi_weights.endswith(".q8.safetensors"):
            nn.quantize(self.model, bits=8, group_size=64)

        print(f"Loading model weights from {moshi_weights}")
        self.model.load_weights(moshi_weights, strict=True)

        print(f"Loading the text tokenizer from {tokenizer}")
        self.text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer)

        print(f"Loading the audio tokenizer {mimi_weights}")
        generated_codebooks = lm_config.generated_codebooks
        other_codebooks = lm_config.other_codebooks
        mimi_codebooks = max(generated_codebooks, other_codebooks)
        self.audio_tokenizer = rustymimi.Tokenizer(mimi_weights, num_codebooks=mimi_codebooks)
        
        print("Warming up the model")
        self.model.warmup()
        self.gen = models.LmGen(
            model=self.model,
            max_steps=self.args.max_steps,
            text_sampler=utils.Sampler(top_k=25, temp=0),
            audio_sampler=utils.Sampler(top_k=250, temp=0.8),
            check=False,
        )
        
        self.other_codebooks = other_codebooks
        
    def audio_callback(self, indata, _frames, _time, _status):
        """Callback for audio input from microphone"""
        self.audio_block_queue.put(indata.copy())

    def process_stt_in_thread(self):
        """Process STT in a separate thread"""
        print("Starting STT processing thread...")
        while not self.should_exit:
            try:
                block = self.audio_block_queue.get(timeout=1.0)
                block = block[None, :, 0]
                other_audio_tokens = self.audio_tokenizer.encode_step(block[None, 0:1])
                other_audio_tokens = mx.array(other_audio_tokens).transpose(0, 2, 1)[
                    :, :, :self.other_codebooks
                ]
                text_token = self.gen.step(other_audio_tokens[0])
                text_token = text_token[0].item()
                
                if text_token not in (0, 3):
                    _text = self.text_tokenizer.id_to_piece(text_token)
                    _text = _text.replace("▁", " ")
                    print(_text, end="", flush=True)
                    # Send text to WebSocket
                    asyncio.run_coroutine_threadsafe(
                        self.text_queue.put(_text), 
                        self.loop
                    )
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"STT processing error: {e}")
                break
                
        print("STT processing thread stopped")

    async def send_text_to_websocket(self):
        """Send recognized text to WebSocket"""
        print("Starting text sender...")
        accumulated_text = ""
        
        while not self.should_exit:
            try:
                text_chunk = await asyncio.wait_for(self.text_queue.get(), timeout=1.0)
                accumulated_text += text_chunk
                
                # Send text in chunks when we have enough or after a delay
                if len(accumulated_text) > 10 or accumulated_text.endswith(". "):
                    await self.websocket.send(msgpack.packb({"type": "Text", "text": accumulated_text}))
                    print(f"\nSent: '{accumulated_text.strip()}'")
                    accumulated_text = ""
                    
            except asyncio.TimeoutError:
                # Send accumulated text if we haven't sent anything in a while
                if accumulated_text.strip():
                    await self.websocket.send(msgpack.packb({"type": "Text", "text": accumulated_text}))
                    print(f"\nSent (timeout): '{accumulated_text.strip()}'")
                    accumulated_text = ""
                continue
            except Exception as e:
                print(f"Text sending error: {e}")
                break
                
        # Send any remaining text and EOS
        if accumulated_text.strip():
            await self.websocket.send(msgpack.packb({"type": "Text", "text": accumulated_text}))
        await self.websocket.send(msgpack.packb({"type": "Eos"}))
        print("Text sender stopped")

    async def receive_audio_from_websocket(self):
        """Receive audio from WebSocket and queue for playback"""
        print("Starting audio receiver...")
        with tqdm.tqdm(desc="Receiving audio", unit=" seconds generated") as pbar:
            accumulated_samples = 0
            last_seconds = 0

            async for message_bytes in self.websocket:
                msg = msgpack.unpackb(message_bytes)

                if msg["type"] == "Audio":
                    pcm = np.array(msg["pcm"]).astype(np.float32)
                    await self.output_queue.put(pcm)

                    accumulated_samples += len(msg["pcm"])
                    current_seconds = accumulated_samples // SAMPLE_RATE
                    if current_seconds > last_seconds:
                        pbar.update(current_seconds - last_seconds)
                        last_seconds = current_seconds

        print("Audio receiver stopped")
        await self.output_queue.put(None)  # Signal end of audio

    async def play_audio(self):
        """Play received audio"""
        print("Starting audio playback...")
        should_exit_playback = False

        def audio_output_callback(outdata, _a, _b, _c):
            nonlocal should_exit_playback
            try:
                pcm_data = self.output_queue.get_nowait()
                if pcm_data is not None:
                    outdata[:, 0] = pcm_data
                else:
                    should_exit_playback = True
                    outdata[:] = 0
            except asyncio.QueueEmpty:
                outdata[:] = 0

        with sd.OutputStream(
            samplerate=SAMPLE_RATE,
            blocksize=1920,
            channels=1,
            callback=audio_output_callback,
        ):
            while not should_exit_playback and not self.should_exit:
                await asyncio.sleep(0.1)
                
        print("Audio playback stopped")

    async def run(self):
        """Main run method"""
        # Set up WebSocket connection
        params = {"voice": self.args.voice, "format": "PcmMessagePack"}
        uri = f"{self.args.url}/api/tts_streaming?{urlencode(params)}"
        print(f"Connecting to: {uri}")
        
        headers = {"kyutai-api-key": self.args.api_key}
        
        async with websockets.connect(uri, additional_headers=headers) as websocket:
            self.websocket = websocket
            self.loop = asyncio.get_event_loop()
            
            # Start STT processing thread
            stt_thread = threading.Thread(target=self.process_stt_in_thread)
            stt_thread.start()
            
            # Start audio input
            print("Starting microphone input...")
            with sd.InputStream(
                channels=1,
                dtype="float32",
                samplerate=SAMPLE_RATE,
                blocksize=1920,
                callback=self.audio_callback,
            ):
                print("Recording audio from microphone, speak to get real-time translation...")
                
                # Start all async tasks
                tasks = [
                    asyncio.create_task(self.send_text_to_websocket()),
                    asyncio.create_task(self.receive_audio_from_websocket()),
                    asyncio.create_task(self.play_audio()),
                ]
                
                try:
                    await asyncio.gather(*tasks)
                except KeyboardInterrupt:
                    print("\nShutting down...")
                    self.should_exit = True
                    
                    # Cancel all tasks
                    for task in tasks:
                        task.cancel()
                    
                    # Wait for STT thread to finish
                    stt_thread.join(timeout=2.0)


async def main():
    parser = argparse.ArgumentParser(description="Real-time Speech-to-Speech pipeline")
    parser.add_argument("--max-steps", default=4096, type=int)
    parser.add_argument("--hf-repo", default="kyutai/stt-1b-en_fr-mlx")
    parser.add_argument(
        "--voice",
        default="expresso/ex03-ex01_happy_001_channel1_334s.wav",
        help="The voice to use, relative to the voice repo root. "
        f"See {DEFAULT_DSM_TTS_VOICE_REPO}",
    )
    parser.add_argument(
        "--url",
        help="The URL of the server to which to send the audio",
        default="ws://213.173.108.203:17594",
    )
    parser.add_argument("--api-key", default="public_token")
    
    args = parser.parse_args()
    
    pipeline = STSPipeline(args)
    await pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())