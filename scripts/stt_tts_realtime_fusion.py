# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "moshi_mlx",
#     "numpy",
#     "rustymimi",
#     "sentencepiece",
#     "sounddevice",
#     "msgpack",
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


class STTTTSFusion:
    def __init__(self, args):
        self.args = args
        self.stt_queue = queue.Queue()  # Use regular queue for sync callback
        self.audio_output_queue = asyncio.Queue()
        self.transcription_buffer = ""
        self.websocket = None
        self.text_queue = queue.Queue()  # Queue for text to send to TTS
        self.setup_stt_model()
        
    def setup_stt_model(self):
        """Initialize the MLX STT model"""
        print("Setting up STT model...")
        
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
        
        self.generated_codebooks = generated_codebooks
        self.other_codebooks = other_codebooks
        print("STT model setup complete!")

    def audio_callback(self, indata, _frames, _time, _status):
        """Callback for microphone input"""
        try:
            # Process audio block through STT
            block = indata.copy()
            block = block[None, :, 0]
            other_audio_tokens = self.audio_tokenizer.encode_step(block[None, 0:1])
            other_audio_tokens = mx.array(other_audio_tokens).transpose(0, 2, 1)[
                :, :, :self.other_codebooks
            ]
            text_token = self.gen.step(other_audio_tokens[0])
            text_token = text_token[0].item()
            
            if text_token not in (0, 3):
                text_piece = self.text_tokenizer.id_to_piece(text_token)
                text_piece = text_piece.replace("▁", " ")
                print(text_piece, end="", flush=True)
                
                # Add to transcription buffer
                self.transcription_buffer += text_piece
                
                # Send complete words/phrases to TTS via queue
                if " " in text_piece or text_piece.endswith(".") or text_piece.endswith("!") or text_piece.endswith("?"):
                    if self.transcription_buffer.strip():
                        self.text_queue.put(self.transcription_buffer.strip())
                        self.transcription_buffer = ""
                    
        except Exception as e:
            print(f"Error in audio callback: {e}")

    async def text_sender_task(self):
        """Monitor text queue and send to TTS server"""
        while True:
            try:
                # Check for text to send (non-blocking)
                try:
                    text = self.text_queue.get_nowait()
                    if self.websocket and text.strip():
                        print(f"\n[TTS] Sending: '{text}'")
                        await self.websocket.send(msgpack.packb({"type": "Text", "text": text + " "}))
                except queue.Empty:
                    pass
                
                await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
                
            except Exception as e:
                print(f"Error in text sender: {e}")
                await asyncio.sleep(0.1)

    async def receive_tts_audio(self):
        """Receive audio from TTS server and queue for playback"""
        try:
            async for message_bytes in self.websocket:
                msg = msgpack.unpackb(message_bytes)
                
                if msg["type"] == "Audio":
                    pcm = np.array(msg["pcm"]).astype(np.float32)
                    await self.audio_output_queue.put(pcm)
                    
        except websockets.exceptions.ConnectionClosed:
            print("TTS connection closed")
        except Exception as e:
            print(f"Error receiving TTS audio: {e}")
        finally:
            await self.audio_output_queue.put(None)  # Signal end

    async def play_audio(self):
        """Play received TTS audio through headphones"""
        should_exit = False
        
        def audio_output_callback(outdata, _a, _b, _c):
            nonlocal should_exit
            try:
                pcm_data = self.audio_output_queue.get_nowait()
                if pcm_data is not None:
                    outdata[:, 0] = pcm_data
                else:
                    should_exit = True
                    outdata[:] = 0
            except asyncio.QueueEmpty:
                outdata[:] = 0

        with sd.OutputStream(
            samplerate=SAMPLE_RATE,
            blocksize=1920,
            channels=1,
            callback=audio_output_callback,
        ):
            while not should_exit:
                await asyncio.sleep(0.1)

    async def connect_to_tts_server(self):
        """Connect to the TTS websocket server"""
        params = {"voice": self.args.voice, "format": "PcmMessagePack"}
        uri = f"{self.args.url}/api/tts_streaming?{urlencode(params)}"
        headers = {"kyutai-api-key": self.args.api_key}
        
        print(f"Connecting to TTS server: {uri}")
        
        try:
            self.websocket = await websockets.connect(uri, additional_headers=headers)
            print("Connected to TTS server!")
        except Exception as e:
            print(f"Failed to connect to TTS server: {e}")
            raise

    async def run(self):
        """Main execution loop"""
        # Connect to TTS server
        await self.connect_to_tts_server()
        
        # Start all async tasks
        text_sender_task = asyncio.create_task(self.text_sender_task())
        receive_task = asyncio.create_task(self.receive_tts_audio())
        playback_task = asyncio.create_task(self.play_audio())
        
        print("\nStarting real-time STT->TTS fusion...")
        print("Speak into your microphone, and you'll hear the TTS voice in your headphones!")
        print("Press Ctrl+C to stop.")
        
        # Start microphone input with STT processing
        with sd.InputStream(
            channels=1,
            dtype="float32",
            samplerate=SAMPLE_RATE,
            blocksize=1920,
            callback=self.audio_callback,
        ):
            try:
                # Keep running until interrupted
                await asyncio.gather(text_sender_task, receive_task, playback_task)
            except KeyboardInterrupt:
                print("\nStopping...")
            except Exception as e:
                print(f"Error in main loop: {e}")
            finally:
                if self.websocket:
                    await self.websocket.close()


async def main():
    parser = argparse.ArgumentParser(description="Real-time STT->TTS fusion using MLX and Rust server")
    parser.add_argument("--max-steps", default=4096, type=int, help="Max steps for STT model")
    parser.add_argument("--hf-repo", default="kyutai/stt-1b-en_fr-mlx", help="HuggingFace STT model repo")
    parser.add_argument(
        "--voice",
        default="expresso/ex03-ex01_happy_001_channel1_334s.wav",
        help=f"Voice to use for TTS. See {DEFAULT_DSM_TTS_VOICE_REPO}",
    )
    parser.add_argument(
        "--url",
        default="ws://213.173.110.201:22141",
        help="URL of the TTS server",
    )
    parser.add_argument("--api-key", default="public_token", help="API key for TTS server")
    
    args = parser.parse_args()
    
    fusion = STTTTSFusion(args)
    await fusion.run()


if __name__ == "__main__":
    asyncio.run(main())
