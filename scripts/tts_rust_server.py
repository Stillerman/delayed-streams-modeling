# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "msgpack",
#     "numpy",
#     "sphn",
#     "websockets",
#     "sounddevice",
#     "tqdm",
# ]
# ///
import argparse
import asyncio
import sys
from urllib.parse import urlencode

import msgpack
import numpy as np
import sounddevice as sd
import sphn
import tqdm
import websockets

SAMPLE_RATE = 24000

TTS_TEXT = "Hello, this is a test of the moshi text to speech system, this should result in some nicely sounding generated voice."
DEFAULT_DSM_TTS_VOICE_REPO = "kyutai/tts-voices"
AUTH_TOKEN = "public_token"


async def receive_messages(websocket: websockets.ClientConnection, output_queue):
    with tqdm.tqdm(desc="Receiving audio", unit=" seconds generated") as pbar:
        accumulated_samples = 0
        last_seconds = 0

        async for message_bytes in websocket:
            msg = msgpack.unpackb(message_bytes)

            if msg["type"] == "Audio":
                pcm = np.array(msg["pcm"]).astype(np.float32)
                await output_queue.put(pcm)

                accumulated_samples += len(msg["pcm"])
                current_seconds = accumulated_samples // SAMPLE_RATE
                if current_seconds > last_seconds:
                    pbar.update(current_seconds - last_seconds)
                    last_seconds = current_seconds

    print("End of audio.")
    await output_queue.put(None)  # Signal end of audio


async def send_text_words(websocket: websockets.ClientConnection, text: str, word_delay: float = 2.0, words_per_batch: int = 5):
    """Send text in batches of words with delays."""
    words = text.split()
    total_batches = (len(words) + words_per_batch - 1) // words_per_batch  # Ceiling division
    print(f"Sending {len(words)} words in {total_batches} batches of {words_per_batch} words each, with {word_delay}s delay between batches...")
    
    for i in range(0, len(words), words_per_batch):
        batch = words[i:i + words_per_batch]
        batch_text = " ".join(batch) + " "
        batch_num = (i // words_per_batch) + 1
        
        print(f"Sending batch {batch_num}/{total_batches}: '{batch_text.strip()}'")
        await websocket.send(msgpack.packb({"type": "Text", "text": batch_text}))
        
        # Add delay between batches (except after the last batch)
        if i + words_per_batch < len(words):
            await asyncio.sleep(word_delay)
    
    print("Sending EOS signal...")
    await websocket.send(msgpack.packb({"type": "Eos"}))


async def output_audio(out: str, output_queue: asyncio.Queue[np.ndarray | None]):
    if out == "-":
        should_exit = False

        def audio_callback(outdata, _a, _b, _c):
            nonlocal should_exit

            try:
                pcm_data = output_queue.get_nowait()
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
            callback=audio_callback,
        ):
            while True:
                if should_exit:
                    break
                await asyncio.sleep(1)
    else:
        frames = []
        while True:
            item = await output_queue.get()
            if item is None:
                break
            frames.append(item)

        sphn.write_wav(out, np.concat(frames, -1), SAMPLE_RATE)
        print(f"Saved audio to {out}")


async def websocket_client():
    parser = argparse.ArgumentParser(description="Use the TTS streaming API with word-by-word input")
    parser.add_argument("inp", type=str, help="Input file, use - for stdin.")
    parser.add_argument(
        "out", type=str, help="Output file to generate, use - for playing the audio"
    )
    parser.add_argument(
        "--voice",
        default="expresso/ex03-ex01_happy_001_channel1_334s.wav",
        help="The voice to use, relative to the voice repo root. "
        f"See {DEFAULT_DSM_TTS_VOICE_REPO}",
    )
    parser.add_argument(
        "--url",
        help="The URL of the server to which to send the audio",
        default="ws://127.0.0.1:8080",
    )
    parser.add_argument("--api-key", default="public_token")
    parser.add_argument(
        "--word-delay",
        type=float,
        default=2.0,
        help="Delay in seconds between sending each batch of words (default: 2.0)"
    )
    parser.add_argument(
        "--words-per-batch",
        type=int,
        default=5,
        help="Number of words to send in each batch (default: 5)"
    )
    args = parser.parse_args()

    params = {"voice": args.voice, "format": "PcmMessagePack"}
    uri = f"{args.url}/api/tts_streaming?{urlencode(params)}"
    print(uri)

    # Read input text
    if args.inp == "-":
        if sys.stdin.isatty():  # Interactive
            print("Enter text to synthesize (Ctrl+D to end input):")
        text_to_tts = sys.stdin.read().strip()
    else:
        with open(args.inp, "r") as fobj:
            text_to_tts = fobj.read().strip()

    headers = {"kyutai-api-key": args.api_key}

    async with websockets.connect(uri, additional_headers=headers) as websocket:
        # Create output queue for audio
        output_queue = asyncio.Queue()
        
        # Start all tasks concurrently
        send_task = asyncio.create_task(send_text_words(websocket, text_to_tts, args.word_delay, args.words_per_batch))
        receive_task = asyncio.create_task(receive_messages(websocket, output_queue))
        output_audio_task = asyncio.create_task(output_audio(args.out, output_queue))
        
        # Wait for all tasks to complete
        await asyncio.gather(send_task, receive_task, output_audio_task)


if __name__ == "__main__":
    asyncio.run(websocket_client())
