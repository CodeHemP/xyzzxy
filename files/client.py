"""
Real-Time Translation Client
Captures mic audio, sends to RunPod server, plays back translated audio.

Usage:
    python client.py --server wss://YOUR-RUNPOD-URL --lang hindi
    python client.py --server wss://YOUR-RUNPOD-URL --lang vietnamese
    python client.py --server wss://YOUR-RUNPOD-URL --lang nepali
    python client.py --server wss://YOUR-RUNPOD-URL --lang sinhala
"""

import asyncio
import argparse
import json
import base64
import sys
import threading
import queue
import numpy as np

try:
    import pyaudio
except ImportError:
    print("ERROR: pyaudio not installed.")
    print("  Mac:     brew install portaudio && pip install pyaudio")
    print("  Linux:   sudo apt install portaudio19-dev && pip install pyaudio")
    print("  Windows: pip install pyaudio")
    sys.exit(1)

try:
    import websockets
except ImportError:
    print("ERROR: websockets not installed. Run: pip install websockets")
    sys.exit(1)


# ── Audio Settings ──────────────────────────────────────

SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_MS = 100                # Send audio every 100ms
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_MS / 1000)  # 1600 samples


class AudioPlayer:
    """Plays received translated audio through speakers/headphones."""

    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.audio_queue = queue.Queue()
        self.streams = {}
        self.running = True
        self.thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.thread.start()

    def _get_stream(self, sample_rate):
        if sample_rate not in self.streams:
            self.streams[sample_rate] = self.pa.open(
                format=pyaudio.paInt16, channels=1, rate=sample_rate,
                output=True, frames_per_buffer=2048,
            )
        return self.streams[sample_rate]

    def _playback_loop(self):
        while self.running:
            try:
                sample_rate, audio_bytes = self.audio_queue.get(timeout=0.1)
                stream = self._get_stream(sample_rate)
                chunk_size = 4096
                for i in range(0, len(audio_bytes), chunk_size):
                    if not self.running:
                        break
                    stream.write(audio_bytes[i:i + chunk_size])
            except queue.Empty:
                continue

    def play(self, audio_bytes, sample_rate):
        self.audio_queue.put((sample_rate, audio_bytes))

    def stop(self):
        self.running = False
        for stream in self.streams.values():
            stream.stop_stream()
            stream.close()
        self.pa.terminate()


class TranslationClient:
    def __init__(self, server_url, target_lang):
        self.server_url = server_url
        self.target_lang = target_lang
        self.player = AudioPlayer()

    async def run(self):
        print(f"\n{'='*60}")
        print(f"  Real-Time Lecture Translation Client")
        print(f"  Server: {self.server_url}")
        print(f"  Target: {self.target_lang}")
        print(f"{'='*60}")
        print(f"\n  Connecting to server...")

        try:
            async with websockets.connect(
                self.server_url, max_size=2**20, ping_interval=20, open_timeout=30,
            ) as ws:
                # Send config
                await ws.send(json.dumps({
                    "type": "config",
                    "target_language": self.target_lang,
                }))

                # Wait for ack
                ack = json.loads(await ws.recv())
                print(f"  Connected! Language: {ack.get('target_language')}")
                print(f"\n  Speak into your microphone. Press Ctrl+C to stop.\n")
                print(f"{'─'*60}")

                # Run send and receive concurrently
                await asyncio.gather(
                    self._send_audio(ws),
                    self._receive_results(ws),
                )
        except ConnectionRefusedError:
            print(f"\n  ERROR: Could not connect to {self.server_url}")
            print(f"  Check that:")
            print(f"    1. The server is running on RunPod (python server.py)")
            print(f"    2. The URL is correct")
            print(f"    3. Port 8765 is exposed in RunPod pod settings")
        except Exception as e:
            print(f"\n  ERROR: {e}")

    async def _send_audio(self, ws):
        """Capture mic and stream to server."""
        pa = pyaudio.PyAudio()
        info = pa.get_default_input_device_info()
        print(f"  [Mic] Using: {info['name']}")

        stream = pa.open(
            format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE,
            input=True, frames_per_buffer=CHUNK_SIZE,
        )

        try:
            while True:
                audio_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                await ws.send(audio_data)
                await asyncio.sleep(0.001)
        except asyncio.CancelledError:
            pass
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

    async def _receive_results(self, ws):
        """Receive translations and play audio."""
        try:
            async for message in ws:
                data = json.loads(message)

                if data["type"] == "translation":
                    uid = data["utterance_id"]
                    orig = data["original_text"]
                    trans = data["translated_text"]
                    ms = data["pipeline_total_ms"]
                    timings = data["timings"]

                    print(f"\n  ┌─ #{uid} ({ms:.0f}ms)")
                    print(f"  │  EN: {orig}")
                    print(f"  │  ->: {trans}")
                    print(
                        f"  └─ ASR={timings['asr_ms']:.0f}ms "
                        f"MT={timings['mt_ms']:.0f}ms "
                        f"TTS={timings['tts_ms']:.0f}ms"
                    )

                elif data["type"] == "audio":
                    audio_bytes = base64.b64decode(data["audio_b64"])
                    sample_rate = data["sample_rate"]
                    self.player.play(audio_bytes, sample_rate)

        except asyncio.CancelledError:
            pass

    def stop(self):
        self.player.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Translation Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python client.py --server wss://dp20abc-8765.proxy.runpod.net --lang hindi
  python client.py --server wss://dp20abc-8765.proxy.runpod.net --lang vietnamese
  python client.py --server ws://194.68.1.2:18765 --lang nepali

Supported languages: hindi, nepali, vietnamese, sinhala
        """,
    )
    parser.add_argument("--server", "-s", required=True,
        help="WebSocket server URL from RunPod")
    parser.add_argument("--lang", "-l", default="hindi",
        choices=["hindi", "nepali", "vietnamese", "sinhala"],
        help="Target language (default: hindi)")
    args = parser.parse_args()

    client = TranslationClient(args.server, args.lang)
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        print("\n\n  [Stopped] Goodbye!")
        client.stop()


if __name__ == "__main__":
    main()
