"""WebSocket server — runs on RunPod 4090."""

import asyncio
import json
import base64
import time
import websockets
from models import ModelManager
from vad import AudioSegmenter
from pipeline import TranslationPipeline
import config


class TranslationServer:
    def __init__(self):
        print("=" * 60)
        print("  Real-Time Lecture Translation Server")
        print("=" * 60)
        print()
        self.models = ModelManager()

    async def handle_client(self, websocket):
        """Handle a single client connection."""
        client_id = id(websocket)
        print(f"\n[Server] Client {client_id} connected")

        segmenter = AudioSegmenter(self.models)
        pipeline = TranslationPipeline(self.models, target_lang="hindi")
        utterance_count = 0

        try:
            async for message in websocket:
                # Handle control messages (JSON text)
                if isinstance(message, str):
                    try:
                        msg = json.loads(message)
                        if msg.get("type") == "config":
                            lang = msg.get("target_language", "hindi")
                            pipeline.set_target_language(lang)
                            await websocket.send(json.dumps({
                                "type": "config_ack",
                                "target_language": lang,
                            }))
                            print(f"[Server] Client {client_id} set language: {lang}")
                        elif msg.get("type") == "reset":
                            segmenter.reset()
                            print(f"[Server] Client {client_id} reset VAD state")
                        continue
                    except json.JSONDecodeError:
                        pass

                # Handle audio data (binary bytes)
                if isinstance(message, bytes):
                    utterances = segmenter.process_chunk(message)

                    for utterance_audio in utterances:
                        utterance_count += 1
                        t_start = time.perf_counter()

                        result = pipeline.process_utterance(utterance_audio)

                        if result is None:
                            continue

                        total_time = (time.perf_counter() - t_start) * 1000

                        # Send text result
                        text_msg = json.dumps({
                            "type": "translation",
                            "utterance_id": utterance_count,
                            "original_text": result["original_text"],
                            "translated_text": result["translated_text"],
                            "timings": result["timings"],
                            "pipeline_total_ms": round(total_time, 1),
                        })
                        await websocket.send(text_msg)

                        # Send audio result
                        if result["audio_bytes"]:
                            audio_msg = json.dumps({
                                "type": "audio",
                                "utterance_id": utterance_count,
                                "sample_rate": result["audio_sample_rate"],
                                "audio_b64": base64.b64encode(
                                    result["audio_bytes"]
                                ).decode("ascii"),
                            })
                            await websocket.send(audio_msg)

                        # Log to terminal
                        t = result["timings"]
                        print(
                            f"  [{utterance_count}] "
                            f"ASR={t['asr_ms']:.0f}ms "
                            f"MT={t['mt_ms']:.0f}ms "
                            f"TTS={t['tts_ms']:.0f}ms "
                            f"Total={total_time:.0f}ms"
                        )
                        print(f'    EN: "{result["original_text"][:80]}"')
                        print(f'    ->: "{result["translated_text"][:80]}"')
                        print()

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            print(f"[Server] Client {client_id} disconnected")

    async def start(self):
        print(f"\n[Server] Starting on ws://{config.HOST}:{config.PORT}")
        print("[Server] Waiting for clients...\n")
        async with websockets.serve(
            self.handle_client, config.HOST, config.PORT,
            max_size=2**20, ping_interval=20, ping_timeout=60,
        ):
            await asyncio.Future()  # Run forever


if __name__ == "__main__":
    server = TranslationServer()
    asyncio.run(server.start())
