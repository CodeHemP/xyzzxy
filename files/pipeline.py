"""The core ASR -> MT -> TTS pipeline."""

import numpy as np
import time
from models import ModelManager


class TranslationPipeline:
    """Processes audio utterances through the full translation pipeline."""

    def __init__(self, model_manager: ModelManager, target_lang: str = "hindi"):
        self.models = model_manager
        self.target_lang = target_lang

    def set_target_language(self, lang: str):
        """Change target language at runtime."""
        self.target_lang = lang

    def process_utterance(self, audio_np: np.ndarray) -> dict:
        """
        Full pipeline: audio -> transcription -> translation -> speech

        Returns dict with original_text, translated_text, audio_bytes,
        audio_sample_rate, and timings.
        """
        timings = {}

        # Step 1: ASR (Speech to Text)
        t0 = time.perf_counter()
        original_text = self.models.transcribe(audio_np)
        timings["asr_ms"] = (time.perf_counter() - t0) * 1000

        if not original_text:
            return None

        # Step 2: Translation (English -> Target Language)
        t0 = time.perf_counter()
        translated_text = self.models.translate(original_text, self.target_lang)
        timings["mt_ms"] = (time.perf_counter() - t0) * 1000

        if not translated_text:
            return None

        # Step 3: TTS (Text to Speech in target language)
        t0 = time.perf_counter()
        audio_out, tts_sr = self.models.synthesize(translated_text, self.target_lang)
        timings["tts_ms"] = (time.perf_counter() - t0) * 1000

        # Convert float32 audio to 16-bit PCM bytes
        if len(audio_out) == 0:
            audio_bytes = b""
        else:
            audio_int16 = (audio_out * 32767).clip(-32768, 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()

        timings["total_ms"] = timings["asr_ms"] + timings["mt_ms"] + timings["tts_ms"]

        return {
            "original_text": original_text,
            "translated_text": translated_text,
            "audio_bytes": audio_bytes,
            "audio_sample_rate": tts_sr,
            "timings": timings,
        }
