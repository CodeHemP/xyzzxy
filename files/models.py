"""Model loading and management."""

import torch
import numpy as np
import ctranslate2
import tempfile
import subprocess
import os
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, VitsModel
import config

try:
    import soundfile as sf
except ImportError:
    sf = None


class ModelManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Models] Using device: {self.device}")
        self._load_vad()
        self._load_asr()
        self._load_mt()
        self._load_tts()
        print("[Models] All models loaded and ready!")

    def _load_vad(self):
        print("[Models] Loading Silero VAD...")
        self.vad_model, self.vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            trust_repo=True
        )
        (self.get_speech_timestamps, self.save_audio,
         self.read_audio, self.VADIterator, self.collect_chunks) = self.vad_utils
        print("[Models] VAD loaded.")

    def _load_asr(self):
        print("[Models] Loading faster-whisper (large-v3-turbo)...")
        self.whisper = WhisperModel(
            config.WHISPER_MODEL, device="cuda", compute_type="int8_float16",
        )
        print("[Models] ASR loaded.")

    def _load_mt(self):
        print("[Models] Loading NLLB-200 (CTranslate2)...")
        self.translator = ctranslate2.Translator(
            config.NLLB_MODEL_PATH, device="cuda", compute_type="int8_float16",
        )
        self.mt_tokenizer = AutoTokenizer.from_pretrained(config.NLLB_TOKENIZER)
        print("[Models] MT loaded.")

    def _load_tts(self):
        print("[Models] Loading TTS models...")
        self.mms_models = {}
        self.mms_tokenizers = {}
        self.tts_config = config.TTS_CONFIG

        for lang, cfg in self.tts_config.items():
            if cfg["engine"] == "mms":
                print(f"  Loading MMS-TTS for {lang}...")
                self.mms_models[lang] = VitsModel.from_pretrained(cfg["model_id"]).to(self.device)
                self.mms_tokenizers[lang] = AutoTokenizer.from_pretrained(cfg["model_id"])
            elif cfg["engine"] == "edge":
                print(f"  {lang} will use Edge-TTS (Microsoft) — no GPU model needed")

        print("[Models] TTS ready.")

    # ── ASR ──────────────────────────────────────────────

    def transcribe(self, audio_np: np.ndarray) -> str:
        """Transcribe audio numpy array (float32, 16kHz) to English text."""
        segments, info = self.whisper.transcribe(
            audio_np, language="en", beam_size=1, best_of=1,
            temperature=0.0, condition_on_previous_text=False, vad_filter=False,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text

    # ── Translation ──────────────────────────────────────

    def translate(self, text: str, target_lang: str) -> str:
        """Translate English text to target language using NLLB-200."""
        if not text:
            return ""
        target_code = config.LANGUAGE_CODES[target_lang]
        self.mt_tokenizer.src_lang = "eng_Latn"
        encoded = self.mt_tokenizer(text, return_tensors=None)
        tokens = self.mt_tokenizer.convert_ids_to_tokens(encoded["input_ids"])
        results = self.translator.translate_batch(
            [tokens], target_prefix=[[target_code]], beam_size=2, max_decoding_length=256,
        )
        output_tokens = results[0].hypotheses[0]
        if output_tokens and output_tokens[0] == target_code:
            output_tokens = output_tokens[1:]
        translated = self.mt_tokenizer.decode(
            self.mt_tokenizer.convert_tokens_to_ids(output_tokens), skip_special_tokens=True,
        )
        return translated

    # ── TTS ──────────────────────────────────────────────

    def synthesize(self, text: str, target_lang: str) -> tuple:
        """
        Synthesize text to audio.
        Returns (audio_np_float32, sample_rate)
        """
        if not text:
            return np.array([], dtype=np.float32), 16000

        cfg = self.tts_config.get(target_lang)
        if not cfg:
            return np.array([], dtype=np.float32), 16000

        if cfg["engine"] == "mms":
            return self._synthesize_mms(text, target_lang)
        elif cfg["engine"] == "edge":
            return self._synthesize_edge(text, cfg["voice"])
        else:
            return np.array([], dtype=np.float32), 16000

    def _synthesize_mms(self, text: str, target_lang: str) -> tuple:
        """MMS-TTS synthesis (local GPU, fast)."""
        tokenizer = self.mms_tokenizers[target_lang]
        model = self.mms_models[target_lang]
        inputs = tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = model(**inputs)
        waveform = output.waveform[0].cpu().numpy().astype(np.float32)
        return waveform, model.config.sampling_rate

    def _synthesize_edge(self, text: str, voice: str) -> tuple:
        """Edge-TTS synthesis (Microsoft, via subprocess)."""
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name

            # Run edge-tts as subprocess
            cmd = ["edge-tts", "--voice", voice, "--text", text, "--write-media", tmp_path]
            result = subprocess.run(cmd, capture_output=True, timeout=15)

            if result.returncode != 0:
                stderr_msg = result.stderr.decode()[:200] if result.stderr else "unknown error"
                print(f"  [TTS] edge-tts error: {stderr_msg}")
                return np.array([], dtype=np.float32), 16000

            # Read the audio file
            if sf is not None:
                audio_data, sample_rate = sf.read(tmp_path, dtype='float32')
            else:
                # Fallback: convert mp3 to wav using ffmpeg, then read
                wav_path = tmp_path.replace(".mp3", ".wav")
                subprocess.run(
                    ["ffmpeg", "-y", "-i", tmp_path, "-ar", "24000", "-ac", "1", "-f", "wav", wav_path],
                    capture_output=True, timeout=10
                )
                import wave
                with wave.open(wav_path, 'rb') as wf:
                    sample_rate = wf.getframerate()
                    frames = wf.readframes(wf.getnframes())
                    audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

            # Convert stereo to mono if needed
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)

            return audio_data.astype(np.float32), sample_rate

        except subprocess.TimeoutExpired:
            print("  [TTS] edge-tts timed out")
            return np.array([], dtype=np.float32), 16000
        except Exception as e:
            print(f"  [TTS] edge-tts exception: {e}")
            return np.array([], dtype=np.float32), 16000
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def get_tts_sample_rate(self, target_lang: str) -> int:
        """Get the sample rate for a TTS engine."""
        cfg = self.tts_config.get(target_lang, {})
        if cfg.get("engine") == "mms" and target_lang in self.mms_models:
            return self.mms_models[target_lang].config.sampling_rate
        return 24000  # edge-tts default sample rate
