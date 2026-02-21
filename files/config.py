"""Configuration for the translation server."""

# Audio settings
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 64
CHANNELS = 1
DTYPE = "int16"

# VAD settings
VAD_THRESHOLD = 0.5
MIN_SPEECH_MS = 250
MAX_SPEECH_MS = 5000
SILENCE_AFTER_SPEECH_MS = 600

# Model paths
WHISPER_MODEL = "deepdml/faster-whisper-large-v3-turbo-ct2"
NLLB_MODEL_PATH = "/workspace/models/nllb-1.3b-ct2"
NLLB_TOKENIZER = "facebook/nllb-200-distilled-1.3B"

# NLLB language codes
LANGUAGE_CODES = {
    "hindi":      "hin_Deva",
    "nepali":     "npi_Deva",
    "vietnamese": "vie_Latn",
    "sinhala":    "sin_Sinh",
}

# TTS configuration
# "mms" = local GPU model (fast), "edge" = Microsoft Edge TTS (needs internet, great quality)
TTS_CONFIG = {
    "hindi":      {"engine": "mms",  "model_id": "facebook/mms-tts-hin"},
    "nepali":     {"engine": "edge", "voice": "ne-NP-SagarNeural"},
    "vietnamese": {"engine": "mms",  "model_id": "facebook/mms-tts-vie"},
    "sinhala":    {"engine": "edge", "voice": "si-LK-SameeraNeural"},
}

# Server
HOST = "0.0.0.0"
PORT = 8765
