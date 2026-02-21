"""
Download all models needed for the translation server.
Run this once — models are cached in /workspace/ and persist across pod restarts.

Usage:
    python download_models.py
"""
import os
import sys

print("=" * 60)
print("  Downloading AI Models (this takes 10-15 minutes)")
print("=" * 60)

# ── Step 1: Download Silero VAD ──────────────────────────
print("\n[1/4] Downloading Silero VAD (voice activity detection)...")
import torch
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    trust_repo=True
)
print("  ✓ Silero VAD ready")

# ── Step 2: Download faster-whisper ──────────────────────
print("\n[2/4] Downloading faster-whisper large-v3-turbo (speech recognition)...")
print("  This is the largest download (~3GB). Please wait...")
from faster_whisper import WhisperModel
import numpy as np

whisper = WhisperModel(
    "deepdml/faster-whisper-large-v3-turbo-ct2",
    device="cuda",
    compute_type="int8_float16",
)
# Quick test
print("  Testing ASR... ", end="")
test_audio = np.zeros(16000, dtype=np.float32)
segments, _ = whisper.transcribe(test_audio, language="en")
list(segments)
print("works!")
print("  ✓ faster-whisper ready")
del whisper
torch.cuda.empty_cache()

# ── Step 3: Download & Convert NLLB-200 ──────────────────
print("\n[3/4] Downloading NLLB-200 translation model...")
nllb_ct2_path = "/workspace/models/nllb-1.3b-ct2"

if os.path.exists(os.path.join(nllb_ct2_path, "model.bin")):
    print("  ✓ NLLB CT2 model already exists, skipping conversion")
else:
    print("  Downloading and converting to CTranslate2 format...")
    print("  (This is the slowest step — up to ~10 minutes)")

    # Remove partial directory if it exists
    if os.path.exists(nllb_ct2_path):
        import shutil
        shutil.rmtree(nllb_ct2_path)

    import ctranslate2
    converter = ctranslate2.converters.TransformersConverter(
        "facebook/nllb-200-distilled-1.3B"
    )
    converter.convert(nllb_ct2_path, quantization="int8_float16", force=True)
    print("  ✓ NLLB-200 converted and ready")

# Test translation
print("  Testing translation... ", end="")
from transformers import AutoTokenizer
import ctranslate2

translator = ctranslate2.Translator(nllb_ct2_path, device="cuda", compute_type="int8_float16")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")
tokenizer.src_lang = "eng_Latn"
encoded = tokenizer("Hello, how are you?", return_tensors=None)
tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
results = translator.translate_batch([tokens], target_prefix=[["hin_Deva"]], beam_size=2)
output_tokens = results[0].hypotheses[0][1:]
translated = tokenizer.decode(tokenizer.convert_tokens_to_ids(output_tokens), skip_special_tokens=True)
print(f'"{translated}"')
print("  ✓ Translation works!")
del translator
torch.cuda.empty_cache()

# ── Step 4: Download TTS Models ──────────────────────────
print("\n[4/4] Downloading TTS models...")
from transformers import VitsModel, AutoTokenizer as AT2

# Only download MMS models (Hindi and Vietnamese)
# Nepali and Sinhala use Edge-TTS (no download needed)
mms_models = {
    "hindi":      "facebook/mms-tts-hin",
    "vietnamese": "facebook/mms-tts-vie",
}

for lang, model_id in mms_models.items():
    print(f"  Downloading MMS-TTS for {lang}... ", end="")
    VitsModel.from_pretrained(model_id)
    AT2.from_pretrained(model_id)
    print("✓")

print("  Nepali: will use Edge-TTS (Microsoft) — no download needed ✓")
print("  Sinhala: will use Edge-TTS (Microsoft) — no download needed ✓")

# ── Test Edge-TTS availability ───────────────────────────
print("\n[Bonus] Testing Edge-TTS availability...")
import subprocess
result = subprocess.run(["edge-tts", "--list-voices"], capture_output=True, timeout=10)
if result.returncode == 0:
    output = result.stdout.decode()
    nepali_ok = "ne-NP" in output
    sinhala_ok = "si-LK" in output
    print(f"  Nepali voice (ne-NP): {'✓ available' if nepali_ok else '✗ not found'}")
    print(f"  Sinhala voice (si-LK): {'✓ available' if sinhala_ok else '✗ not found'}")
else:
    print("  ⚠ edge-tts not working. Install with: pip install edge-tts")

print("\n" + "=" * 60)
print("  ALL MODELS READY!")
print("  Start the server with:")
print("    cd /workspace/realtime-translator/server")
print("    python server.py")
print("=" * 60)
