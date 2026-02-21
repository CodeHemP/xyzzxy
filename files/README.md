# Real-Time Lecture Audio Translation System

Translates English lecture audio to Hindi, Nepali, Vietnamese, or Sinhala in real-time.
Speaker talks → audience hears translation in their language through headphones.

## Architecture

```
LAPTOP (Client)                    RUNPOD 4090 (Server)
┌──────────────┐                   ┌──────────────────────┐
│ 🎤 Mic ──────┼───── audio ─────►│ Silero VAD           │
│              │                   │ → faster-whisper ASR  │
│ 🔊 Speaker ◄─┼── translated ◄──│ → NLLB-200 MT        │
│ 📝 Text      │   audio + text   │ → MMS-TTS / Edge-TTS │
└──────────────┘                   └──────────────────────┘
```

**Latency: ~350-500ms per utterance**

## Quick Start

### RunPod Server Setup (one-time)

```bash
# 1. Create a RunPod pod with RTX 4090, expose TCP port 8765

# 2. In the Web Terminal:
mkdir -p /workspace/realtime-translator/server
cd /workspace/realtime-translator/server

# 3. Download all files from this repo into /workspace/realtime-translator/server/
#    (wget each file from your GitHub raw URLs)

# 4. Run setup
chmod +x setup.sh
./setup.sh    # Takes ~20 minutes first time
```

### Start Server (every time)

```bash
source /workspace/translator-env/bin/activate
cd /workspace/realtime-translator/server
python server.py
```

### Laptop Client Setup (one-time)

```bash
# Mac
brew install portaudio
pip install pyaudio websockets numpy

# Linux
sudo apt install portaudio19-dev
pip install pyaudio websockets numpy

# Windows
pip install pyaudio websockets numpy
```

### Run Client

```bash
python client.py --server wss://YOUR-RUNPOD-URL --lang hindi
```

Languages: `hindi`, `nepali`, `vietnamese`, `sinhala`

## Files

### Server (RunPod)
| File | Purpose |
|------|---------|
| `setup.sh` | One-time setup script (installs everything) |
| `download_models.py` | Downloads all AI models |
| `config.py` | Configuration (ports, model paths, languages) |
| `models.py` | Model loading and inference (ASR, MT, TTS) |
| `vad.py` | Voice Activity Detection |
| `pipeline.py` | ASR → MT → TTS pipeline |
| `server.py` | WebSocket server (main entry point) |
| `requirements.txt` | Python dependencies |

### Client (Laptop)
| File | Purpose |
|------|---------|
| `client.py` | Mic capture + audio playback client |
| `requirements.txt` | Python dependencies |

## Models Used

| Component | Model | Engine | Languages |
|-----------|-------|--------|-----------|
| VAD | Silero VAD v5 | PyTorch | All |
| ASR | faster-whisper large-v3-turbo | CTranslate2 INT8 | English |
| Translation | NLLB-200-distilled-1.3B | CTranslate2 INT8 | All 4 targets |
| TTS (Hindi, Vietnamese) | MMS-TTS | PyTorch GPU | Hindi, Vietnamese |
| TTS (Nepali, Sinhala) | Edge-TTS (Microsoft) | Cloud API | Nepali, Sinhala |

## GPU Memory: ~7GB of 24GB (RTX 4090)
