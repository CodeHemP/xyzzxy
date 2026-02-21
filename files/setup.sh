#!/bin/bash
# setup.sh — Run this ONCE on your RunPod 4090 pod
# It installs all dependencies and downloads all AI models.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh

set -e

echo "============================================================"
echo "  Setting Up Real-Time Translation Server"
echo "============================================================"
echo ""

# ── System dependencies ──
echo "[1/5] Installing system dependencies..."
apt-get update && apt-get install -y ffmpeg
echo "  ✓ System dependencies installed"

# ── Python virtual environment ──
echo ""
echo "[2/5] Creating Python virtual environment..."
python3 -m venv /workspace/translator-env
source /workspace/translator-env/bin/activate
echo "  ✓ Virtual environment created"

# ── Set LD_LIBRARY_PATH for cuDNN ──
echo ""
echo "[3/5] Configuring cuDNN library path..."
CUDNN_PATH="/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib"
if [ -d "$CUDNN_PATH" ]; then
    export LD_LIBRARY_PATH=$CUDNN_PATH:$LD_LIBRARY_PATH
    # Make it permanent in the venv
    echo "export LD_LIBRARY_PATH=$CUDNN_PATH:\$LD_LIBRARY_PATH" >> /workspace/translator-env/bin/activate
    echo "  ✓ cuDNN path configured"
else
    echo "  ⚠ cuDNN path not found at $CUDNN_PATH — may need manual fix"
fi

# ── Install Python packages ──
echo ""
echo "[4/5] Installing Python packages (this takes 3-5 minutes)..."
pip install --upgrade pip

# PyTorch from its own index
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Everything else from PyPI
pip install \
    faster-whisper==1.1.0 \
    ctranslate2==4.5.0 \
    transformers==4.47.0 \
    sentencepiece==0.2.0 \
    websockets==13.1 \
    numpy==1.26.4 \
    soundfile==0.12.1 \
    scipy==1.14.1 \
    edge-tts==6.1.18 \
    huggingface_hub

echo "  ✓ Python packages installed"

# ── Download AI models ──
echo ""
echo "[5/5] Downloading AI models (this takes 10-15 minutes)..."
cd /workspace/realtime-translator
python download_models.py

echo ""
echo "============================================================"
echo "  SETUP COMPLETE!"
echo ""
echo "  To start the server, run:"
echo "    source /workspace/translator-env/bin/activate"
echo "    cd /workspace/realtime-translator/server"
echo "    python server.py"
echo "============================================================"
