#!/bin/bash
# setup_infinitetalk.sh - Script d'installation pour RunPod

echo "🚀 Starting InfiniteTalk setup..."

# Installation des dépendances système
apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    curl \
    libsndfile1 \
    libgomp1

# Configuration Python
cd /workspace/InfiniteTalk

# Installation des dépendances Python
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -U xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121
pip install misaki[en]
pip install ninja psutil packaging
pip install flash_attn==2.7.4.post1

# Installation des dépendances depuis requirements.txt
pip install -r requirements.txt

# Installation FastAPI
pip install fastapi uvicorn python-multipart aiofiles

# Téléchargement des modèles
if [ ! -d "/workspace/weights" ]; then
    echo "📥 Downloading models..."
    pip install huggingface-hub
    
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P \
        --local-dir /workspace/weights/Wan2.1-I2V-14B-480P
    
    huggingface-cli download TencentGameMate/chinese-wav2vec2-base \
        --local-dir /workspace/weights/chinese-wav2vec2-base
    
    huggingface-cli download MeiGen-AI/InfiniteTalk \
        --local-dir /workspace/weights/InfiniteTalk
fi

echo "✅ Setup completed!"
