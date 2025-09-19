#!/bin/bash
# setup_infinitetalk.sh - Script d'installation corrigé pour RunPod

echo "🚀 Starting InfiniteTalk setup with all dependencies..."

# Installation des dépendances système CRITIQUES
apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    curl \
    libsndfile1 \
    libsndfile1-dev \
    libgomp1 \
    build-essential \
    python3-dev \
    sox \
    libsox-dev \
    libsox-fmt-all

echo "✅ System dependencies installed"

# Configuration Python et pip
cd /workspace/InfiniteTalk

# Mise à jour pip
pip install --upgrade pip

# Installation PyTorch et CUDA
echo "📦 Installing PyTorch..."
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Installation xformers
pip install -U xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121

# Installation des dépendances audio CRITIQUES
echo "🎵 Installing audio dependencies..."
pip install soundfile==0.12.1
pip install librosa==0.10.1
pip install scipy==1.11.4
pip install numpy==1.24.3
pip install audioread==3.0.0
pip install PySoundFile==0.9.0.post1

# Installation des autres dépendances InfiniteTalk
echo "📦 Installing InfiniteTalk dependencies..."
pip install misaki[en]
pip install ninja
pip install psutil
pip install packaging

# Installation Flash Attention (optionnel mais recommandé)
echo "⚡ Installing Flash Attention..."
pip install flash_attn==2.7.4.post1 || echo "Flash Attention installation failed, continuing with PyTorch attention"

# Installation des dépendances depuis requirements.txt
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# Installation FastAPI pour l'API
echo "🌐 Installing API dependencies..."
pip install fastapi==0.104.1
pip install uvicorn==0.24.0
pip install python-multipart==0.0.6
pip install aiofiles==23.2.1

# Installation Gradio si app.py est utilisé
if [ -f "app.py" ]; then
    pip install gradio==4.19.2
fi

# Vérifier l'installation de soundfile
echo "🔍 Verifying soundfile installation..."
python -c "import soundfile; print('✅ soundfile imported successfully')" || {
    echo "❌ soundfile import failed, trying alternative installation..."
    conda install -c conda-forge libsndfile -y
    pip install soundfile --force-reinstall
}

# Téléchargement des modèles si nécessaire
if [ ! -d "/workspace/weights" ] && [ ! -d "/workspace/persistent/weights" ]; then
    echo "📥 Downloading models..."
    pip install huggingface-hub
    
    # Créer le répertoire weights
    mkdir -p /workspace/weights
    
    echo "Downloading Wan2.1-I2V-14B (this may take 15-30 minutes)..."
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P \
        --local-dir /workspace/weights/Wan2.1-I2V-14B-480P \
        --resume-download
    
    echo "Downloading chinese-wav2vec2-base..."
    huggingface-cli download TencentGameMate/chinese-wav2vec2-base \
        --local-dir /workspace/weights/chinese-wav2vec2-base \
        --resume-download
    
    # Télécharger aussi le modèle safetensors
    huggingface-cli download TencentGameMate/chinese-wav2vec2-base model.safetensors \
        --revision refs/pr/1 \
        --local-dir /workspace/weights/chinese-wav2vec2-base
    
    echo "Downloading InfiniteTalk weights..."
    huggingface-cli download MeiGen-AI/InfiniteTalk \
        --local-dir /workspace/weights/InfiniteTalk \
        --resume-download
else
    echo "✅ Models already present, skipping download"
fi

# Créer les répertoires nécessaires
mkdir -p /workspace/uploads
mkdir -p /workspace/outputs
mkdir -p /workspace/cache

# Test final des imports
echo "🧪 Running final import tests..."
python -c "
import torch
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')

import soundfile
print('✅ soundfile imported')

import librosa
print('✅ librosa imported')

import fastapi
print('✅ FastAPI imported')

try:
    import flash_attn
    print('✅ Flash Attention available')
except:
    print('⚠️ Flash Attention not available, using PyTorch attention')

print('🎉 All critical dependencies imported successfully!')
" || {
    echo "❌ Some imports failed. Check the errors above."
    exit 1
}

echo "✅ Setup completed successfully!"
echo "📝 You can now run: python api_server.py or python app.py"
