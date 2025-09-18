#!/bin/bash
# setup_infinitetalk.sh - Version avec stockage persistent

echo "🚀 Starting InfiniteTalk setup..."

# Définir les chemins
PERSISTENT_DIR="/workspace/persistent"
WEIGHTS_DIR="${PERSISTENT_DIR}/weights"
LOCAL_WEIGHTS="/workspace/weights"

# Créer un lien symbolique si les modèles sont sur le volume persistent
if [ -d "${WEIGHTS_DIR}" ]; then
    echo "✅ Modèles trouvés sur le volume persistent"
    ln -sf ${WEIGHTS_DIR} ${LOCAL_WEIGHTS}
else
    echo "📥 Première installation - Téléchargement des modèles..."
    mkdir -p ${WEIGHTS_DIR}
    
    # Installation des dépendances
    apt-get update && apt-get install -y ffmpeg git wget curl libsndfile1
    pip install huggingface-hub
    
    # Télécharger les modèles dans le volume persistent
    echo "Téléchargement de Wan2.1-I2V-14B (30GB)..."
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P \
        --local-dir ${WEIGHTS_DIR}/Wan2.1-I2V-14B-480P \
        --resume-download
    
    echo "Téléchargement de chinese-wav2vec2-base..."
    huggingface-cli download TencentGameMate/chinese-wav2vec2-base \
        --local-dir ${WEIGHTS_DIR}/chinese-wav2vec2-base \
        --resume-download
    
    echo "Téléchargement d'InfiniteTalk weights..."
    huggingface-cli download MeiGen-AI/InfiniteTalk \
        --local-dir ${WEIGHTS_DIR}/InfiniteTalk \
        --resume-download
    
    # Créer le lien symbolique
    ln -sf ${WEIGHTS_DIR} ${LOCAL_WEIGHTS}
    
    echo "✅ Modèles téléchargés et sauvegardés dans le volume persistent"
fi

# Installation des dépendances Python (toujours nécessaire)
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -U xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart aiofiles

echo "✅ Setup completed!"
