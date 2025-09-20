# Base GPU RunPod (PyTorch 2.1.1 + CUDA 12.1 + Python 3.10)
FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Paquets système essentiels
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg git git-lfs \
        libsndfile1 libgomp1 \
        wget curl && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# Dépendances Python
# IMPORTANT:
# - Ne réinstalle PAS torch/torchvision/torchaudio (déjà dans l'image)
# - OpenCV headless pour éviter X/GL
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# Code
COPY . /app

# Environnement par défaut (surchargés par l'Endpoint si besoin)
ENV PORT=8000 \
    PORT_HEALTH=8001 \
    MODELS_DIR=/workspace/persistent/models \
    HF_HOME=/workspace/cache/huggingface

# start.sh lance health_app + API (HTTP mode)
RUN chmod +x /app/start.sh
CMD ["/app/start.sh"]
