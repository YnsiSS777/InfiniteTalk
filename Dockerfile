# Base GPU RunPod (PyTorch, CUDA)
FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Paquets système
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg git git-lfs && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# Dépendances Python
# Ton requirements.txt doit inclure: fastapi, uvicorn[standard], requests, pydantic>=2
# + toutes libs InfiniteTalk (transformers, accelerate, safetensors, opencv-python-headless, librosa, soundfile, etc.)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Code
COPY . /app

# Config par défaut (tu peux aussi les définir dans l'Endpoint RunPod)
ENV PORT=8000
ENV PORT_HEALTH=8001
ENV MODELS_DIR=/workspace/persistent/models
ENV HF_HOME=/workspace/cache/huggingface

# Entrypoint -> HTTP mode
RUN chmod +x /app/start.sh
CMD ["/app/start.sh"]
