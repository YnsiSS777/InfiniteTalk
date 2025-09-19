# Image GPU RunPod officielle (torch déjà présent)
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
# Assure-toi que requirements.txt contient: fastapi, uvicorn[standard], requests, pydantic>=2
# + toutes les libs nécessaires à InfiniteTalk (transformers, accelerate, safetensors, opencv-python-headless, librosa, soundfile, etc.)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Code
COPY . /app

# Ports (RunPod lira ces env vars dans la conf de l'endpoint)
ENV PORT=8000
ENV PORT_HEALTH=8001

# Entrypoint -> HTTP mode (PAS handler.py)
RUN chmod +x /app/start.sh
CMD ["/app/start.sh"]
