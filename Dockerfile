# Image GPU prête pour RunPod (PyTorch + CUDA + Python 3.10)
FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# On travaille sous /app (ton code + repo InfiniteTalk)
WORKDIR /app

# Outils système utiles
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg git git-lfs && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# Dépendances Python
# Assure-toi que requirements.txt contient au moins: runpod, soundfile, librosa, numpy (torch est déjà fourni par l'image)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# --- Code de TON endpoint serverless (incluant handler.py) ---
# Le fichier handler.py doit contenir:
#   if __name__ == "__main__":
#       runpod.serverless.start({"handler": handler})
COPY . /app

# --- (Optionnel) Bake du repo InfiniteTalk au build ---
# Si ton repo local ne contient pas déjà InfiniteTalk/, décommente la ligne suivante
# pour cloner la version officielle directement DANS l'image (au build, pas au runtime).
# RUN git clone --depth 1 https://github.com/MeiGen-AI/InfiniteTalk.git /app/InfiniteTalk

# Pas de port à exposer en mode serverless
# EXPOSE 7860

# Démarrage: lance directement le handler RunPod Serverless
# IMPORTANT: ton handler utilise WORKSPACE_DIR="/app/InfiniteTalk" et WEIGHTS_DIR="/workspace/weights"
CMD ["python", "/app/handler.py"]
