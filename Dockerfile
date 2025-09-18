# Image GPU prête pour Runpod (PyTorch + CUDA + Python 3.10)
FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Paquets système utiles (ffmpeg fréquent pour audio/vidéo)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg git git-lfs && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# (Facultatif) si tu as un requirements.txt, on l’installe d'abord pour profiter du cache
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Copie du code
COPY . /app

# Si besoin d’épingler numpy/opencv, dé-commente :
# RUN pip install --no-cache-dir "numpy==1.26.*" "opencv-python-headless"

# Expose le port de ton app (Gradio/FastAPI)
EXPOSE 7860

# ✏️ REMPLACE app.py par ton script de lancement
# Exemples :
#   Gradio:  python app.py
#   FastAPI: uvicorn main:app --host 0.0.0.0 --port 7860
CMD ["python", "app.py"]
