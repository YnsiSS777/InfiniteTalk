#!/bin/bash
set -euo pipefail

export PORT="${PORT:-8000}"
export PORT_HEALTH="${PORT_HEALTH:-8001}"
export HF_HOME="${HF_HOME:-/workspace/cache/huggingface}"

# Candidats probables de montage (inclut $MODELS_DIR si fourni)
CANDIDATES=(
  "${MODELS_DIR:-/workspace/persistent/models}"
  "/workspace/persistent/models"
  "/workspace/persistent"
  "/workspace/data"
  "/data"
  "/workspace/models"
  "/models"
  "/mnt/models"
  "/runpod-volume"
)

FOUND=""
need_setup=1
for base in "${CANDIDATES[@]}"; do
  if [ -d "$base/Wan2.1-I2V-14B-480P" ] && \
     [ -d "$base/chinese-wav2vec2-base" ] && \
     [ -f "$base/InfiniteTalk/single/infinitetalk.safetensors" ]; then
    FOUND="$base"
    need_setup=0
    break
  fi
done

if [ -z "$FOUND" ]; then
  export MODELS_DIR="${MODELS_DIR:-/workspace/persistent/models}"
else
  export MODELS_DIR="$FOUND"
fi

echo "[start.sh] PORT=$PORT PORT_HEALTH=$PORT_HEALTH MODELS_DIR=$MODELS_DIR"

# 0) Préparer dossiers de travail
mkdir -p /workspace/uploads /workspace/outputs /workspace/cache

# 1) S'assurer que le repo InfiniteTalk est là où l'API l'attend
if [ ! -d "/workspace/InfiniteTalk" ]; then
  echo "[start.sh] Cloning InfiniteTalk into /workspace/InfiniteTalk"
  git clone --depth 1 https://github.com/MeiGen-AI/InfiniteTalk.git /workspace/InfiniteTalk
fi

# 2) Purger un éventuel /workspace/weights réel, puis symlink -> MODELS_DIR
if [ -d "/workspace/weights" ] && [ ! -L "/workspace/weights" ]; then
  echo "[start.sh] Removing stale /workspace/weights to free space"
  rm -rf /workspace/weights
fi
if [ ! -e "/workspace/weights" ]; then
  ln -s "$MODELS_DIR" /workspace/weights || true
  echo "[start.sh] Symlinked /workspace/weights -> $MODELS_DIR"
fi

# 3) Vérifier la présence des modèles (ne pas re-télécharger si déjà OK)
if [ $need_setup -eq 1 ]; then
  echo "[start.sh] Models present under ${MODELS_DIR}, skipping setup."
else
  echo "[start.sh] Some models are missing under MODELS_DIR=$MODELS_DIR"
  if [ -f "/app/setup_infinitetalk.sh" ]; then
    echo "[start.sh] Running setup_infinitetalk.sh (may download big files)"
    bash /app/setup_infinitetalk.sh || true
  else
    echo "[start.sh] WARNING: setup_infinitetalk.sh not found. Continuing anyway."
  fi
fi

# 4) Lancer health server (readiness) + API
echo "[start.sh] Launch health_app on :$PORT_HEALTH"
uvicorn api_server:health_app --host 0.0.0.0 --port "${PORT_HEALTH}" --log-level info &

echo "[start.sh] Launch main API on :$PORT"
uvicorn api_server:app --host 0.0.0.0 --port "${PORT}" --log-level info
