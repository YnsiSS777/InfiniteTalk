#!/bin/bash
set -euo pipefail

export PORT="${PORT:-8000}"
export PORT_HEALTH="${PORT_HEALTH:-8001}"
export HF_HOME="${HF_HOME:-/workspace/cache/huggingface}"

# ---------------------------
# 1) Détection du dossier modèles
# ---------------------------
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

# ---------------------------
# 2) Préparer dossiers de travail
# ---------------------------
mkdir -p /workspace/uploads /workspace/outputs /workspace/cache

# ---------------------------
# 3) Cloner InfiniteTalk si absent
# ---------------------------
if [ ! -d "/workspace/InfiniteTalk" ]; then
  echo "[start.sh] Cloning InfiniteTalk into /workspace/InfiniteTalk"
  git clone --depth 1 https://github.com/MeiGen-AI/InfiniteTalk.git /workspace/InfiniteTalk
fi

# ---------------------------
# 4) Fix chemins & compatibilité
# ---------------------------
# Symlink /workspace/weights vers $MODELS_DIR
if [ -d "/workspace/weights" ] && [ ! -L "/workspace/weights" ]; then
  echo "[start.sh] Removing stale /workspace/weights to free space"
  rm -rf /workspace/weights
fi
if [ ! -e "/workspace/weights" ]; then
  ln -s "$MODELS_DIR" /workspace/weights || true
  echo "[start.sh] Symlinked /workspace/weights -> $MODELS_DIR"
fi

# Ajouter InfiniteTalk au PYTHONPATH
export PYTHONPATH="/workspace/InfiniteTalk:/workspace/InfiniteTalk/src:${PYTHONPATH:-}"

# Installer InfiniteTalk en editable si possible
if [ -f "/workspace/InfiniteTalk/setup.py" ] || [ -f "/workspace/InfiniteTalk/pyproject.toml" ]; then
  echo "[start.sh] Installing InfiniteTalk in editable mode"
  pip install -e /workspace/InfiniteTalk || true
else
  echo "[start.sh] No setup.py/pyproject.toml, skipping editable install"
fi

# ---------------------------
# 5) Setup des modèles si manquants
# ---------------------------
if [ $need_setup -eq 0 ]; then
  echo "[start.sh] ✅ Models present under ${MODELS_DIR}, skipping setup."
else
  echo "[start.sh] ⚠️ Some models missing under MODELS_DIR=$MODELS_DIR"
  if [ -f "/app/setup_infinitetalk.sh" ]; then
    echo "[start.sh] Running setup_infinitetalk.sh (may download big files)"
    bash /app/setup_infinitetalk.sh || true
  else
    echo "[start.sh] WARNING: setup_infinitetalk.sh not found. Continuing anyway."
  fi
fi

# ---------------------------
# 6) Lancer health server + API principale
# ---------------------------
echo "[start.sh] Launch health_app on :$PORT_HEALTH"
uvicorn api_server:health_app --host 0.0.0.0 --port "${PORT_HEALTH}" --log-level info &

echo "[start.sh] Launch main API on :$PORT"
uvicorn api_server:app --host 0.0.0.0 --port "${PORT}" --log-level info
