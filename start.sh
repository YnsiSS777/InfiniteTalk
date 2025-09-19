#!/bin/bash
set -euo pipefail

export PORT="${PORT:-8000}"
export PORT_HEALTH="${PORT_HEALTH:-8001}"
export MODELS_DIR="${MODELS_DIR:-/workspace/persistent/models}"
export HF_HOME="${HF_HOME:-/workspace/cache/huggingface}"

echo "[start.sh] PORT=$PORT PORT_HEALTH=$PORT_HEALTH MODELS_DIR=$MODELS_DIR"

# 0) Préparer dossiers de travail
mkdir -p /workspace/uploads /workspace/outputs /workspace/cache

# 1) S'assurer que le repo InfiniteTalk est là où l'API l'attend
if [ ! -d "/workspace/InfiniteTalk" ]; then
  echo "[start.sh] Cloning InfiniteTalk into /workspace/InfiniteTalk"
  git clone --depth 1 https://github.com/MeiGen-AI/InfiniteTalk.git /workspace/InfiniteTalk
fi

# 2) Vérifier la présence des modèles (ne pas re-télécharger si déjà OK)
need_setup=0
if [ ! -d "${MODELS_DIR}/Wan2.1-I2V-14B-480P" ]; then need_setup=1; fi
if [ ! -d "${MODELS_DIR}/chinese-wav2vec2-base" ]; then need_setup=1; fi
if [ ! -f "${MODELS_DIR}/InfiniteTalk/single/infinitetalk.safetensors" ]; then need_setup=1; fi

if [ $need_setup -eq 1 ]; then
  echo "[start.sh] Models missing under MODELS_DIR=$MODELS_DIR"
  if [ -f "/app/setup_infinitetalk.sh" ]; then
    echo "[start.sh] Running setup_infinitetalk.sh (may take time)"
    bash /app/setup_infinitetalk.sh
  else
    echo "[start.sh] WARNING: setup_infinitetalk.sh not found. Continuing anyway."
  fi
else
  echo "[start.sh] Models present under ${MODELS_DIR}, skipping setup."
fi

# 3) Lancer health server (readiness) + API
echo "[start.sh] Launch health_app on :$PORT_HEALTH"
uvicorn api_server:health_app --host 0.0.0.0 --port "${PORT_HEALTH}" --log-level info &

echo "[start.sh] Launch main API on :$PORT"
uvicorn api_server:app --host 0.0.0.0 --port "${PORT}" --log-level info
