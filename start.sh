#!/bin/bash
set -euo pipefail

export PORT="${PORT:-8000}"
export PORT_HEALTH="${PORT_HEALTH:-8001}"

echo "[start.sh] Using PORT=$PORT, PORT_HEALTH=$PORT_HEALTH"

# 1) S'assurer que InfiniteTalk est présent au bon endroit
if [ ! -d "/workspace/InfiniteTalk" ]; then
  echo "[start.sh] /workspace/InfiniteTalk manquant -> clone"
  git clone --depth 1 https://github.com/MeiGen-AI/InfiniteTalk.git /workspace/InfiniteTalk
fi

# 2) Installer les poids si absents
if [ ! -d "/workspace/weights" ]; then
  echo "[start.sh] /workspace/weights manquant -> setup"
  if [ -f "/app/setup_infinitetalk.sh" ]; then
    bash /app/setup_infinitetalk.sh
  else
    echo "[start.sh] WARNING: setup_infinitetalk.sh introuvable, on continue..."
  fi
fi

# 3) Lancer health server (readiness) + API
echo "[start.sh] Launch health_app on :$PORT_HEALTH"
uvicorn api_server:health_app --host 0.0.0.0 --port "${PORT_HEALTH}" --log-level info &

echo "[start.sh] Launch main API on :$PORT"
uvicorn api_server:app --host 0.0.0.0 --port "${PORT}" --log-level info
