#!/bin/bash
# setup_infinitetalk.sh — Idempotent, évite tout re-download si les modèles sont déjà montés
set -euo pipefail

echo "🚀 setup_infinitetalk.sh: start"

# 0) Détecter MODELS_DIR (pareil que start.sh / api_server)
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
for base in "${CANDIDATES[@]}"; do
  if [ -d "$base/Wan2.1-I2V-14B-480P" ] && \
     [ -d "$base/chinese-wav2vec2-base" ] && \
     [ -f "$base/InfiniteTalk/single/infinitetalk.safetensors" ]; then
    FOUND="$base"
    break
  fi
done

if [ -n "$FOUND" ]; then
  export MODELS_DIR="$FOUND"
else
  export MODELS_DIR="${MODELS_DIR:-/workspace/persistent/models}"
fi

echo "📁 MODELS_DIR=$MODELS_DIR"

# 1) Ne réinstalle pas torch/vision/audio (déjà fournis par l'image)
#    N'installe pas xformers ici non plus (inutile si tu restes avec torch 2.1.1)
#    Juste libs audio utiles si manquantes (gérées via requirements.txt normalement).
python - <<'PY'
try:
    import soundfile, librosa
    print("✅ Python audio deps ok")
except Exception as e:
    print("⚠️ Audio deps issue:", e)
PY

# 2) Vérifie la présence des modèles
missing=0
[ ! -d "${MODELS_DIR}/Wan2.1-I2V-14B-480P" ] && missing=1
[ ! -d "${MODELS_DIR}/chinese-wav2vec2-base" ] && missing=1
[ ! -f "${MODELS_DIR}/InfiniteTalk/single/infinitetalk.safetensors" ] && missing=1

if [ $missing -eq 0 ]; then
  echo "✅ All models present under ${MODELS_DIR}"
else
  echo "⚠️ Some models are missing under ${MODELS_DIR}"
  echo "   → This script will NOT auto-download to avoid filling ephemeral disk."
  echo "   → Please mount your Network Volume with the models at one of the known paths."
fi

# 3) Final import sanity check (informative)
echo "🧪 Final import checks..."
python - <<'PY'
import torch, fastapi
try:
    import soundfile, librosa
    print("✅ soundfile/librosa imported")
except Exception as e:
    print("⚠️ audio import issue:", e)
print(f"✅ Torch: {torch.__version__} CUDA ok? {torch.cuda.is_available()}")
print("✅ FastAPI ok")
PY

echo "✅ setup_infinitetalk.sh: done"
