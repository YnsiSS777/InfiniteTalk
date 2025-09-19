#!/usr/bin/env python
"""
RunPod Serverless Handler pour InfiniteTalk — version safe (cold start friendly)
"""

import os
import sys
import json
import base64
import tempfile
import subprocess
from typing import Dict, Any
import traceback

import runpod  # léger

# Constantes
WORKSPACE_DIR = "/app/InfiniteTalk"
WEIGHTS_DIR = "/workspace/weights"
UPLOAD_DIR = "/tmp/uploads"
OUTPUT_DIR = "/tmp/outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- UTIL --------------------------------------------------

def _lazy_import_audio_stack():
    # Importer seulement quand nécessaire (évite le cold start lent)
    import soundfile  # noqa
    import librosa    # noqa
    import numpy as np  # noqa

def _lazy_import_torch():
    import torch
    return torch

def _models_paths_ok() -> bool:
    # Ne pas appeler ça dans le healthcheck. Seulement à la génération.
    return all([
        os.path.exists(f"{WEIGHTS_DIR}/Wan2.1-I2V-14B-480P"),
        os.path.exists(f"{WEIGHTS_DIR}/chinese-wav2vec2-base"),
        os.path.exists(f"{WEIGHTS_DIR}/InfiniteTalk")
    ])

# --- HANDLERS ---------------------------------------------

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        job_input = (job or {}).get("input", {}) or {}
        endpoint = job_input.get("endpoint", "generate")

        # Warmup/health ultra rapide (0 I/O lourd)
        if endpoint in ("health", "warmup", "ping"):
            return {"status": "healthy", "ok": True}

        if endpoint == "status":
            return get_status(job_input)
        if endpoint == "generate":
            return generate_video(job_input)

        return {"error": f"Unknown endpoint: {endpoint}"}
    except Exception as e:
        print(f"❌ Error in handler: {e}")
        print(traceback.format_exc())
        return {"error": str(e), "traceback": traceback.format_exc()}

def generate_video(job_input: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # Imports lourds ici, pas au top-level
        _lazy_import_audio_stack()
        torch = _lazy_import_torch()

        image_b64 = job_input.get("image")
        video_b64 = job_input.get("video")
        audio_b64 = job_input.get("audio")

        if not audio_b64:
            return {"error": "Audio is required"}
        if not image_b64 and not video_b64:
            return {"error": "Either image or video is required"}

        # Vérifier les modèles maintenant (et seulement maintenant)
        if not _models_paths_ok():
            return {"error": "Models not found under /workspace/weights. Ensure weights are baked into the image or mounted before calling generate."}

        import uuid
        task_id = str(uuid.uuid4())

        # Écrire l'audio
        audio_path = os.path.join(UPLOAD_DIR, f"{task_id}_audio.wav")
        with open(audio_path, "wb") as f:
            f.write(base64.b64decode(audio_b64))

        # Écrire image/vidéo
        input_path = None
        if image_b64:
            input_path = os.path.join(UPLOAD_DIR, f"{task_id}_image.jpg")
            with open(input_path, "wb") as f:
                f.write(base64.b64decode(image_b64))
        else:
            input_path = os.path.join(UPLOAD_DIR, f"{task_id}_video.mp4")
            with open(input_path, "wb") as f:
                f.write(base64.b64decode(video_b64))

        # Config JSON pour ton script
        cfg = [{
            "input_type": "image" if image_b64 else "video",
            "input_path": input_path,
            "audio_path": audio_path,
            "output_path": os.path.join(OUTPUT_DIR, f"{task_id}_output.mp4")
        }]
        cfg_path = os.path.join(UPLOAD_DIR, f"{task_id}_config.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)

        # Paramètres
        resolution = str(job_input.get("resolution", "480"))
        mode = job_input.get("mode", "streaming")
        steps = int(job_input.get("steps", 40))
        audio_cfg = float(job_input.get("audio_cfg", 4.0))
        text_cfg = float(job_input.get("text_cfg", 5.0))

        # Commande
        cmd = [
            sys.executable, f"{WORKSPACE_DIR}/generate_infinitetalk.py",
            "--ckpt_dir", f"{WEIGHTS_DIR}/Wan2.1-I2V-14B-480P",
            "--wav2vec_dir", f"{WEIGHTS_DIR}/chinese-wav2vec2-base",
            "--infinitetalk_dir", f"{WEIGHTS_DIR}/InfiniteTalk/single/infinitetalk.safetensors",
            "--input_json", cfg_path,
            "--size", f"infinitetalk-{resolution}",
            "--sample_steps", str(steps),
            "--mode", mode,
            "--motion_frame", "9",
            "--sample_audio_guide_scale", str(audio_cfg),
            "--sample_text_guide_scale", str(text_cfg),
            "--num_persistent_param_in_dit", "0",
            "--save_file", os.path.join(OUTPUT_DIR, task_id)
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=WORKSPACE_DIR
        )
        if result.returncode != 0:
            # Renvoyer les 600 1ers chars pour debug
            return {"error": "Generation failed", "stderr": result.stderr[:600]}

        # Chercher la sortie
        out_path_candidates = [
            os.path.join(OUTPUT_DIR, f"{task_id}_0.mp4"),
            os.path.join(OUTPUT_DIR, f"{task_id}.mp4"),
            os.path.join(OUTPUT_DIR, f"{task_id}_output.mp4"),
        ]
        output_path = next((p for p in out_path_candidates if os.path.exists(p)), None)
        if not output_path:
            return {"error": "Output file not found after generation."}

        with open(output_path, "rb") as f:
            video_b64_out = base64.b64encode(f.read()).decode("utf-8")

        # Cleanup best effort
        for p in {audio_path, input_path, cfg_path, output_path}:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

        return {"status": "completed", "task_id": task_id, "video_base64": video_b64_out}

    except Exception as e:
        print(f"❌ Error in generate_video: {e}")
        print(traceback.format_exc())
        return {"error": str(e), "traceback": traceback.format_exc()}

def get_status(job_input: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_id": job_input.get("task_id", "unknown"),
        "status": "completed",
        "progress": 100
    }

if __name__ == "__main__":
    # Ne rien faire de bloquant ici.
    # Pas de git clone, pas de setup, pas d'import torch ici.
    print("✅ Starting RunPod serverless handler (light init)")
    runpod.serverless.start({"handler": handler})
