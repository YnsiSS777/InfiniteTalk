#!/usr/bin/env python
"""
RunPod Serverless Handler pour InfiniteTalk
"""

import runpod
import os
import sys
import json
import base64
import tempfile
import subprocess
from typing import Dict, Any, Optional
import traceback

# Configuration des chemins
WORKSPACE_DIR = "/workspace/InfiniteTalk"
WEIGHTS_DIR = "/workspace/weights"
UPLOAD_DIR = "/tmp/uploads"
OUTPUT_DIR = "/tmp/outputs"

# Créer les répertoires
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Vérifier les imports critiques au démarrage
try:
    print("🔍 Checking critical imports...")
    import torch
    import soundfile
    import librosa
    import numpy as np
    print("✅ All critical imports successful")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Installing missing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "soundfile", "librosa", "torch"])
    sys.exit(1)

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handler principal pour RunPod
    
    Args:
        job: Dictionary contenant l'input du job
        
    Returns:
        Dictionary avec l'output ou l'erreur
    """
    try:
        print(f"📨 Received job: {job.get('id', 'unknown')}")
        
        # Récupérer l'input
        job_input = job.get("input", {})
        
        # Router vers la bonne fonction selon l'endpoint
        endpoint = job_input.get("endpoint", "generate")
        
        if endpoint == "health":
            return health_check()
        elif endpoint == "generate":
            return generate_video(job_input)
        elif endpoint == "status":
            return get_status(job_input)
        else:
            return {"error": f"Unknown endpoint: {endpoint}"}
            
    except Exception as e:
        print(f"❌ Error in handler: {str(e)}")
        print(traceback.format_exc())
        return {"error": str(e), "traceback": traceback.format_exc()}

def health_check() -> Dict[str, Any]:
    """Vérification de santé du service"""
    try:
        import torch
        
        # Vérifier que les modèles existent
        models_exist = all([
            os.path.exists(f"{WEIGHTS_DIR}/Wan2.1-I2V-14B-480P"),
            os.path.exists(f"{WEIGHTS_DIR}/chinese-wav2vec2-base"),
            os.path.exists(f"{WEIGHTS_DIR}/InfiniteTalk")
        ])
        
        return {
            "status": "healthy",
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "models_loaded": models_exist,
            "workspace_exists": os.path.exists(WORKSPACE_DIR)
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

def generate_video(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Génère une vidéo avec InfiniteTalk
    """
    try:
        print("🎬 Starting video generation...")
        
        # Décoder les fichiers base64
        image_base64 = job_input.get("image")
        video_base64 = job_input.get("video")
        audio_base64 = job_input.get("audio")
        
        if not audio_base64:
            return {"error": "Audio is required"}
        
        if not image_base64 and not video_base64:
            return {"error": "Either image or video is required"}
        
        # Créer des fichiers temporaires
        import uuid
        task_id = str(uuid.uuid4())
        
        # Sauvegarder l'audio
        audio_path = os.path.join(UPLOAD_DIR, f"{task_id}_audio.wav")
        with open(audio_path, "wb") as f:
            f.write(base64.b64decode(audio_base64))
        print(f"✅ Audio saved: {audio_path}")
        
        # Sauvegarder l'image ou vidéo
        input_path = None
        if image_base64:
            input_path = os.path.join(UPLOAD_DIR, f"{task_id}_image.jpg")
            with open(input_path, "wb") as f:
                f.write(base64.b64decode(image_base64))
            print(f"✅ Image saved: {input_path}")
        elif video_base64:
            input_path = os.path.join(UPLOAD_DIR, f"{task_id}_video.mp4")
            with open(input_path, "wb") as f:
                f.write(base64.b64decode(video_base64))
            print(f"✅ Video saved: {input_path}")
        
        # Créer le fichier de configuration
        config = [{
            "input_type": "image" if image_base64 else "video",
            "input_path": input_path,
            "audio_path": audio_path,
            "output_path": os.path.join(OUTPUT_DIR, f"{task_id}_output.mp4")
        }]
        
        config_path = os.path.join(UPLOAD_DIR, f"{task_id}_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)
        
        # Paramètres de génération
        resolution = job_input.get("resolution", "480")
        mode = job_input.get("mode", "streaming")
        steps = job_input.get("steps", 40)
        audio_cfg = job_input.get("audio_cfg", 4.0)
        text_cfg = job_input.get("text_cfg", 5.0)
        
        # Construire la commande InfiniteTalk
        cmd = [
            "python", f"{WORKSPACE_DIR}/generate_infinitetalk.py",
            "--ckpt_dir", f"{WEIGHTS_DIR}/Wan2.1-I2V-14B-480P",
            "--wav2vec_dir", f"{WEIGHTS_DIR}/chinese-wav2vec2-base",
            "--infinitetalk_dir", f"{WEIGHTS_DIR}/InfiniteTalk/single/infinitetalk.safetensors",
            "--input_json", config_path,
            "--size", f"infinitetalk-{resolution}",
            "--sample_steps", str(steps),
            "--mode", mode,
            "--motion_frame", "9",
            "--sample_audio_guide_scale", str(audio_cfg),
            "--sample_text_guide_scale", str(text_cfg),
            "--num_persistent_param_in_dit", "0",
            "--save_file", os.path.join(OUTPUT_DIR, task_id)
        ]
        
        print(f"🚀 Running command: {' '.join(cmd)}")
        
        # Exécuter la commande
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=WORKSPACE_DIR
        )
        
        if result.returncode != 0:
            print(f"❌ InfiniteTalk error: {result.stderr}")
            return {"error": f"Generation failed: {result.stderr[:500]}"}
        
        # Rechercher le fichier de sortie
        output_path = os.path.join(OUTPUT_DIR, f"{task_id}_0.mp4")
        if not os.path.exists(output_path):
            output_path = os.path.join(OUTPUT_DIR, f"{task_id}.mp4")
        
        if not os.path.exists(output_path):
            return {"error": "Output file not found"}
        
        # Encoder la vidéo en base64
        with open(output_path, "rb") as f:
            video_base64_output = base64.b64encode(f.read()).decode('utf-8')
        
        # Nettoyer les fichiers temporaires
        for file in [audio_path, input_path, config_path, output_path]:
            if file and os.path.exists(file):
                os.remove(file)
        
        print("✅ Video generation completed successfully")
        
        return {
            "video_base64": video_base64_output,
            "task_id": task_id,
            "status": "completed"
        }
        
    except Exception as e:
        print(f"❌ Error in generate_video: {str(e)}")
        print(traceback.format_exc())
        return {"error": str(e), "traceback": traceback.format_exc()}

def get_status(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Obtenir le statut d'une tâche (simplified)"""
    task_id = job_input.get("task_id")
    if not task_id:
        return {"error": "task_id is required"}
    
    # Pour RunPod serverless, on retourne directement completed ou error
    # car le processing est synchrone
    return {
        "task_id": task_id,
        "status": "completed",
        "progress": 100
    }

# Point d'entrée RunPod
if __name__ == "__main__":
    print("🚀 Starting RunPod InfiniteTalk Handler")
    print(f"Working directory: {os.getcwd()}")
    print(f"Files in /workspace: {os.listdir('/workspace')}")
    
    # Vérifier que InfiniteTalk est présent
    if not os.path.exists(WORKSPACE_DIR):
        print(f"❌ InfiniteTalk not found at {WORKSPACE_DIR}")
        print("Cloning repository...")
        subprocess.run([
            "git", "clone", 
            "https://github.com/YnsiSS777/InfiniteTalk.git",
            WORKSPACE_DIR
        ])
    
    # Vérifier les modèles
    if not os.path.exists(WEIGHTS_DIR):
        print("⚠️ Models not found. Running setup...")
        setup_script = os.path.join(WORKSPACE_DIR, "setup_infinitetalk.sh")
        if os.path.exists(setup_script):
            subprocess.run(["bash", setup_script], cwd=WORKSPACE_DIR)
    
    print("✅ Handler ready, starting RunPod serverless...")
    runpod.serverless.start({"handler": handler})
