# api_server.py
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import uuid
import os
import subprocess
import asyncio
from typing import Optional
import tempfile
import shutil
from datetime import datetime
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="InfiniteTalk API", version="1.0.0")

# Configuration CORS pour Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Remplacez par votre domaine en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Répertoires de travail
UPLOAD_DIR = "/workspace/uploads"
OUTPUT_DIR = "/workspace/outputs"

# Vérifier d'abord le stockage persistent, sinon local
PERSISTENT_WEIGHTS = "/workspace/persistent/weights"
LOCAL_WEIGHTS = "/workspace/weights"
WEIGHTS_DIR = PERSISTENT_WEIGHTS if os.path.exists(PERSISTENT_WEIGHTS) else LOCAL_WEIGHTS

# Créer les répertoires s'ils n'existent pas
for dir_path in [UPLOAD_DIR, OUTPUT_DIR, WEIGHTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# File d'attente pour gérer les tâches
processing_queue = {}

# Configuration InfiniteTalk
INFINITETALK_CONFIG = {
    "ckpt_dir": f"{WEIGHTS_DIR}/Wan2.1-I2V-14B-480P",
    "wav2vec_dir": f"{WEIGHTS_DIR}/chinese-wav2vec2-base",
    "infinitetalk_dir": f"{WEIGHTS_DIR}/InfiniteTalk/single/infinitetalk.safetensors",
    "motion_frame": 9,
    "num_persistent_param_in_dit": 0  # Pour économiser la VRAM
}

# Vérification au démarrage
@app.on_event("startup")
async def startup_event():
    """Vérification des modèles requis au démarrage"""
    logger.info(f"🚀 Démarrage de l'API InfiniteTalk")
    logger.info(f"📁 Répertoire des poids utilisé: {WEIGHTS_DIR}")
    
    required_models = [
        INFINITETALK_CONFIG["ckpt_dir"],
        INFINITETALK_CONFIG["wav2vec_dir"],
        INFINITETALK_CONFIG["infinitetalk_dir"]
    ]
    
    missing_models = [m for m in required_models if not os.path.exists(m)]
    
    if missing_models:
        logger.error(f"⚠️ Modèles manquants: {missing_models}")
        logger.error("Exécutez setup_infinitetalk.sh pour télécharger les modèles")
        logger.warning("L'API peut ne pas fonctionner correctement sans ces modèles")
    else:
        logger.info("✅ Tous les modèles sont chargés et prêts")
    
    # Afficher l'espace disque disponible
    try:
        disk_usage = shutil.disk_usage(WEIGHTS_DIR)
        free_gb = disk_usage.free / (1024**3)
        total_gb = disk_usage.total / (1024**3)
        logger.info(f"💾 Espace disque: {free_gb:.1f}GB libre sur {total_gb:.1f}GB total")
    except Exception as e:
        logger.warning(f"Impossible de vérifier l'espace disque: {e}")

@app.get("/")
async def root():
    """Endpoint de base pour vérifier que l'API fonctionne"""
    return {
        "message": "InfiniteTalk API is running",
        "version": "1.0.0",
        "weights_dir": WEIGHTS_DIR,
        "endpoints": {
            "health": "/health",
            "generate": "/generate",
            "status": "/status/{task_id}",
            "download": "/download/{task_id}"
        }
    }

@app.get("/health")
async def health_check():
    """Vérification de l'état du service"""
    try:
        # Vérifier que les modèles sont présents
        models_exist = all([
            os.path.exists(INFINITETALK_CONFIG["ckpt_dir"]),
            os.path.exists(INFINITETALK_CONFIG["wav2vec_dir"]),
            os.path.exists(INFINITETALK_CONFIG["infinitetalk_dir"])
        ])
        
        # Vérifier l'espace disque disponible
        disk_usage = shutil.disk_usage("/workspace")
        free_gb = disk_usage.free / (1024**3)
        
        return {
            "status": "healthy",
            "gpu_available": True,
            "models_loaded": models_exist,
            "weights_dir": WEIGHTS_DIR,
            "free_disk_gb": round(free_gb, 2),
            "queue_size": len(processing_queue)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.post("/generate")
async def generate_video(
    background_tasks: BackgroundTasks,
    image: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    audio: UploadFile = File(...),
    resolution: str = Form("480"),  # 480 ou 720
    mode: str = Form("streaming"),  # streaming ou clip
    steps: int = Form(40),
    audio_cfg: float = Form(4.0),
    text_cfg: float = Form(5.0),
    use_lora: bool = Form(False),
    lora_scale: float = Form(1.0)
):
    """
    Génère une vidéo avec synchronisation labiale
    
    Args:
        image: Image source (JPG/PNG)
        video: Vidéo source (MP4)
        audio: Audio à synchroniser (WAV/MP3)
        resolution: Résolution de sortie (480 ou 720)
        mode: Mode de génération (streaming ou clip)
        steps: Nombre d'étapes de génération (20-50)
        audio_cfg: Guide scale pour l'audio (3-5)
        text_cfg: Guide scale pour le texte (4-6)
        use_lora: Utiliser LoRA pour accélération
        lora_scale: Échelle LoRA (0.5-1.5)
    """
    
    # Validation des entrées
    if not image and not video:
        raise HTTPException(
            status_code=400,
            detail="Une image ou vidéo source est requise"
        )
    
    if resolution not in ["480", "720"]:
        raise HTTPException(
            status_code=400,
            detail="La résolution doit être 480 ou 720"
        )
    
    # Génération d'ID unique pour la tâche
    task_id = str(uuid.uuid4())
    logger.info(f"Nouvelle tâche créée: {task_id}")
    
    try:
        # Sauvegarde des fichiers uploadés
        audio_path = f"{UPLOAD_DIR}/{task_id}_audio.wav"
        with open(audio_path, "wb") as f:
            content = await audio.read()
            f.write(content)
        logger.info(f"Audio sauvegardé: {audio_path}")
        
        input_type = "image"
        input_path = None
        
        if image:
            # Sauvegarder l'image
            file_ext = image.filename.split('.')[-1] if image.filename else 'jpg'
            input_path = f"{UPLOAD_DIR}/{task_id}_image.{file_ext}"
            with open(input_path, "wb") as f:
                content = await image.read()
                f.write(content)
            input_type = "image"
            logger.info(f"Image sauvegardée: {input_path}")
            
        elif video:
            # Sauvegarder la vidéo
            input_path = f"{UPLOAD_DIR}/{task_id}_video.mp4"
            with open(input_path, "wb") as f:
                content = await video.read()
                f.write(content)
            input_type = "video"
            logger.info(f"Vidéo sauvegardée: {input_path}")
        
        # Création du fichier JSON de configuration pour InfiniteTalk
        config = {
            "input_type": input_type,
            "input_path": input_path,
            "audio_path": audio_path,
            "output_path": f"{OUTPUT_DIR}/{task_id}_output.mp4"
        }
        
        config_path = f"{UPLOAD_DIR}/{task_id}_config.json"
        with open(config_path, "w") as f:
            json.dump([config], f)
        
        # Ajout à la file d'attente
        processing_queue[task_id] = {
            "status": "queued",
            "progress": 0,
            "output": None,
            "error": None,
            "created_at": datetime.now().isoformat(),
            "config": {
                "resolution": resolution,
                "mode": mode,
                "steps": steps,
                "audio_cfg": audio_cfg,
                "text_cfg": text_cfg
            }
        }
        
        # Lancement du traitement en arrière-plan
        background_tasks.add_task(
            process_video,
            task_id,
            config_path,
            resolution,
            mode,
            steps,
            audio_cfg,
            text_cfg,
            use_lora,
            lora_scale
        )
        
        return {
            "task_id": task_id,
            "status": "queued",
            "message": "Vidéo en cours de génération",
            "estimated_time_seconds": 60 if resolution == "480" else 90
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la création de la tâche: {e}")
        # Nettoyer en cas d'erreur
        if task_id in processing_queue:
            del processing_queue[task_id]
        raise HTTPException(status_code=500, detail=str(e))

async def process_video(
    task_id: str,
    config_path: str,
    resolution: str,
    mode: str,
    steps: int,
    audio_cfg: float,
    text_cfg: float,
    use_lora: bool,
    lora_scale: float
):
    """
    Traite la vidéo en arrière-plan avec InfiniteTalk
    """
    try:
        logger.info(f"Début du traitement pour {task_id}")
        processing_queue[task_id]["status"] = "processing"
        processing_queue[task_id]["progress"] = 10
        
        # Construction de la commande InfiniteTalk
        cmd = [
            "python", "generate_infinitetalk.py",
            "--ckpt_dir", INFINITETALK_CONFIG["ckpt_dir"],
            "--wav2vec_dir", INFINITETALK_CONFIG["wav2vec_dir"],
            "--infinitetalk_dir", INFINITETALK_CONFIG["infinitetalk_dir"],
            "--input_json", config_path,
            "--size", f"infinitetalk-{resolution}",
            "--sample_steps", str(steps),
            "--mode", mode,
            "--motion_frame", str(INFINITETALK_CONFIG["motion_frame"]),
            "--sample_audio_guide_scale", str(audio_cfg),
            "--sample_text_guide_scale", str(text_cfg),
            "--num_persistent_param_in_dit", str(INFINITETALK_CONFIG["num_persistent_param_in_dit"]),
            "--save_file", f"{OUTPUT_DIR}/{task_id}"
        ]
        
        # Ajouter LoRA si activé
        if use_lora:
            lora_path = f"{WEIGHTS_DIR}/Wan2.1_I2V_14B_FusionX_LoRA.safetensors"
            if os.path.exists(lora_path):
                cmd.extend([
                    "--lora_dir", lora_path,
                    "--lora_scale", str(lora_scale),
                    "--sample_steps", "8",  # LoRA permet moins d'étapes
                    "--sample_shift", "2"
                ])
                logger.info("LoRA activé pour accélération")
        
        # Mise à jour du progrès
        processing_queue[task_id]["progress"] = 20
        
        # Exécution de la commande
        logger.info(f"Commande: {' '.join(cmd)}")
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd="/workspace/InfiniteTalk"
        )
        
        # Simuler la progression pendant l'exécution
        async def update_progress():
            progress_steps = [30, 40, 50, 60, 70, 80, 90]
            for progress in progress_steps:
                await asyncio.sleep(10)  # Attendre 10 secondes entre chaque mise à jour
                if processing_queue[task_id]["status"] == "processing":
                    processing_queue[task_id]["progress"] = progress
        
        # Lancer la mise à jour du progrès en parallèle
        progress_task = asyncio.create_task(update_progress())
        
        # Attendre la fin du processus
        stdout, stderr = await process.communicate()
        
        # Annuler la tâche de progrès
        progress_task.cancel()
        
        if process.returncode == 0:
            # Rechercher le fichier de sortie
            output_file = f"{OUTPUT_DIR}/{task_id}_0.mp4"
            if not os.path.exists(output_file):
                # Essayer sans le _0
                output_file = f"{OUTPUT_DIR}/{task_id}.mp4"
            
            if os.path.exists(output_file):
                processing_queue[task_id]["status"] = "completed"
                processing_queue[task_id]["output"] = output_file
                processing_queue[task_id]["progress"] = 100
                processing_queue[task_id]["completed_at"] = datetime.now().isoformat()
                
                # Obtenir la taille du fichier
                file_size = os.path.getsize(output_file) / (1024 * 1024)  # En MB
                processing_queue[task_id]["file_size_mb"] = round(file_size, 2)
                
                logger.info(f"Vidéo générée avec succès: {output_file}")
            else:
                raise Exception(f"Fichier de sortie non trouvé: {output_file}")
        else:
            error_msg = stderr.decode() if stderr else "Erreur inconnue lors de la génération"
            logger.error(f"Erreur InfiniteTalk: {error_msg}")
            raise Exception(f"Erreur de génération: {error_msg[:500]}")  # Limiter la taille du message
            
    except asyncio.CancelledError:
        # Gestion de l'annulation
        processing_queue[task_id]["status"] = "cancelled"
        processing_queue[task_id]["error"] = "Tâche annulée"
        logger.warning(f"Tâche {task_id} annulée")
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement de {task_id}: {e}")
        processing_queue[task_id]["status"] = "error"
        processing_queue[task_id]["error"] = str(e)
        processing_queue[task_id]["progress"] = 0

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """
    Vérifie le statut d'une tâche de génération
    
    Returns:
        Status de la tâche avec progression et détails
    """
    if task_id not in processing_queue:
        raise HTTPException(
            status_code=404,
            detail="Tâche non trouvée"
        )
    
    return processing_queue[task_id]

@app.get("/download/{task_id}")
async def download_video(task_id: str):
    """
    Télécharge la vidéo générée
    
    Args:
        task_id: ID de la tâche
        
    Returns:
        Fichier vidéo MP4
    """
    if task_id not in processing_queue:
        raise HTTPException(
            status_code=404,
            detail="Tâche non trouvée"
        )
    
    task = processing_queue[task_id]
    
    if task["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Vidéo pas encore prête. Statut: {task['status']}"
        )
    
    output_path = task["output"]
    if not os.path.exists(output_path):
        raise HTTPException(
            status_code=404,
            detail="Fichier vidéo non trouvé sur le serveur"
        )
    
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"infinitetalk_{task_id}.mp4",
        headers={
            "Content-Disposition": f"attachment; filename=infinitetalk_{task_id}.mp4"
        }
    )

@app.delete("/cleanup/{task_id}")
async def cleanup_task(task_id: str):
    """
    Nettoie les fichiers d'une tâche terminée
    
    Args:
        task_id: ID de la tâche à nettoyer
        
    Returns:
        Message de confirmation
    """
    if task_id not in processing_queue:
        raise HTTPException(
            status_code=404,
            detail="Tâche non trouvée"
        )
    
    try:
        # Supprimer les fichiers uploadés
        files_to_delete = [
            f"{UPLOAD_DIR}/{task_id}_audio.wav",
            f"{UPLOAD_DIR}/{task_id}_image.jpg",
            f"{UPLOAD_DIR}/{task_id}_image.png",
            f"{UPLOAD_DIR}/{task_id}_video.mp4",
            f"{UPLOAD_DIR}/{task_id}_config.json"
        ]
        
        for file_path in files_to_delete:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Fichier supprimé: {file_path}")
        
        # Supprimer le fichier de sortie
        if processing_queue[task_id].get("output"):
            output_path = processing_queue[task_id]["output"]
            if os.path.exists(output_path):
                os.remove(output_path)
                logger.info(f"Vidéo de sortie supprimée: {output_path}")
        
        # Retirer de la file d'attente
        del processing_queue[task_id]
        
        return {"message": f"Tâche {task_id} nettoyée avec succès"}
        
    except Exception as e:
        logger.error(f"Erreur lors du nettoyage de {task_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du nettoyage: {str(e)}"
        )

@app.get("/queue")
async def get_queue_status():
    """
    Obtient le statut de la file d'attente
    
    Returns:
        Liste des tâches et leurs statuts
    """
    queue_summary = {
        "total": len(processing_queue),
        "queued": sum(1 for t in processing_queue.values() if t["status"] == "queued"),
        "processing": sum(1 for t in processing_queue.values() if t["status"] == "processing"),
        "completed": sum(1 for t in processing_queue.values() if t["status"] == "completed"),
        "error": sum(1 for t in processing_queue.values() if t["status"] == "error"),
        "tasks": []
    }
    
    # Ajouter les détails des tâches récentes (dernières 10)
    for task_id, task in list(processing_queue.items())[-10:]:
        queue_summary["tasks"].append({
            "task_id": task_id,
            "status": task["status"],
            "progress": task["progress"],
            "created_at": task.get("created_at")
        })
    
    return queue_summary

@app.post("/cleanup-old")
async def cleanup_old_files():
    """
    Nettoie les fichiers de plus de 24 heures
    
    Returns:
        Nombre de fichiers supprimés
    """
    try:
        from datetime import datetime, timedelta
        
        deleted_count = 0
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        # Nettoyer les uploads
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            
            if file_time < cutoff_time:
                os.remove(file_path)
                deleted_count += 1
                logger.info(f"Ancien fichier supprimé: {file_path}")
        
        # Nettoyer les outputs
        for filename in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, filename)
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            
            if file_time < cutoff_time:
                os.remove(file_path)
                deleted_count += 1
                logger.info(f"Ancienne vidéo supprimée: {file_path}")
        
        # Nettoyer la file d'attente
        old_tasks = []
        for task_id, task in processing_queue.items():
            if task.get("created_at"):
                task_time = datetime.fromisoformat(task["created_at"])
                if task_time < cutoff_time and task["status"] in ["completed", "error"]:
                    old_tasks.append(task_id)
        
        for task_id in old_tasks:
            del processing_queue[task_id]
            logger.info(f"Ancienne tâche supprimée de la queue: {task_id}")
        
        return {
            "files_deleted": deleted_count,
            "tasks_removed": len(old_tasks),
            "message": "Nettoyage terminé"
        }
        
    except Exception as e:
        logger.error(f"Erreur lors du nettoyage automatique: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du nettoyage: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    
    # Configuration du serveur
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"
    
    logger.info(f"Démarrage du serveur InfiniteTalk API sur {host}:{port}")
    
    # Lancer le serveur
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        reload=False  # Pas de reload en production
    )
