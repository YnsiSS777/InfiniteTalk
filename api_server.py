# api_server.py
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json, uuid, os, subprocess, asyncio, tempfile, shutil, logging
from typing import Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("infinitetalk-api")

# ---------------------------
# Health app (readiness probe)
# ---------------------------
health_app = FastAPI(title="Health", version="1.0.0")
READY = {"ok": False, "reason": "starting"}

@health_app.get("/ping")
def ping():
    """
    204 = en cours d'initialisation
    200 = prêt
    """
    if READY["ok"]:
        return ("", 200)
    else:
        return ("", 204)

# ---------------------------
# API principale
# ---------------------------
app = FastAPI(title="InfiniteTalk API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # à restreindre en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Répertoires de travail
UPLOAD_DIR = "/workspace/uploads"
OUTPUT_DIR = "/workspace/outputs"
for d in (UPLOAD_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)

# Chemins MODELS —> configurable par env
MODELS_DIR = os.getenv("MODELS_DIR", "/workspace/persistent/models")  # <- chez toi c’est ici
LOCAL_FALLBACK = "/workspace/models"  # fallback si besoin
if not os.path.exists(MODELS_DIR) and os.path.exists(LOCAL_FALLBACK):
    MODELS_DIR = LOCAL_FALLBACK

logger.info(f"📁 MODELS_DIR = {MODELS_DIR}")

# File d'attente en mémoire (statuts)
processing_queue = {}

# Config InfiniteTalk (aligne avec ton disque)
INFINITETALK_CONFIG = {
    "ckpt_dir": f"{MODELS_DIR}/Wan2.1-I2V-14B-480P",
    "wav2vec_dir": f"{MODELS_DIR}/chinese-wav2vec2-base",
    "infinitetalk_dir": f"{MODELS_DIR}/InfiniteTalk/single/infinitetalk.safetensors",
    "motion_frame": 9,
    "num_persistent_param_in_dit": 0
}

def _check_models_exist() -> list[str]:
    req = [
        INFINITETALK_CONFIG["ckpt_dir"],
        INFINITETALK_CONFIG["wav2vec_dir"],
        INFINITETALK_CONFIG["infinitetalk_dir"]
    ]
    return [p for p in req if not os.path.exists(p)]

async def _warmup():
    """
    Ici, simple vérification de présence des modèles.
    Si besoin tu peux charger un encodeur léger pour tester CUDA/FFmpeg.
    """
    missing = _check_models_exist()
    if missing:
        READY["ok"] = False
        READY["reason"] = f"missing models: {missing}"
        logger.error(f"⚠️ Modèles manquants: {missing}")
    else:
        READY["ok"] = True
        READY["reason"] = "ready"
        logger.info("✅ Modèles présents, API prête.")

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Démarrage InfiniteTalk API (HTTP mode)")
    try:
        if os.path.exists(MODELS_DIR):
            disk = shutil.disk_usage(MODELS_DIR)
            logger.info(f"💾 Free {disk.free/1e9:.1f} GB / Total {disk.total/1e9:.1f} GB (dans MODELS_DIR)")
    except Exception as e:
        logger.warning(f"Disk usage check failed: {e}")

    READY["ok"] = False
    READY["reason"] = "initializing"
    await _warmup()  # bascule READY à True si OK

@app.get("/")
async def root():
    return {
        "message": "InfiniteTalk API is running",
        "version": "1.0.0",
        "models_dir": MODELS_DIR,
        "ready": READY,
        "endpoints": {
            "health": "/health",
            "ping": "/ping (health_app)",
            "generate": "/generate",
            "status": "/status/{task_id}",
            "download": "/download/{task_id}"
        }
    }

@app.get("/health")
async def health_check():
    try:
        models_exist = len(_check_models_exist()) == 0
        base = "/workspace"
        disk = shutil.disk_usage(base) if os.path.exists(base) else None
        return {
            "status": "healthy" if READY["ok"] and models_exist else "initializing",
            "ready": READY,
            "gpu_available": True,   # tu peux remplacer par torch.cuda.is_available()
            "models_loaded": models_exist,
            "models_dir": MODELS_DIR,
            "free_disk_gb": round(disk.free/(1024**3), 2) if disk else None,
            "queue_size": len(processing_queue)
        }
    except Exception as e:
        logger.exception("Health check failed")
        return JSONResponse(status_code=503, content={"status": "unhealthy", "error": str(e)})

# --------- endpoint génération (proche du tien)
@app.post("/generate")
async def generate_video(
    background_tasks: BackgroundTasks,
    image: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    audio: UploadFile = File(...),
    resolution: str = Form("480"),
    mode: str = Form("streaming"),
    steps: int = Form(40),
    audio_cfg: float = Form(4.0),
    text_cfg: float = Form(5.0),
    use_lora: bool = Form(False),
    lora_scale: float = Form(1.0)
):
    if not READY["ok"]:
        raise HTTPException(503, f"Service not ready: {READY['reason']}")

    if not image and not video:
        raise HTTPException(400, "Une image ou vidéo source est requise")
    if resolution not in ("480", "720"):
        raise HTTPException(400, "La résolution doit être 480 ou 720")

    task_id = str(uuid.uuid4())
    try:
        # audio
        audio_path = f"{UPLOAD_DIR}/{task_id}_audio.wav"
        with open(audio_path, "wb") as f:
            f.write(await audio.read())

        # input
        input_type = "image"
        input_path = None
        if image:
            ext = (image.filename or "jpg").split(".")[-1]
            input_path = f"{UPLOAD_DIR}/{task_id}_image.{ext}"
            with open(input_path, "wb") as f:
                f.write(await image.read())
            input_type = "image"
        elif video:
            input_path = f"{UPLOAD_DIR}/{task_id}_video.mp4"
            with open(input_path, "wb") as f:
                f.write(await video.read())
            input_type = "video"

        # config job
        config = {
            "input_type": input_type,
            "input_path": input_path,
            "audio_path": audio_path,
            "output_path": f"{OUTPUT_DIR}/{task_id}_output.mp4"
        }
        config_path = f"{UPLOAD_DIR}/{task_id}_config.json"
        with open(config_path, "w") as f:
            json.dump([config], f)

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

        background_tasks.add_task(
            process_video, task_id, config_path, resolution, mode,
            steps, audio_cfg, text_cfg, use_lora, lora_scale
        )

        return {
            "task_id": task_id,
            "status": "queued",
            "message": "Vidéo en cours de génération",
            "estimated_time_seconds": 60 if resolution == "480" else 90
        }
    except Exception as e:
        processing_queue.pop(task_id, None)
        raise HTTPException(500, str(e))

async def process_video(task_id: str, config_path: str, resolution: str, mode: str,
                        steps: int, audio_cfg: float, text_cfg: float,
                        use_lora: bool, lora_scale: float):
    try:
        processing_queue[task_id]["status"] = "processing"
        processing_queue[task_id]["progress"] = 10

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

        if use_lora:
            lora_path = f"{MODELS_DIR}/Wan2.1_I2V_14B_FusionX_LoRA.safetensors"
            if os.path.exists(lora_path):
                cmd += ["--lora_dir", lora_path, "--lora_scale", str(lora_scale),
                        "--sample_steps", "8", "--sample_shift", "2"]

        processing_queue[task_id]["progress"] = 20

        # IMPORTANT: cwd = /workspace/InfiniteTalk
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            cwd="/workspace/InfiniteTalk"
        )

        async def fake_progress():
            for p in (30, 40, 50, 60, 70, 80, 90):
                await asyncio.sleep(10)
                if processing_queue[task_id]["status"] == "processing":
                    processing_queue[task_id]["progress"] = p

        prog_task = asyncio.create_task(fake_progress())
        stdout, stderr = await process.communicate()
        prog_task.cancel()

        if process.returncode == 0:
            out = f"{OUTPUT_DIR}/{task_id}_0.mp4"
            if not os.path.exists(out):
                out = f"{OUTPUT_DIR}/{task_id}.mp4"
            if not os.path.exists(out):
                raise FileNotFoundError("Fichier de sortie non trouvé")

            processing_queue[task_id].update({
                "status": "completed",
                "output": out,
                "progress": 100,
                "completed_at": datetime.now().isoformat(),
                "file_size_mb": round(os.path.getsize(out) / (1024*1024), 2)
            })
        else:
            err = (stderr or b"").decode()
            raise RuntimeError(err[:500])

    except asyncio.CancelledError:
        processing_queue[task_id]["status"] = "cancelled"
        processing_queue[task_id]["error"] = "Tâche annulée"
    except Exception as e:
        processing_queue[task_id]["status"] = "error"
        processing_queue[task_id]["error"] = str(e)
        processing_queue[task_id]["progress"] = 0

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in processing_queue:
        raise HTTPException(404, "Tâche non trouvée")
    return processing_queue[task_id]

@app.get("/download/{task_id}")
async def download_video(task_id: str):
    if task_id not in processing_queue:
        raise HTTPException(404, "Tâche non trouvée")
    task = processing_queue[task_id]
    if task["status"] != "completed":
        raise HTTPException(400, f"Vidéo pas encore prête (statut: {task['status']})")
    output_path = task["output"]
    if not os.path.exists(output_path):
        raise HTTPException(404, "Fichier vidéo non trouvé")
    return FileResponse(output_path, media_type="video/mp4",
                        filename=f"infinitetalk_{task_id}.mp4")

@app.delete("/cleanup/{task_id}")
async def cleanup_task(task_id: str):
    if task_id not in processing_queue:
        raise HTTPException(404, "Tâche non trouvée")
    try:
        files = [
            f"{UPLOAD_DIR}/{task_id}_audio.wav",
            f"{UPLOAD_DIR}/{task_id}_image.jpg",
            f"{UPLOAD_DIR}/{task_id}_image.png",
            f"{UPLOAD_DIR}/{task_id}_video.mp4",
            f"{UPLOAD_DIR}/{task_id}_config.json",
        ]
        for p in files:
            if os.path.exists(p):
                os.remove(p)
        if processing_queue[task_id].get("output"):
            out = processing_queue[task_id]["output"]
            if os.path.exists(out):
                os.remove(out)
        del processing_queue[task_id]
        return {"message": f"Tâche {task_id} nettoyée"}
    except Exception as e:
        raise HTTPException(500, f"Erreur lors du nettoyage: {e}")

@app.get("/queue")
async def get_queue_status():
    summary = {
        "total": len(processing_queue),
        "queued": sum(1 for t in processing_queue.values() if t["status"] == "queued"),
        "processing": sum(1 for t in processing_queue.values() if t["status"] == "processing"),
        "completed": sum(1 for t in processing_queue.values() if t["status"] == "completed"),
        "error": sum(1 for t in processing_queue.values() if t["status"] == "error"),
        "tasks": []
    }
    for tid, task in list(processing_queue.items())[-10:]:
        summary["tasks"].append({
            "task_id": tid,
            "status": task["status"],
            "progress": task["progress"],
            "created_at": task.get("created_at")
        })
    return summary

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
