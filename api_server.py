# api_server.py
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json, uuid, os, asyncio, shutil, logging
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
    """204 = en cours d'initialisation ; 200 = prêt"""
    return ("", 200) if READY["ok"] else ("", 204)

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

# ---------- AUTO-DÉTECTION MODELS_DIR ----------
CANDIDATES = [
    os.getenv("MODELS_DIR", "/workspace/persistent/models"),
    "/workspace/persistent/models",
    "/workspace/persistent",
    "/workspace/data",
    "/data",
    "/workspace/models",
    "/models",
    "/mnt/models",
    "/runpod-volume",
]
REQS = [
    ("Wan2.1-I2V-14B-480P", False),  # dossier
    ("chinese-wav2vec2-base", False),
    (os.path.join("InfiniteTalk", "single", "infinitetalk.safetensors"), True),  # fichier
]

def _has_all(base: str) -> bool:
    try:
        for rel, is_file in REQS:
            full = os.path.join(base, rel)
            if is_file:
                if not os.path.isfile(full):
                    return False
            else:
                if not os.path.isdir(full):
                    return False
        return True
    except Exception:
        return False

MODELS_DIR = None
for base in CANDIDATES:
    if _has_all(base):
        MODELS_DIR = base
        break
if MODELS_DIR is None:
    # garde la valeur d'env même si manquante pour une erreur explicite
    MODELS_DIR = os.getenv("MODELS_DIR", "/workspace/persistent/models")

logger.info(f"📁 MODELS_DIR (effective) = {MODELS_DIR}")

# File d'attente en mémoire (statuts) + logs courts
processing_queue = {}
TASK_LOGS = {}  # ring buffer simple pour /logs/{task_id}

def log_task(tid: str, msg: str):
    arr = TASK_LOGS.setdefault(tid, [])
    arr.append(msg)
    if len(arr) > 200:
        TASK_LOGS[tid] = arr[-200:]

# Config InfiniteTalk (utilise MODELS_DIR détecté)
INFINITETALK_CONFIG = {
    "ckpt_dir": os.path.join(MODELS_DIR, "Wan2.1-I2V-14B-480P"),
    "wav2vec_dir": os.path.join(MODELS_DIR, "chinese-wav2vec2-base"),
    "infinitetalk_dir": os.path.join(MODELS_DIR, "InfiniteTalk", "single", "infinitetalk.safetensors"),
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
    """Vérification des modèles et bascule READY."""
    missing = _check_models_exist()
    if missing:
        READY["ok"] = False
        READY["reason"] = f"missing models: {missing}"
        logger.error(f"⚠️ Modèles manquants: {missing}")
    else:
        READY["ok"] = True
        READY["reason"] = "ready"
        try:
            if os.path.exists(MODELS_DIR):
                disk = shutil.disk_usage(MODELS_DIR)
                logger.info(f"💾 Models dir free {disk.free/1e9:.1f} GB / total {disk.total/1e9:.1f} GB")
        except Exception as e:
            logger.warning(f"Disk usage check failed: {e}")
        logger.info("✅ Modèles présents, API prête.")

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Démarrage InfiniteTalk API (HTTP mode)")
    READY["ok"] = False
    READY["reason"] = "initializing"
    await _warmup()

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
            "download": "/download/{task_id}",
            "logs": "/logs/{task_id}",
            "gpu": "/gpu"
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
            "gpu_available": True,   # remplaçable par torch.cuda.is_available()
            "models_loaded": models_exist,
            "models_dir": MODELS_DIR,
            "free_disk_gb": round(disk.free/(1024**3), 2) if disk else None,
            "queue_size": len(processing_queue)
        }
    except Exception as e:
        logger.exception("Health check failed")
        return JSONResponse(status_code=503, content={"status": "unhealthy", "error": str(e)})

@app.get("/gpu")
def gpu():
    """Petit endpoint debug pour voir l'activité GPU."""
    try:
        import subprocess, shutil as _sh
        if not _sh.which("nvidia-smi"):
            return {"nvidia_smi": "not found"}
        out = subprocess.check_output([
            "nvidia-smi","--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits"
        ]).decode().strip()
        return {"nvidia_smi": out}  # ex: "65, 18000"
    except Exception as e:
        return {"error": str(e)}

# --------- endpoint génération
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
        log_task(task_id, "build command")

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
            lora_path = os.path.join(MODELS_DIR, "Wan2.1_I2V_14B_FusionX_LoRA.safetensors")
            if os.path.exists(lora_path):
                cmd += ["--lora_dir", lora_path, "--lora_scale", str(lora_scale),
                        "--sample_steps", "8", "--sample_shift", "2"]

        processing_queue[task_id]["progress"] = 20
        log_task(task_id, "run generate_infinitetalk.py")

        # ENV explicite pour le sous-processus (clé pour débloquer 'import wan')
        child_env = os.environ.copy()
        child_env["PYTHONPATH"] = "/workspace/InfiniteTalk:/workspace/InfiniteTalk/src:" + child_env.get("PYTHONPATH", "")
        # Évite des compils trop larges si extension CUDA est buildée à la volée
        child_env.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")   # Ada (RTX 4090/50xx)
        child_env.setdefault("CUDA_VISIBLE_DEVICES", "0")

        # IMPORTANT: cwd = /workspace/InfiniteTalk
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd="/workspace/InfiniteTalk",
            env=child_env,
        )

        async def fake_progress():
            for p in (30, 40, 50, 60, 70, 80, 90):
                await asyncio.sleep(10)
                if processing_queue[task_id]["status"] == "processing":
                    processing_queue[task_id]["progress"] = p

        prog_task = asyncio.create_task(fake_progress())
        stdout, stderr = await process.communicate()
        prog_task.cancel()
        log_task(task_id, f"returncode={process.returncode}")

        # Sauvegarder logs complets pour debug
        err_log = f"{OUTPUT_DIR}/{task_id}.stderr.log"
        try:
            with open(err_log, "wb") as f:
                if stdout:
                    f.write(b"=== STDOUT ===\n")
                    f.write(stdout)
                    f.write(b"\n\n")
                if stderr:
                    f.write(b"=== STDERR ===\n")
                    f.write(stderr)
        except Exception:
            pass

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
            log_task(task_id, f"completed size={processing_queue[task_id]['file_size_mb']}MB")
        else:
            err = (stderr or b"").decode(errors="ignore")
            log_task(task_id, f"stderr: {err[:1000]}")
            raise RuntimeError(err[:500])

    except asyncio.CancelledError:
        processing_queue[task_id]["status"] = "cancelled"
        processing_queue[task_id]["error"] = "Tâche annulée"
        log_task(task_id, "cancelled")
    except Exception as e:
        processing_queue[task_id]["status"] = "error"
        processing_queue[task_id]["error"] = str(e)
        processing_queue[task_id]["progress"] = 0
        log_task(task_id, f"error: {e}")

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

@app.get("/logs/{task_id}")
def get_logs(task_id: str):
    # Renvoie le ring buffer + lien vers le fichier stderr si présent
    log_path = f"{OUTPUT_DIR}/{task_id}.stderr.log"
    return {
        "task_id": task_id,
        "logs": TASK_LOGS.get(task_id, []),
        "stderr_log_file": log_path if os.path.exists(log_path) else None
    }

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
            f"{OUTPUT_DIR}/{task_id}.stderr.log",
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
