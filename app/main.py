from __future__ import annotations

import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import settings
from .jobs import JobManager, JobOptions


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
manager = JobManager(settings, ROOT)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await manager.start()
    yield
    await manager.stop()


app = FastAPI(
    title="声轨刻度 Subtitle Alignment",
    version=__version__,
    description="音视频与逐行原文的 GPU 强制对齐服务",
    lifespan=lifespan,
)

if settings.allow_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allow_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def require_api_key(x_api_key: Annotated[Optional[str], Header()] = None) -> None:
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="API key 无效")


async def save_upload(upload: UploadFile, destination: Path) -> int:
    total = 0
    limit = settings.max_upload_mb * 1024 * 1024
    try:
        with destination.open("wb") as handle:
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"单个文件不能超过 {settings.max_upload_mb} MB",
                    )
                handle.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
    if total == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="上传文件为空")
    return total


@app.get("/api/health")
def health() -> dict:
    models = {
        "asr": (settings.model_root / "Qwen3-ASR-1.7B").exists(),
        "forced_aligner": (settings.model_root / "Qwen3-ForcedAligner-0.6B").exists(),
    }
    return {
        "status": "ok" if shutil.which("ffmpeg") and all(models.values()) else "degraded",
        "version": __version__,
        "config_file": str(settings.config_path),
        "port": settings.server_port,
        "gpu_visible_devices": settings.cuda_visible_devices,
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "models": models,
        "v4_flash": manager.flash.available,
        "max_concurrent_jobs": settings.max_concurrent_jobs,
    }


@app.post("/api/jobs", status_code=202, dependencies=[Depends(require_api_key)])
async def create_job(
    media: Annotated[UploadFile, File(description="音频或视频文件")],
    transcript: Annotated[UploadFile, File(description="TXT/SRT/JSONL/JSON/CSV/TSV")],
    language: Annotated[str, Form()] = "English",
    text_field: Annotated[str, Form()] = "",
    use_flash: Annotated[bool, Form()] = False,
    asr_context: Annotated[str, Form()] = "",
    flash_attention: Annotated[bool, Form()] = False,
    local_refine: Annotated[bool, Form()] = True,
) -> dict:
    media_name = Path(media.filename or "media.bin").name
    transcript_name = Path(transcript.filename or "transcript.txt").name
    if language not in {"Chinese", "English", "Japanese"}:
        raise HTTPException(status_code=400, detail="原文语言仅支持 Chinese、English、Japanese")
    allowed_transcript = {".txt", ".srt", ".jsonl", ".json", ".csv", ".tsv"}
    if Path(transcript_name).suffix.casefold() not in allowed_transcript:
        raise HTTPException(status_code=400, detail="字幕仅支持 TXT、SRT、JSONL、JSON、CSV、TSV")
    options = JobOptions(
        language=language,
        text_field=text_field.strip() or None,
        use_flash=use_flash,
        asr_context=asr_context.strip(),
        flash_attention=flash_attention,
        local_refine=local_refine,
    )
    state, media_path, transcript_path = manager.create(
        media_name, transcript_name, options
    )
    try:
        await save_upload(media, media_path)
        await save_upload(transcript, transcript_path)
    except Exception:
        manager._update(state, status="failed", stage="上传失败", error="文件上传失败")
        raise
    await manager.enqueue(state.id, options)
    return state.public_dict()


@app.get("/api/jobs/{job_id}", dependencies=[Depends(require_api_key)])
def get_job(job_id: str) -> dict:
    state = manager.get(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return state.public_dict()


@app.get("/api/jobs/{job_id}/download/{kind}", dependencies=[Depends(require_api_key)])
def download(job_id: str, kind: str) -> FileResponse:
    state = manager.get(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if state.status != "completed" or kind not in state.files:
        raise HTTPException(status_code=404, detail="结果文件尚未生成")
    path = Path(state.files[kind])
    media_type = "application/x-subrip" if kind == "srt" else "application/x-ndjson"
    stem = Path(state.media_name).stem or "aligned"
    return FileResponse(path, media_type=media_type, filename=f"{stem}.aligned.{kind}")


app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
