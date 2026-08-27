from __future__ import annotations

import asyncio
import json
import logging
import shutil
import sys
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from urllib.parse import quote

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.gpu_worker_client import PersistentGpuWorker  # noqa: E402
from app.srt import write_outputs  # noqa: E402
from app.transcript import ParsedTranscript, parse_transcript  # noqa: E402
from simple_srt_service.language import detect_language  # noqa: E402
from simple_srt_service.settings import settings  # noqa: E402


LOGGER = logging.getLogger("simple_srt_service")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

worker = PersistentGpuWorker(settings, PROJECT_ROOT)
alignment_lock = asyncio.Lock()
request_root = settings.data_dir / "requests"


@asynccontextmanager
async def lifespan(_: FastAPI):
    request_root.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "Loading resident alignment models on CUDA device %s",
        settings.cuda_visible_devices,
    )
    await asyncio.to_thread(worker.start)
    LOGGER.info("Simple SRT alignment service is ready")
    yield
    await asyncio.to_thread(worker.stop)


app = FastAPI(
    title="Simple SRT Alignment Service",
    description="上传一个音视频和一个原文 SRT，直接返回对齐后的 SRT。",
    version="1.0.0",
    lifespan=lifespan,
)


async def save_upload(upload: UploadFile, destination: Path) -> None:
    limit = settings.max_upload_mb * 1024 * 1024
    total = 0
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


def choose_language(transcript: ParsedTranscript) -> str:
    if settings.source_language != "auto":
        return settings.source_language
    return detect_language("\n".join(line.text for line in transcript.lines))


def write_normalized_jsonl(transcript: ParsedTranscript, destination: Path) -> None:
    with destination.open("w", encoding="utf-8") as handle:
        for line in transcript.lines:
            handle.write(
                json.dumps(
                    {"text": line.text, "source": line.source},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def run_alignment(
    request_id: str,
    media_path: Path,
    transcript: ParsedTranscript,
    work_dir: Path,
) -> bytes:
    normalized_path = work_dir / "source.jsonl"
    raw_path = work_dir / "aligned.raw.jsonl"
    internal_jsonl_path = work_dir / "aligned.internal.jsonl"
    result_path = work_dir / "aligned.srt"
    log_path = work_dir / "alignment.log"
    write_normalized_jsonl(transcript, normalized_path)
    language = choose_language(transcript)
    LOGGER.info("Request %s uses source language %s", request_id, language)
    argv = [
        "--media",
        str(media_path),
        "--jsonl",
        str(normalized_path),
        "--text-field",
        "text",
        "--time-field",
        "time",
        "--method-field",
        "alignment_method",
        "--source-language",
        language,
        "--output",
        str(raw_path),
    ]
    worker.run_job(
        request_id,
        argv,
        log_path,
        progress_callback=lambda progress, stage: LOGGER.info(
            "Request %s: %s%% %s", request_id, progress, stage
        ),
    )
    _, resolved = write_outputs(raw_path, internal_jsonl_path, result_path)
    LOGGER.info("Request %s completed with %s aligned cues", request_id, resolved)
    return result_path.read_bytes()


def remove_request_dir(path: Path) -> None:
    resolved = path.resolve()
    if resolved.parent != request_root.resolve():
        raise RuntimeError(f"拒绝清理非请求目录：{resolved}")
    shutil.rmtree(resolved, ignore_errors=True)


@app.get("/health", tags=["system"])
def health() -> dict[str, object]:
    status = worker.current_status
    return {
        "status": "ok" if status == "ready" else "degraded",
        "models_resident": status == "ready",
        "worker": status,
        "gpu_visible_devices": settings.cuda_visible_devices,
    }


@app.post(
    "/align",
    response_class=Response,
    responses={200: {"content": {"application/x-subrip": {}}}},
    tags=["alignment"],
)
async def align(
    media: Annotated[UploadFile, File(description="音频或视频文件")],
    srt: Annotated[UploadFile, File(description="原文 SRT 字幕文件")],
) -> Response:
    media_name = Path(media.filename or "media.bin").name
    subtitle_name = Path(srt.filename or "subtitle.srt").name
    if Path(subtitle_name).suffix.casefold() != ".srt":
        await media.close()
        await srt.close()
        raise HTTPException(status_code=400, detail="srt 必须是 SRT 文件")

    request_id = uuid.uuid4().hex
    request_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f"{request_id}-", dir=request_root))
    media_path = work_dir / ("media" + Path(media_name).suffix.casefold())
    subtitle_path = work_dir / "subtitle.srt"
    try:
        await save_upload(media, media_path)
        await save_upload(srt, subtitle_path)
        try:
            transcript = parse_transcript(
                subtitle_path.read_bytes(), filename=subtitle_name
            )
        except (UnicodeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"SRT 解析失败：{exc}") from exc

        LOGGER.info("Request %s queued: %s + %s", request_id, media_name, subtitle_name)
        async with alignment_lock:
            content = await asyncio.to_thread(
                run_alignment, request_id, media_path, transcript, work_dir
            )
        output_name = f"{Path(subtitle_name).stem}.aligned.srt"
        return Response(
            content=content,
            media_type="application/x-subrip; charset=utf-8",
            headers={
                "Content-Disposition": (
                    f"attachment; filename=aligned.srt; "
                    f"filename*=UTF-8''{quote(output_name)}"
                )
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Request %s failed", request_id)
        raise HTTPException(status_code=500, detail=f"字幕对齐失败：{exc}") from exc
    finally:
        await media.close()
        await srt.close()
        remove_request_dir(work_dir)
