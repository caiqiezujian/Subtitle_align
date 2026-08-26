from __future__ import annotations

import asyncio
import json
import logging
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Settings
from .gpu_worker_client import PersistentGpuWorker
from .llm_adapter import FlashAssistant
from .srt import write_outputs
from .transcript import parse_transcript


LOGGER = logging.getLogger("subtitle_align.jobs")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobState:
    id: str
    status: str = "queued"
    progress: int = 0
    stage: str = "等待处理"
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    media_name: str = ""
    transcript_name: str = ""
    detected_format: str | None = None
    detected_text_field: str | None = None
    line_count: int | None = None
    aligned_count: int | None = None
    source_language: str = "English"
    use_flash: bool = False
    error: str | None = None
    files: dict[str, str] = field(default_factory=dict)

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["download_urls"] = {
            kind: f"/api/jobs/{self.id}/download/{kind}" for kind in self.files
        }
        value.pop("files", None)
        return value


@dataclass
class JobOptions:
    language: str
    text_field: str | None = None
    use_flash: bool = False
    asr_context: str = ""
    local_refine: bool = True


class JobManager:
    def __init__(self, settings: Settings, project_root: Path) -> None:
        self.settings = settings
        self.project_root = project_root
        self.queue: asyncio.Queue[tuple[str, JobOptions]] = asyncio.Queue()
        self.jobs: dict[str, JobState] = {}
        self.workers: list[asyncio.Task[None]] = []
        self.flash = FlashAssistant(settings)
        self.gpu_worker = PersistentGpuWorker(settings, project_root)

    def _job_dir(self, job_id: str) -> Path:
        return self.settings.jobs_dir / job_id

    def _save(self, state: JobState) -> None:
        state.updated_at = now_iso()
        job_dir = self._job_dir(state.id)
        job_dir.mkdir(parents=True, exist_ok=True)
        temp = job_dir / "job.json.tmp"
        temp.write_text(
            json.dumps(asdict(state), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temp.replace(job_dir / "job.json")

    def _update(self, state: JobState, **changes: Any) -> None:
        for key, value in changes.items():
            setattr(state, key, value)
        self._save(state)

    async def start(self) -> None:
        if self.settings.max_concurrent_jobs != 1:
            raise RuntimeError(
                "常驻单 GPU Worker 要求 gpu.max_concurrent_jobs: 1。"
                "多个用户仍可同时提交，任务会自动排队。"
            )
        self.settings.jobs_dir.mkdir(parents=True, exist_ok=True)
        for meta in self.settings.jobs_dir.glob("*/job.json"):
            try:
                state = JobState(**json.loads(meta.read_text(encoding="utf-8")))
                if state.status in {"queued", "running"}:
                    state.status = "failed"
                    state.stage = "服务重启，任务已中止"
                    state.error = "服务在任务完成前重启，请重新提交"
                    self._save(state)
                self.jobs[state.id] = state
            except Exception:
                LOGGER.warning("Ignored unreadable job metadata: %s", meta)
        LOGGER.info("Loading persistent GPU worker on CUDA device %s", self.settings.cuda_visible_devices)
        await asyncio.to_thread(self.gpu_worker.start)
        LOGGER.info("Persistent GPU worker is ready; models remain resident")
        self.workers.append(asyncio.create_task(self._worker(0)))

    async def stop(self) -> None:
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        await asyncio.to_thread(self.gpu_worker.stop)

    def create(
        self,
        media_name: str,
        transcript_name: str,
        options: JobOptions,
    ) -> tuple[JobState, Path, Path]:
        job_id = uuid.uuid4().hex
        state = JobState(
            id=job_id,
            media_name=media_name,
            transcript_name=transcript_name,
            source_language=options.language,
            use_flash=options.use_flash,
        )
        self.jobs[job_id] = state
        job_dir = self._job_dir(job_id)
        upload_dir = job_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        media_path = upload_dir / ("media" + Path(media_name).suffix.lower())
        transcript_path = upload_dir / ("transcript" + Path(transcript_name).suffix.lower())
        self._save(state)
        return state, media_path, transcript_path

    async def enqueue(self, job_id: str, options: JobOptions) -> None:
        await self.queue.put((job_id, options))

    def get(self, job_id: str) -> JobState | None:
        return self.jobs.get(job_id)

    async def _worker(self, number: int) -> None:
        while True:
            job_id, options = await self.queue.get()
            state = self.jobs[job_id]
            try:
                await asyncio.to_thread(self._run_sync, state, options)
            except Exception as exc:
                LOGGER.exception("Job %s failed", job_id)
                self._update(
                    state,
                    status="failed",
                    stage="处理失败",
                    error=str(exc)[:1000],
                )
                (self._job_dir(job_id) / "traceback.log").write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
            finally:
                self.queue.task_done()

    def _run_sync(self, state: JobState, options: JobOptions) -> None:
        job_dir = self._job_dir(state.id)
        upload_dir = job_dir / "uploads"
        media_path = next(upload_dir.glob("media.*"), None)
        transcript_path = next(upload_dir.glob("transcript.*"), None)
        if media_path is None or transcript_path is None:
            raise RuntimeError("上传文件不完整")

        self._update(state, status="running", progress=5, stage="解析字幕文件")
        parsed = parse_transcript(
            transcript_path.read_bytes(),
            filename=state.transcript_name,
            text_field=options.text_field,
        )
        self._update(
            state,
            progress=12,
            stage="输入检查完成",
            detected_format=parsed.detected_format,
            detected_text_field=parsed.text_field,
            line_count=len(parsed.lines),
        )

        if options.use_flash:
            self._update(state, progress=14, stage="v4-flash 正在清洗文本")
            parsed = self.flash.normalize(
                parsed,
                progress_callback=lambda done, total: self._update(
                    state,
                    progress=14 + round(7 * done / max(1, total)),
                    stage=f"v4-flash 文本清洗 {done}/{total} 批",
                ),
            )
        else:
            self._update(state, progress=20, stage="文本无需模型清洗")

        normalized_path = job_dir / "source.normalized.jsonl"
        with normalized_path.open("w", encoding="utf-8") as handle:
            for line in parsed.lines:
                handle.write(
                    json.dumps(
                        {"text": line.text, "source": line.source},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

        raw_aligned = job_dir / "aligned.raw.jsonl"
        log_path = job_dir / "alignment.log"
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
            options.language,
            "--output",
            str(raw_aligned),
        ]
        if options.asr_context:
            argv.extend(["--asr-context", options.asr_context])
        if not options.local_refine:
            argv.append("--no-local-refine")

        self._update(state, progress=22, stage="正在提交给常驻 GPU 模型")
        self.gpu_worker.run_job(
            state.id,
            argv,
            log_path,
            progress_callback=lambda progress, stage: self._update(
                state, progress=progress, stage=stage
            ),
        )

        self._update(state, progress=96, stage="生成 SRT 和 JSONL")
        result_dir = job_dir / "results"
        result_dir.mkdir(exist_ok=True)
        jsonl_path = result_dir / "aligned.jsonl"
        srt_path = result_dir / "aligned.srt"
        total, resolved = write_outputs(raw_aligned, jsonl_path, srt_path)
        self._update(
            state,
            status="completed",
            progress=100,
            stage="对齐完成",
            line_count=total,
            aligned_count=resolved,
            files={"jsonl": str(jsonl_path), "srt": str(srt_path)},
            error=None,
        )
