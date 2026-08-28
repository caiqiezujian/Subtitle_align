from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import sys
from collections import Counter
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterator, Sequence


LOGGER = logging.getLogger("simple_srt_service.trace")
TRACE_SCHEMA_VERSION = 1


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _model_manifest(root: Path) -> dict[str, Any]:
    """Fingerprint model layout without hashing multi-gigabyte weights."""
    if not root.is_dir():
        return {"path": str(root), "exists": False}

    files: list[dict[str, Any]] = []
    manifest_digest = hashlib.sha256()
    total_bytes = 0
    small_metadata_suffixes = {
        ".json",
        ".txt",
        ".yaml",
        ".yml",
        ".model",
        ".tiktoken",
    }
    for path in sorted((item for item in root.rglob("*") if item.is_file())):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        total_bytes += size
        record: dict[str, Any] = {"path": relative, "size_bytes": size}
        if size <= 16 * 1024 * 1024 and path.suffix.casefold() in small_metadata_suffixes:
            record["sha256"] = _sha256(path)
        files.append(record)
        manifest_digest.update(relative.encode("utf-8"))
        manifest_digest.update(b"\0")
        manifest_digest.update(str(size).encode("ascii"))
        manifest_digest.update(b"\0")
        if "sha256" in record:
            manifest_digest.update(str(record["sha256"]).encode("ascii"))
        manifest_digest.update(b"\n")

    return {
        "path": str(root),
        "exists": True,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "manifest_sha256": manifest_digest.hexdigest(),
        "files": files,
    }


def _time_record(item: Any) -> dict[str, Any] | None:
    if item is None:
        return None
    start = getattr(item, "start", None)
    end = getattr(item, "end", None)
    return {
        "start": None if start is None else round(float(start), 6),
        "end": None if end is None else round(float(end), 6),
        "matched_hyp_start": getattr(item, "matched_hyp_start", None),
        "matched_hyp_end": getattr(item, "matched_hyp_end", None),
        "method": str(getattr(item, "method", "unknown")),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


class _TracingEngine:
    def __init__(self, engine: Any, trace: "DetailedAlignmentTrace") -> None:
        self._engine = engine
        self._trace = trace

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)

    def tokenize_text(self, text: str, language: str) -> list[str]:
        return list(self._engine.tokenize_text(text, language))

    def transcribe_chunks(self, *args: Any, **kwargs: Any) -> Any:
        chunks = kwargs.get("chunks")
        if chunks is None and len(args) > 1:
            chunks = args[1]
        if chunks is not None:
            self._trace.vad_chunks = [
                {
                    "chunk_index": int(chunk.index),
                    "start_sample": int(chunk.start_sample),
                    "end_sample": int(chunk.end_sample),
                    "start": round(float(chunk.start_sec), 6),
                    "end": round(float(chunk.end_sec), 6),
                    "duration": round(float(chunk.duration_sec), 6),
                }
                for chunk in chunks
            ]

        results = self._engine.transcribe_chunks(*args, **kwargs)
        self._trace.asr_timeline = [
            {
                "token_index": index,
                "chunk_index": int(item.chunk_index),
                "text": str(item.text),
                "normalized": str(item.norm),
                "start": round(float(item.start), 6),
                "end": round(float(item.end), 6),
                "duration": round(float(item.end) - float(item.start), 6),
            }
            for index, item in enumerate(results)
        ]
        return results

    def refine_lines(self, *args: Any, **kwargs: Any) -> Any:
        rows = kwargs.get("rows")
        coarse_times = kwargs.get("coarse_times")
        if rows is None and len(args) > 1:
            rows = args[1]
        if coarse_times is None and len(args) > 2:
            coarse_times = args[2]
        if rows is not None and coarse_times is not None:
            self._trace.capture_line_stage(rows, coarse_times, "coarse")

        results = self._engine.refine_lines(*args, **kwargs)
        if rows is not None:
            self._trace.capture_line_stage(rows, results, "refined")
        return results


class DetailedAlignmentTrace:
    """Capture every alignment boundary without changing the shared engine."""

    def __init__(
        self,
        *,
        work_dir: Path,
        job_id: str,
        engine_options: dict[str, Any],
    ) -> None:
        self.work_dir = work_dir
        self.job_id = job_id
        self.engine_options = engine_options
        self.tokenized_lines: list[dict[str, Any]] = []
        self.vad_chunks: list[dict[str, Any]] = []
        self.asr_timeline: list[dict[str, Any]] = []
        self.token_mapping: list[dict[str, Any]] = []
        self.line_stages: dict[int, dict[str, Any]] = {}
        self.capture_errors: list[str] = []
        self.runtime: dict[str, Any] = {}

    def _capture(self, label: str, callback: Any) -> None:
        try:
            callback()
        except Exception as exc:  # Diagnostics must never fail an alignment.
            message = f"{label}: {type(exc).__name__}: {exc}"
            self.capture_errors.append(message)
            LOGGER.exception("Request %s trace capture failed: %s", self.job_id, label)

    def capture_runtime(self, aligner: Any, engine: Any) -> None:
        def collect() -> None:
            model_paths = {
                "asr": Path(str(aligner.ASR_MODEL_PATH)),
                "forced_aligner": Path(str(aligner.FORCED_ALIGNER_PATH)),
            }
            self.runtime = {
                "schema_version": TRACE_SCHEMA_VERSION,
                "job_id": self.job_id,
                "python": sys.version,
                "platform": platform.platform(),
                "packages": {
                    name: _package_version(name)
                    for name in (
                        "torch",
                        "torch-npu",
                        "transformers",
                        "qwen-asr",
                        "vllm",
                        "vllm-ascend",
                        "silero-vad",
                    )
                },
                "environment": {
                    "ASCEND_RT_VISIBLE_DEVICES": os.environ.get(
                        "ASCEND_RT_VISIBLE_DEVICES"
                    ),
                    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "VLLM_WORKER_MULTIPROC_METHOD": os.environ.get(
                        "VLLM_WORKER_MULTIPROC_METHOD"
                    ),
                },
                "engine": {
                    "wrapper_class": type(engine).__name__,
                    "asr_class": type(getattr(engine, "asr", None)).__name__,
                    "forced_aligner_class": type(
                        getattr(getattr(engine, "asr", None), "forced_aligner", None)
                    ).__name__,
                    "options": self.engine_options,
                },
                "models": {
                    name: _model_manifest(path) for name, path in model_paths.items()
                },
            }

        self._capture("runtime", collect)

    def capture_mapping(
        self,
        ref_tokens: Sequence[str],
        hyp_tokens: Sequence[str],
        mapping: Sequence[int | None],
    ) -> None:
        def collect() -> None:
            self.token_mapping = [
                {
                    "reference_index": index,
                    "reference_token": str(ref_token),
                    "hypothesis_index": None if hyp_index is None else int(hyp_index),
                    "hypothesis_token": (
                        None
                        if hyp_index is None or not 0 <= int(hyp_index) < len(hyp_tokens)
                        else str(hyp_tokens[int(hyp_index)])
                    ),
                    "exact_normalized_match": (
                        False
                        if hyp_index is None or not 0 <= int(hyp_index) < len(hyp_tokens)
                        else str(ref_token) == str(hyp_tokens[int(hyp_index)])
                    ),
                }
                for index, (ref_token, hyp_index) in enumerate(zip(ref_tokens, mapping))
            ]

        self._capture("token_mapping", collect)

    def capture_tokenized_lines(
        self,
        rows: Sequence[Any],
        all_tokens: Sequence[str],
        language: str,
    ) -> None:
        def collect() -> None:
            self.tokenized_lines = [
                {
                    "line_index": int(row.index) + 1,
                    "physical_line_number": int(row.physical_line_number),
                    "language": language,
                    "text": str(row.text),
                    "token_start": int(row.token_start),
                    "token_end": int(row.token_end),
                    "token_count": int(row.token_end) - int(row.token_start),
                    "tokens": list(all_tokens[row.token_start : row.token_end]),
                }
                for row in rows
            ]

        self._capture("tokenized_lines", collect)

    def capture_line_stage(
        self,
        rows: Sequence[Any],
        times: Sequence[Any],
        stage: str,
    ) -> None:
        def collect() -> None:
            for row, timing in zip(rows, times):
                line_index = int(row.index) + 1
                record = self.line_stages.setdefault(
                    line_index,
                    {
                        "line_index": line_index,
                        "physical_line_number": int(row.physical_line_number),
                        "text": str(row.text),
                        "token_start": int(row.token_start),
                        "token_end": int(row.token_end),
                        "token_count": int(row.token_end) - int(row.token_start),
                    },
                )
                record[stage] = _time_record(timing)

        self._capture(f"line_stage_{stage}", collect)

    @contextmanager
    def instrument(self, aligner: Any, engine: Any) -> Iterator[Any]:
        self.capture_runtime(aligner, engine)
        original_tokenize_rows = aligner.tokenize_jsonl_rows
        original_mapping = aligner.build_ref_to_hyp_mapping
        original_coarse = aligner.coarse_line_times
        original_write = aligner.write_aligned_jsonl

        def traced_tokenize_rows(*args: Any, **kwargs: Any) -> Any:
            result = original_tokenize_rows(*args, **kwargs)
            rows = kwargs.get("rows", args[1] if len(args) > 1 else [])
            language = kwargs.get("language", args[2] if len(args) > 2 else "")
            self.capture_tokenized_lines(rows, result, str(language))
            return result

        def traced_mapping(*args: Any, **kwargs: Any) -> Any:
            result = original_mapping(*args, **kwargs)
            ref_tokens = kwargs.get("ref_tokens", args[0] if args else [])
            hyp_tokens = kwargs.get("hyp_tokens", args[1] if len(args) > 1 else [])
            self.capture_mapping(ref_tokens, hyp_tokens, result)
            return result

        def traced_coarse(*args: Any, **kwargs: Any) -> Any:
            result = original_coarse(*args, **kwargs)
            rows = kwargs.get("rows", args[0] if args else [])
            self.capture_line_stage(rows, result, "coarse")
            return result

        def traced_write(*args: Any, **kwargs: Any) -> Any:
            rows = kwargs.get("rows", args[0] if args else [])
            line_times = kwargs.get("line_times", args[1] if len(args) > 1 else [])
            self.capture_line_stage(rows, line_times, "final")
            return original_write(*args, **kwargs)

        aligner.tokenize_jsonl_rows = traced_tokenize_rows
        aligner.build_ref_to_hyp_mapping = traced_mapping
        aligner.coarse_line_times = traced_coarse
        aligner.write_aligned_jsonl = traced_write
        try:
            yield _TracingEngine(engine, self)
        finally:
            aligner.tokenize_jsonl_rows = original_tokenize_rows
            aligner.build_ref_to_hyp_mapping = original_mapping
            aligner.coarse_line_times = original_coarse
            aligner.write_aligned_jsonl = original_write

    def finish(self, *, status: str, error: str | None = None) -> None:
        def write_all() -> None:
            line_rows = [self.line_stages[key] for key in sorted(self.line_stages)]
            mapped = sum(
                1 for item in self.token_mapping if item["hypothesis_index"] is not None
            )
            exact = sum(
                1 for item in self.token_mapping if item["exact_normalized_match"]
            )
            final_methods = Counter(
                str((item.get("final") or {}).get("method", "missing"))
                for item in line_rows
            )
            refined_methods = Counter(
                str((item.get("refined") or {}).get("method", "missing"))
                for item in line_rows
            )
            summary = {
                "schema_version": TRACE_SCHEMA_VERSION,
                "job_id": self.job_id,
                "status": status,
                "error": error,
                "counts": {
                    "tokenized_lines": len(self.tokenized_lines),
                    "reference_tokens": len(self.token_mapping),
                    "mapped_reference_tokens": mapped,
                    "exact_reference_tokens": exact,
                    "mapping_percent": round(
                        100.0 * mapped / max(1, len(self.token_mapping)), 4
                    ),
                    "exact_mapping_percent": round(
                        100.0 * exact / max(1, len(self.token_mapping)), 4
                    ),
                    "vad_chunks": len(self.vad_chunks),
                    "asr_timestamp_tokens": len(self.asr_timeline),
                    "line_records": len(line_rows),
                },
                "refined_method_counts": dict(sorted(refined_methods.items())),
                "final_method_counts": dict(sorted(final_methods.items())),
                "capture_errors": self.capture_errors,
            }

            _write_json(self.work_dir / "trace.runtime.json", self.runtime)
            _write_json(self.work_dir / "trace.summary.json", summary)
            _write_jsonl(
                self.work_dir / "trace.tokenized-lines.jsonl", self.tokenized_lines
            )
            _write_jsonl(self.work_dir / "trace.vad-chunks.jsonl", self.vad_chunks)
            _write_jsonl(
                self.work_dir / "trace.asr-timeline.jsonl", self.asr_timeline
            )
            _write_jsonl(
                self.work_dir / "trace.token-mapping.jsonl", self.token_mapping
            )
            _write_jsonl(self.work_dir / "trace.line-stages.jsonl", line_rows)
            LOGGER.info(
                "Request %s detailed trace saved in %s | asr_tokens=%d | "
                "mapped=%d/%d | lines=%d",
                self.job_id,
                self.work_dir,
                len(self.asr_timeline),
                mapped,
                len(self.token_mapping),
                len(line_rows),
            )

        self._capture("write_trace_artifacts", write_all)
