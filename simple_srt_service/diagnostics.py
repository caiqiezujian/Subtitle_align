from __future__ import annotations

import json
import re
import shutil
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.srt import read_jsonl


METRIC_PATTERNS: dict[str, re.Pattern[str]] = {
    "asr_timestamp_tokens": re.compile(r"ASR timestamp tokens:\s*(\d+)"),
    "mapped_reference_tokens": re.compile(
        r"Mapped reference tokens:\s*(\d+)/(\d+)\s*\(([\d.]+)%\)"
    ),
    "speech_chunks": re.compile(
        r"Speech chunks:\s*(\d+)\s*\|\s*mean\s*([\d.]+)s\s*\|\s*max\s*([\d.]+)s"
    ),
}


def parse_alignment_metrics(log_text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    token_match = METRIC_PATTERNS["asr_timestamp_tokens"].search(log_text)
    if token_match:
        metrics["asr_timestamp_tokens"] = int(token_match.group(1))

    mapped_match = METRIC_PATTERNS["mapped_reference_tokens"].search(log_text)
    if mapped_match:
        metrics["mapped_reference_tokens"] = {
            "mapped": int(mapped_match.group(1)),
            "total": int(mapped_match.group(2)),
            "percent": float(mapped_match.group(3)),
        }

    chunk_match = METRIC_PATTERNS["speech_chunks"].search(log_text)
    if chunk_match:
        metrics["speech_chunks"] = {
            "count": int(chunk_match.group(1)),
            "mean_seconds": float(chunk_match.group(2)),
            "max_seconds": float(chunk_match.group(3)),
        }

    return metrics


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def persist_request_diagnostics(
    *,
    request_id: str,
    work_dir: Path,
    diagnostics_dir: Path,
    media_name: str,
    subtitle_name: str,
    language: str | None,
    status: str,
    error: str | None = None,
) -> Path:
    """Persist small diagnostic artifacts but never retain uploaded media."""
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    prefix = diagnostics_dir / request_id

    source_log = work_dir / "alignment.log"
    source_jsonl = work_dir / "aligned.internal.jsonl"
    source_srt = work_dir / "aligned.srt"

    log_text = ""
    if source_log.is_file():
        log_text = source_log.read_text(encoding="utf-8", errors="replace")
        shutil.copy2(source_log, prefix.with_suffix(".alignment.log"))
    if source_jsonl.is_file():
        shutil.copy2(source_jsonl, prefix.with_suffix(".aligned.jsonl"))
    if source_srt.is_file():
        shutil.copy2(source_srt, prefix.with_suffix(".aligned.srt"))

    rows = read_jsonl(source_jsonl) if source_jsonl.is_file() else []
    method_counts = Counter(str(row.get("method", "unknown")) for row in rows)
    unresolved_indices = [
        int(row.get("index", offset))
        for offset, row in enumerate(rows, 1)
        if row.get("start") is None or row.get("end") is None
    ]
    interpolated_methods = {
        "punctuation_gap_interpolation",
        "unresolved_gap_interpolation",
        "neighbor_gap_interpolation",
        "left_neighbor_fallback",
        "right_neighbor_fallback",
    }

    summary: dict[str, Any] = {
        "request_id": request_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "media_name": media_name,
        "subtitle_name": subtitle_name,
        "language": language,
        "total_lines": len(rows),
        "aligned_lines": len(rows) - len(unresolved_indices),
        "unresolved_lines": len(unresolved_indices),
        "unresolved_indices": unresolved_indices,
        "method_counts": dict(sorted(method_counts.items())),
        "local_forced_aligner_lines": method_counts.get(
            "local_forced_aligner", 0
        ),
        "interpolated_lines": sum(
            count
            for method, count in method_counts.items()
            if method in interpolated_methods
        ),
        "alignment_metrics": parse_alignment_metrics(log_text),
        "error": error,
    }

    summary_path = prefix.with_suffix(".summary.json")
    _write_json_atomic(summary_path, summary)
    _write_json_atomic(diagnostics_dir / "latest.summary.json", summary)
    return summary_path
