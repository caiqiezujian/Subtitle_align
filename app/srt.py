from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def format_srt_time(seconds: float) -> str:
    milliseconds = max(0, round(float(seconds) * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def rows_to_srt(rows: Iterable[dict[str, Any]]) -> str:
    blocks: list[str] = []
    sequence = 1
    for row in rows:
        start, end = row.get("start"), row.get("end")
        if start is None or end is None:
            continue
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        blocks.append(
            f"{sequence}\n{format_srt_time(start)} --> {format_srt_time(end)}\n{text}"
        )
        sequence += 1
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for raw in handle:
            if raw.strip():
                rows.append(json.loads(raw))
    return rows


def write_outputs(
    aligned_source_path: Path, jsonl_path: Path, srt_path: Path
) -> tuple[int, int]:
    source_rows = read_jsonl(aligned_source_path)
    output: list[dict[str, Any]] = []
    resolved = 0
    for index, row in enumerate(source_rows, 1):
        timing = row.get("time") or [None, None]
        start = timing[0] if len(timing) > 0 else None
        end = timing[1] if len(timing) > 1 else None
        if start is not None and end is not None:
            start, end = round(float(start), 3), round(float(end), 3)
            resolved += 1
        item = {
            "index": index,
            "text": row.get("text", ""),
            "start": start,
            "end": end,
            "duration": round(end - start, 3) if start is not None and end is not None else None,
            "status": "aligned" if start is not None and end is not None else "unresolved",
            "method": row.get("alignment_method", "unknown"),
        }
        source = row.get("source")
        if source:
            item["source"] = source
        output.append(item)

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    srt_path.write_text(rows_to_srt(output), encoding="utf-8-sig")
    return len(output), resolved
