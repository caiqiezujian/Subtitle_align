from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


TEXT_FIELD_CANDIDATES = (
    "src",
    "text",
    "transcript",
    "content",
    "sentence",
    "original",
    "source",
    "原文",
    "字幕",
    "台词",
)


@dataclass
class TranscriptLine:
    index: int
    text: str
    source: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedTranscript:
    lines: list[TranscriptLine]
    detected_format: str
    text_field: str | None = None
    encoding: str = "utf-8"


def decode_text(data: bytes) -> tuple[str, str]:
    if not data:
        raise ValueError("字幕文件为空")
    for encoding in ("utf-8-sig", "utf-16", "gb18030", "big5"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise ValueError("无法识别字幕文件编码，请转换为 UTF-8、UTF-16、GB18030 或 Big5")


def clean_line(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    value = value.replace("\u00a0", " ").replace("\ufeff", "")
    return re.sub(r"[ \t]+", " ", value).strip()


def _choose_text_field(objects: list[dict[str, Any]], requested: str | None) -> str:
    if requested:
        if any(requested in obj for obj in objects):
            return requested
        raise ValueError(f"找不到指定文本字段：{requested}")

    keys: list[str] = []
    for obj in objects[:100]:
        for key in obj:
            if key not in keys:
                keys.append(key)

    lowered = {key.casefold(): key for key in keys}
    for candidate in TEXT_FIELD_CANDIDATES:
        if candidate.casefold() in lowered:
            return lowered[candidate.casefold()]

    scored: list[tuple[int, str]] = []
    for key in keys:
        score = sum(
            1
            for obj in objects[:100]
            if isinstance(obj.get(key), str) and clean_line(obj.get(key))
        )
        scored.append((score, key))
    scored.sort(reverse=True)
    if not scored or scored[0][0] == 0:
        raise ValueError("JSON/JSONL 中没有可用的文本字段")
    return scored[0][1]


def _from_objects(
    objects: list[dict[str, Any]], requested_field: str | None, detected_format: str
) -> ParsedTranscript:
    if not objects:
        raise ValueError("字幕文件中没有数据")
    field_name = _choose_text_field(objects, requested_field)
    lines = []
    for obj in objects:
        text = clean_line(obj.get(field_name))
        if text:
            lines.append(TranscriptLine(len(lines) + 1, text, dict(obj)))
    if not lines:
        raise ValueError(f"字段 {field_name!r} 中没有可对齐的文本")
    return ParsedTranscript(lines, detected_format, field_name)


def _parse_json(text: str, requested_field: str | None) -> ParsedTranscript:
    value = json.loads(text)
    if isinstance(value, dict):
        for key in ("lines", "items", "data", "subtitles", "segments"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
        else:
            value = [value]
    if not isinstance(value, list) or not all(isinstance(x, dict) for x in value):
        raise ValueError("JSON 必须是对象数组，或包含 lines/items/data/subtitles/segments 数组")
    return _from_objects(value, requested_field, "json")


def _parse_jsonl(text: str, requested_field: str | None) -> ParsedTranscript:
    objects: list[dict[str, Any]] = []
    for number, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSONL 第 {number} 行格式错误：{exc.msg}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"JSONL 第 {number} 行必须是对象")
        objects.append(obj)
    return _from_objects(objects, requested_field, "jsonl")


SRT_TIME_RE = re.compile(
    r"(?P<sh>\d{1,2}):(?P<sm>\d{2}):(?P<ss>\d{2})[,.](?P<sms>\d{3})\s*-->\s*"
    r"(?P<eh>\d{1,2}):(?P<em>\d{2}):(?P<es>\d{2})[,.](?P<ems>\d{3})"
)


def _srt_seconds(match: re.Match[str], prefix: str) -> float:
    return (
        int(match[f"{prefix}h"]) * 3600
        + int(match[f"{prefix}m"]) * 60
        + int(match[f"{prefix}s"])
        + int(match[f"{prefix}ms"]) / 1000
    )


def _parse_srt(text: str) -> ParsedTranscript:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", normalized.strip())
    lines: list[TranscriptLine] = []
    for block in blocks:
        parts = [x.strip() for x in block.split("\n") if x.strip()]
        time_pos = next((i for i, part in enumerate(parts) if "-->" in part), None)
        if time_pos is None:
            continue
        match = SRT_TIME_RE.search(parts[time_pos])
        if not match:
            continue
        content = clean_line(" ".join(parts[time_pos + 1 :]))
        if not content:
            continue
        source = {
            "input_start": round(_srt_seconds(match, "s"), 3),
            "input_end": round(_srt_seconds(match, "e"), 3),
        }
        lines.append(TranscriptLine(len(lines) + 1, content, source))
    if not lines:
        raise ValueError("没有识别到有效的 SRT 字幕段")
    return ParsedTranscript(lines, "srt")


def _parse_delimited(
    text: str, delimiter: str, requested_field: str | None, detected_format: str
) -> ParsedTranscript:
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    objects = [dict(row) for row in reader]
    return _from_objects(objects, requested_field, detected_format)


def _parse_plain(text: str) -> ParsedTranscript:
    lines = [clean_line(raw) for raw in text.splitlines()]
    result = [
        TranscriptLine(index + 1, value, {"physical_line": physical})
        for index, (physical, value) in enumerate(
            (x for x in enumerate(lines, 1) if x[1])
        )
    ]
    if not result:
        raise ValueError("TXT 文件中没有非空文本行")
    return ParsedTranscript(result, "txt")


def parse_transcript(
    data: bytes, filename: str = "transcript.txt", text_field: str | None = None
) -> ParsedTranscript:
    text, encoding = decode_text(data)
    suffix = Path(filename).suffix.casefold()
    stripped = text.lstrip()

    if suffix == ".srt" or ("-->" in text and SRT_TIME_RE.search(text)):
        parsed = _parse_srt(text)
    elif suffix == ".json":
        parsed = _parse_json(text, text_field)
    elif suffix == ".jsonl" or (
        stripped.startswith("{") and "\n" in stripped and not stripped.startswith("[")
    ):
        try:
            parsed = _parse_jsonl(text, text_field)
        except ValueError:
            if suffix == ".jsonl":
                raise
            parsed = _parse_json(text, text_field)
    elif suffix == ".csv":
        parsed = _parse_delimited(text, ",", text_field, "csv")
    elif suffix == ".tsv":
        parsed = _parse_delimited(text, "\t", text_field, "tsv")
    elif stripped.startswith("[") or stripped.startswith("{"):
        parsed = _parse_json(text, text_field)
    else:
        parsed = _parse_plain(text)

    parsed.encoding = encoding
    return parsed
