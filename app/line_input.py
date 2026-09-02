from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MAX_LINE_COUNT = 100_000
MAX_TOTAL_CHARACTERS = 10_000_000


def parse_lines_json(value: str) -> list[str]:
    """Parse a JSON string array without merging, splitting, or reordering."""
    try:
        parsed: Any = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"lines 不是有效的 JSON：{exc.msg}") from exc

    if not isinstance(parsed, list):
        raise ValueError("lines 必须是 JSON 字符串数组")
    if not parsed:
        raise ValueError("lines 不能为空数组")
    if len(parsed) > MAX_LINE_COUNT:
        raise ValueError(f"lines 不能超过 {MAX_LINE_COUNT} 行")

    lines: list[str] = []
    total_characters = 0
    invalid_indices: list[int] = []
    empty_indices: list[int] = []
    for index, item in enumerate(parsed, 1):
        if not isinstance(item, str):
            invalid_indices.append(index)
            continue
        if not item.strip():
            empty_indices.append(index)
        total_characters += len(item)
        lines.append(item)

    if invalid_indices:
        preview = ", ".join(map(str, invalid_indices[:10]))
        raise ValueError(f"lines 中以下位置不是字符串：{preview}")
    if empty_indices:
        preview = ", ".join(map(str, empty_indices[:10]))
        raise ValueError(f"lines 中以下位置是空字符串：{preview}")
    if total_characters > MAX_TOTAL_CHARACTERS:
        raise ValueError(
            f"lines 总字符数不能超过 {MAX_TOTAL_CHARACTERS}，"
            f"当前为 {total_characters}"
        )
    return lines


def detect_spoken_language(lines: list[str]) -> str:
    """Choose the language accepted by Qwen3-ASR from transcript characters."""
    text = "\n".join(lines)
    if any(
        0x3040 <= ord(char) <= 0x30FF or 0x31F0 <= ord(char) <= 0x31FF
        for char in text
    ):
        return "Japanese"
    if any(
        0x3400 <= ord(char) <= 0x4DBF or 0x4E00 <= ord(char) <= 0x9FFF
        for char in text
    ):
        return "Chinese"
    return "English"


def write_lines_jsonl(lines: list[str], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for index, text in enumerate(lines, 1):
            handle.write(
                json.dumps(
                    {"text": text, "source_index": index},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
