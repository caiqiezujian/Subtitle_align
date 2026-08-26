from __future__ import annotations

import json
import logging
import time
from dataclasses import replace
from typing import Any, Callable, Optional

import httpx

from .config import Settings
from .transcript import ParsedTranscript, clean_line


LOGGER = logging.getLogger("subtitle_align.flash")


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from pure JSON, fenced JSON, or surrounding prose."""
    if not text or not text.strip():
        return None
    value = text.strip()
    candidates = [value]
    lines = value.splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    unfenced = "\n".join(lines).strip()
    if unfenced and unfenced != value:
        candidates.append(unfenced)
    start, end = value.find("{"), value.rfind("}")
    if start >= 0 and end > start:
        candidates.append(value[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


class FlashAssistant:
    """Optional OpenAI-compatible v4-flash adapter with strict fallbacks."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def available(self) -> bool:
        return bool(self.settings.flash_enabled and self.settings.flash_base_url)

    def normalize(
        self,
        transcript: ParsedTranscript,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> ParsedTranscript:
        if not self.available:
            return transcript

        normalized = []
        total_batches = max(1, (len(transcript.lines) + 99) // 100)
        for offset in range(0, len(transcript.lines), 100):
            batch = transcript.lines[offset : offset + 100]
            try:
                cleaned = self._normalize_batch([line.text for line in batch])
            except Exception as exc:  # The alignment service must remain usable.
                LOGGER.warning("v4-flash cleanup skipped for batch: %s", exc)
                cleaned = [line.text for line in batch]
            for line, text in zip(batch, cleaned):
                normalized.append(replace(line, text=text))
            if progress_callback:
                progress_callback(offset // 100 + 1, total_batches)
        return replace(transcript, lines=normalized)

    def _normalize_batch(self, lines: list[str]) -> list[str]:
        url = f"{self.settings.flash_base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.settings.flash_api_key:
            headers["Authorization"] = f"Bearer {self.settings.flash_api_key}"
        payload = {
            "model": self.settings.flash_model,
            "temperature": 0,
            "max_tokens": 8192,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是语音强制对齐前的文本清洗器。仅修复乱码、异常空白和明显的转写格式噪声；"
                        "不要翻译，不要改写，不要增删、合并、拆分或重排行。"
                        "返回严格 JSON：{\"lines\":[...]}，数组长度必须与输入相同。"
                        "只输出 JSON 对象，不要输出 markdown 围栏或额外解释。"
                    ),
                },
                {"role": "user", "content": json.dumps({"lines": lines}, ensure_ascii=False)},
            ],
        }
        last_error: Exception | None = None
        content = ""
        for attempt in range(self.settings.flash_retry_count + 1):
            try:
                # The internal service must bypass Huawei MWG/system proxies.
                with httpx.Client(
                    trust_env=False,
                    verify=self.settings.flash_verify_ssl,
                    timeout=self.settings.flash_timeout_seconds,
                ) as client:
                    response = client.post(url, headers=headers, json=payload)
                if response.status_code != 200:
                    raise RuntimeError(
                        f"v4-flash HTTP {response.status_code}: {response.text[:500]}"
                    )
                if not response.text or not response.text.strip():
                    raise RuntimeError("v4-flash 返回空响应")
                body = response.json()
                content = body["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("v4-flash choices 中没有文本内容")
                break
            except Exception as exc:
                last_error = exc
                if attempt >= self.settings.flash_retry_count:
                    raise RuntimeError(
                        f"v4-flash 请求失败（已重试 {self.settings.flash_retry_count} 次）：{exc}"
                    ) from exc
                time.sleep(attempt + 1)

        value = extract_json_object(content)
        if value is None:
            raise ValueError("v4-flash 没有返回合法 JSON 对象") from last_error
        result = value.get("lines")
        if not isinstance(result, list) or len(result) != len(lines):
            raise ValueError("v4-flash returned an invalid line count")
        cleaned = [clean_line(item) for item in result]
        if any(not item for item in cleaned):
            raise ValueError("v4-flash returned an empty line")
        return cleaned
