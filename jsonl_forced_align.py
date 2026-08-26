#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Long media + JSONL source-transcript forced alignment.

Input
-----
1) Audio/video:
   mp3/mp4/wav/wmv/m4a/flac/mkv/webm/... (anything FFmpeg can decode)

2) JSONL subtitles, one JSON object per physical line, for example:

   {"src": " Dr. Tara,", "show": "塔拉博士，"}
   {"src": " it's such a pleasure to meet you.", "show": "很高兴见到您。"}
   {"src": " I don't know if you know,", "show": "不知道您是否知道，"}

Run
---
CUDA_VISIBLE_DEVICES=0 python jsonl_forced_align.py \
    --media /data/demo.mp4 \
    --jsonl /data/demo.jsonl \
    --text-field src \
    --source-language English \
    --output /data/demo.aligned.jsonl \
    --flash-attn

Output
------
The original JSON object is preserved and one field is added:

   {"src":" Dr. Tara,","show":"塔拉博士，","src_time":[1.12,2.08]}

If --text-field show is used, the default added field is show_time.
You can override it explicitly with --time-field.

Architecture
------------
Long media
  -> FFmpeg: mono 16 kHz
  -> Silero VAD: speech-aware 20~90 s chunks
  -> Qwen3-ASR-1.7B (vLLM) + Qwen3-ForcedAligner timestamps
  -> global ASR token timeline
  -> monotonic token sequence alignment:
       provided JSONL field <-> ASR token timeline
  -> coarse time range for each JSONL line
  -> local exact refinement with Qwen3-ForcedAligner using the ORIGINAL line text
  -> append <field>_time to each JSON object

Important
---------
This script is intended for a field that contains the text ACTUALLY SPOKEN
in the audio, e.g. `src`.

If the selected field is a translation that is not spoken in the audio
(e.g. English speech + Chinese `show`), do NOT use forced alignment directly.
That requires the cross-lingual semantic projection pipeline instead.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
from rapidfuzz.distance import Levenshtein
from silero_vad import get_speech_timestamps, load_silero_vad

from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForceAlignProcessor

from app.alignment_policy import has_spoken_content, interpolate_missing_ranges


LOGGER = logging.getLogger("jsonl_forced_align")
SAMPLE_RATE = 16000

# ---------------------------------------------------------------------------
# Server-local model paths
# ---------------------------------------------------------------------------

MODEL_ROOT = Path(
    os.environ.get("QWEN_MODEL_ROOT", "/data/yb/Code/models")
).expanduser().resolve()

ASR_MODEL_PATH = MODEL_ROOT / "Qwen3-ASR-1.7B"
FORCED_ALIGNER_PATH = MODEL_ROOT / "Qwen3-ForcedAligner-0.6B"


LANG_ALIASES = {
    # English
    "en": "English",
    "en-us": "English",
    "english": "English",
    "英文": "English",
    "英语": "English",

    # Chinese
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "chinese": "Chinese",
    "中文": "Chinese",
    "汉语": "Chinese",
    "普通话": "Chinese",

    # Japanese
    "ja": "Japanese",
    "ja-jp": "Japanese",
    "jp": "Japanese",
    "japanese": "Japanese",
    "日本語": "Japanese",
    "日语": "Japanese",
    "日文": "Japanese",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SubtitleRow:
    index: int
    physical_line_number: int
    obj: Dict[str, Any]
    text: str
    token_start: int = 0
    token_end: int = 0  # exclusive


@dataclass
class AudioChunk:
    index: int
    start_sample: int
    end_sample: int
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


@dataclass
class TimedToken:
    text: str
    norm: str
    start: float
    end: float
    chunk_index: int


@dataclass
class LineTime:
    start: Optional[float]
    end: Optional[float]
    matched_hyp_start: Optional[int]
    matched_hyp_end: Optional[int]
    method: str


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def canonical_language(value: str) -> str:
    key = value.strip().lower()
    if key not in LANG_ALIASES:
        raise ValueError(
            f"Unsupported language {value!r}. "
            f"This script currently exposes English, Chinese, Japanese."
        )
    return LANG_ALIASES[key]


def run_checked(cmd: Sequence[str]) -> None:
    LOGGER.debug("RUN: %s", " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), check=True)


def normalize_media_with_ffmpeg(media_path: Path, wav_path: Path) -> None:
    if not media_path.exists():
        raise FileNotFoundError(f"Media file not found: {media_path}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not available in PATH.")

    wav_path.parent.mkdir(parents=True, exist_ok=True)

    run_checked(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(media_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(wav_path),
        ]
    )


def load_wav(path: Path) -> np.ndarray:
    wav, sr = sf.read(
        str(path),
        dtype="float32",
        always_2d=False,
    )
    if sr != SAMPLE_RATE:
        raise RuntimeError(f"Expected {SAMPLE_RATE} Hz, got {sr}")
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    return np.ascontiguousarray(wav, dtype=np.float32)


def validate_models() -> None:
    missing = []
    for name, path in [
        ("Qwen3-ASR-1.7B", ASR_MODEL_PATH),
        ("Qwen3-ForcedAligner-0.6B", FORCED_ALIGNER_PATH),
    ]:
        if not path.exists():
            missing.append(f"{name}: {path}")

    if missing:
        raise FileNotFoundError(
            "Missing local model directories:\n  - "
            + "\n  - ".join(missing)
            + f"\nMODEL_ROOT={MODEL_ROOT}"
        )


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def read_jsonl(path: Path, text_field: str) -> List[SubtitleRow]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    rows: List[SubtitleRow] = []

    with path.open("r", encoding="utf-8-sig") as f:
        for physical_line_number, raw in enumerate(f, start=1):
            stripped = raw.strip()

            # Strict JSONL: blank physical lines are ignored.
            if not stripped:
                continue

            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at physical line {physical_line_number}: {exc}"
                ) from exc

            if not isinstance(obj, dict):
                raise ValueError(
                    f"JSONL physical line {physical_line_number} "
                    f"must be a JSON object, got {type(obj).__name__}."
                )

            if text_field not in obj:
                raise KeyError(
                    f"JSONL physical line {physical_line_number} "
                    f"is missing field {text_field!r}."
                )

            value = obj[text_field]
            if value is None:
                text = ""
            elif isinstance(value, str):
                text = value.strip()
            else:
                raise TypeError(
                    f"Field {text_field!r} at physical line "
                    f"{physical_line_number} must be a string or null, "
                    f"got {type(value).__name__}."
                )

            rows.append(
                SubtitleRow(
                    index=len(rows),
                    physical_line_number=physical_line_number,
                    obj=obj,
                    text=text,
                )
            )

    if not rows:
        raise RuntimeError("JSONL contains no JSON objects.")

    return rows


def write_aligned_jsonl(
    rows: Sequence[SubtitleRow],
    line_times: Sequence[LineTime],
    output_path: Path,
    time_field: str,
    method_field: Optional[str] = None,
) -> None:
    if len(rows) != len(line_times):
        raise ValueError("rows/line_times length mismatch")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for row, timing in zip(rows, line_times):
            # Copy so we do not mutate input objects unexpectedly elsewhere.
            obj = dict(row.obj)

            if timing.start is None or timing.end is None:
                obj[time_field] = [None, None]
            else:
                obj[time_field] = [
                    round(float(timing.start), 3),
                    round(float(timing.end), 3),
                ]

            if method_field:
                obj[method_field] = timing.method

            f.write(
                json.dumps(
                    obj,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


# ---------------------------------------------------------------------------
# Long-audio chunking
# ---------------------------------------------------------------------------

def _find_low_energy_cut(
    wav: np.ndarray,
    target_sample: int,
    min_sample: int,
    max_sample: int,
    search_sec: float = 4.0,
    window_ms: float = 40.0,
) -> int:
    search = int(search_sec * SAMPLE_RATE)
    left = max(min_sample, target_sample - search)
    right = min(max_sample, target_sample + search)

    if right <= left:
        return int(np.clip(target_sample, min_sample, max_sample))

    window = max(1, int(window_ms / 1000.0 * SAMPLE_RATE))
    step = max(1, window // 2)

    best_pos = int(np.clip(target_sample, left, right))
    best_score = float("inf")

    pos = left
    while pos + window <= right:
        frame = wav[pos : pos + window]
        rms = float(np.sqrt(np.mean(frame * frame) + 1e-12))
        distance_penalty = (
            abs((pos + window // 2) - target_sample)
            / max(search, 1)
        )
        score = rms * (1.0 + 0.08 * distance_penalty)

        if score < best_score:
            best_score = score
            best_pos = pos + window // 2

        pos += step

    return int(np.clip(best_pos, min_sample, max_sample))


class SpeechAwareChunker:
    def __init__(
        self,
        preferred_sec: float = 50.0,
        hard_max_sec: float = 90.0,
        min_chunk_sec: float = 12.0,
        vad_threshold: float = 0.50,
        min_silence_ms: int = 300,
        speech_pad_ms: int = 120,
        edge_pad_sec: float = 0.20,
    ) -> None:
        if not (
            0 < min_chunk_sec < preferred_sec <= hard_max_sec < 180
        ):
            raise ValueError(
                "Require 0 < min_chunk_sec < preferred_sec "
                "<= hard_max_sec < 180."
            )

        self.preferred_sec = preferred_sec
        self.hard_max_sec = hard_max_sec
        self.min_chunk_sec = min_chunk_sec
        self.vad_threshold = vad_threshold
        self.min_silence_ms = min_silence_ms
        self.speech_pad_ms = speech_pad_ms
        self.edge_pad_sec = edge_pad_sec

        # Keep VAD on CPU.
        self.vad = load_silero_vad()

    @torch.inference_mode()
    def split(self, wav: np.ndarray) -> List[AudioChunk]:
        tensor = torch.from_numpy(wav)

        speech = get_speech_timestamps(
            tensor,
            self.vad,
            sampling_rate=SAMPLE_RATE,
            threshold=self.vad_threshold,
            min_silence_duration_ms=self.min_silence_ms,
            speech_pad_ms=self.speech_pad_ms,
            return_seconds=False,
        )

        if not speech:
            return []

        edge_pad = int(self.edge_pad_sec * SAMPLE_RATE)
        total_samples = len(wav)

        speech_start = max(
            0,
            int(speech[0]["start"]) - edge_pad,
        )
        speech_end = min(
            total_samples,
            int(speech[-1]["end"]) + edge_pad,
        )

        # Candidate cuts are midpoints of VAD silence gaps.
        silence_candidates: List[int] = []
        for a, b in zip(speech[:-1], speech[1:]):
            gap_start = int(a["end"])
            gap_end = int(b["start"])
            if gap_end > gap_start:
                silence_candidates.append(
                    (gap_start + gap_end) // 2
                )

        preferred = int(self.preferred_sec * SAMPLE_RATE)
        hard_max = int(self.hard_max_sec * SAMPLE_RATE)
        min_chunk = int(self.min_chunk_sec * SAMPLE_RATE)

        boundaries = [speech_start]
        cur = speech_start

        while speech_end - cur > hard_max:
            min_allowed = cur + min_chunk
            max_allowed = min(
                speech_end,
                cur + hard_max,
            )
            target = min(
                cur + preferred,
                max_allowed,
            )

            candidates = [
                x
                for x in silence_candidates
                if min_allowed <= x <= max_allowed
            ]

            if candidates:
                cut = min(
                    candidates,
                    key=lambda x: abs(x - target),
                )
            else:
                cut = _find_low_energy_cut(
                    wav=wav,
                    target_sample=target,
                    min_sample=min_allowed,
                    max_sample=max_allowed,
                )

            if cut <= cur:
                cut = max_allowed

            boundaries.append(cut)
            cur = cut

        boundaries.append(speech_end)

        # Merge a very short tail if doing so stays under hard max.
        if len(boundaries) >= 3:
            tail = boundaries[-1] - boundaries[-2]
            merged = boundaries[-1] - boundaries[-3]
            if tail < min_chunk and merged <= hard_max:
                del boundaries[-2]

        chunks: List[AudioChunk] = []
        for idx, (s, e) in enumerate(
            zip(boundaries[:-1], boundaries[1:])
        ):
            if e <= s:
                continue

            chunks.append(
                AudioChunk(
                    index=idx,
                    start_sample=int(s),
                    end_sample=int(e),
                    start_sec=s / SAMPLE_RATE,
                    end_sec=e / SAMPLE_RATE,
                )
            )

        return chunks


# ---------------------------------------------------------------------------
# Qwen3 ASR + ForcedAligner
# ---------------------------------------------------------------------------

class QwenEngine:
    def __init__(
        self,
        gpu_memory_utilization: float = 0.65,
        max_inference_batch_size: int = 32,
        max_new_tokens: int = 2048,
        use_flash_attention: bool = False,
    ) -> None:
        aligner_kwargs: Dict[str, Any] = {
            "dtype": torch.bfloat16,
            "device_map": "cuda:0",
        }

        if use_flash_attention:
            aligner_kwargs[
                "attn_implementation"
            ] = "flash_attention_2"

        LOGGER.info(
            "Loading ASR(vLLM): %s",
            ASR_MODEL_PATH,
        )
        LOGGER.info(
            "Loading ForcedAligner: %s",
            FORCED_ALIGNER_PATH,
        )

        self.asr = Qwen3ASRModel.LLM(
            model=str(ASR_MODEL_PATH),
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=1,
            dtype="bfloat16",
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
            forced_aligner=str(FORCED_ALIGNER_PATH),
            forced_aligner_kwargs=aligner_kwargs,
        )

        if self.asr.forced_aligner is None:
            raise RuntimeError(
                "Qwen3ASRModel was initialized without a forced aligner."
            )

        # Reuse the exact processor owned by the loaded ForcedAligner.
        self.fa_processor: Qwen3ForceAlignProcessor = (
            self.asr.forced_aligner.aligner_processor
        )

    def tokenize_text(
        self,
        text: str,
        language: str,
    ) -> List[str]:
        words, _ = self.fa_processor.encode_timestamp(
            text,
            language,
        )
        return list(words)

    def transcribe_chunks(
        self,
        wav: np.ndarray,
        chunks: Sequence[AudioChunk],
        language: str,
        batch_size: int,
        context: str = "",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[TimedToken]:
        output: List[TimedToken] = []

        for base in range(0, len(chunks), batch_size):
            batch_chunks = list(
                chunks[base : base + batch_size]
            )

            audios = [
                (
                    np.ascontiguousarray(
                        wav[c.start_sample : c.end_sample]
                    ),
                    SAMPLE_RATE,
                )
                for c in batch_chunks
            ]

            LOGGER.info(
                "ASR timestamp batch %d..%d / %d",
                base,
                base + len(batch_chunks) - 1,
                len(chunks),
            )

            results = self.asr.transcribe(
                audio=audios,
                context=[context] * len(audios),
                language=[language] * len(audios),
                return_time_stamps=True,
            )

            for chunk, result in zip(
                batch_chunks,
                results,
            ):
                if result.time_stamps is None:
                    continue

                for item in result.time_stamps:
                    text = str(item.text)
                    norm = normalize_token(text)

                    if not norm:
                        continue

                    output.append(
                        TimedToken(
                            text=text,
                            norm=norm,
                            start=(
                                float(item.start_time)
                                + chunk.start_sec
                            ),
                            end=(
                                float(item.end_time)
                                + chunk.start_sec
                            ),
                            chunk_index=chunk.index,
                        )
                    )

            if progress_callback:
                progress_callback(
                    min(base + len(batch_chunks), len(chunks)),
                    len(chunks),
                )

        # Chunk order should already be monotonic, but sort defensively.
        output.sort(
            key=lambda x: (
                x.start,
                x.end,
            )
        )
        return output

    @torch.inference_mode()
    def refine_lines(
        self,
        wav: np.ndarray,
        rows: Sequence[SubtitleRow],
        coarse_times: Sequence[LineTime],
        language: str,
        batch_size: int = 24,
        margin_sec: float = 1.25,
        max_window_sec: float = 45.0,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[LineTime]:
        """
        Exact local forced alignment using the ORIGINAL JSONL line text.

        The coarse ASR/token match provides a local audio window.
        Qwen3-ForcedAligner then aligns the exact user-provided transcript
        inside that window.

        If a line has no coarse time, it stays unresolved.
        """
        refined = list(coarse_times)

        jobs: List[
            Tuple[
                int,
                float,
                np.ndarray,
                str,
            ]
        ] = []

        total_duration = len(wav) / SAMPLE_RATE

        for row, coarse in zip(
            rows,
            coarse_times,
        ):
            if not row.text:
                continue

            if (
                coarse.start is None
                or coarse.end is None
            ):
                continue

            # Larger text gets slightly more context.
            token_count = max(
                1,
                row.token_end - row.token_start,
            )
            adaptive_margin = min(
                3.0,
                margin_sec
                + 0.025 * token_count,
            )

            local_start = max(
                0.0,
                coarse.start - adaptive_margin,
            )
            local_end = min(
                total_duration,
                coarse.end + adaptive_margin,
            )

            # Protect against pathological coarse mappings.
            if local_end - local_start > max_window_sec:
                center = (
                    coarse.start + coarse.end
                ) / 2.0
                half = max_window_sec / 2.0
                local_start = max(
                    0.0,
                    center - half,
                )
                local_end = min(
                    total_duration,
                    center + half,
                )

            s = int(
                round(local_start * SAMPLE_RATE)
            )
            e = int(
                round(local_end * SAMPLE_RATE)
            )

            if e <= s:
                continue

            clip = np.ascontiguousarray(
                wav[s:e],
                dtype=np.float32,
            )

            jobs.append(
                (
                    row.index,
                    local_start,
                    clip,
                    row.text,
                )
            )

        LOGGER.info(
            "Local exact ForcedAligner jobs: %d",
            len(jobs),
        )

        for base in range(0, len(jobs), batch_size):
            batch = jobs[
                base : base + batch_size
            ]

            audios = [
                (x[2], SAMPLE_RATE)
                for x in batch
            ]
            texts = [
                x[3]
                for x in batch
            ]

            LOGGER.info(
                "Exact FA batch %d..%d / %d",
                base,
                base + len(batch) - 1,
                len(jobs),
            )

            results = self.asr.forced_aligner.align(
                audio=audios,
                text=texts,
                language=[language] * len(batch),
            )

            for job, result in zip(
                batch,
                results,
            ):
                row_idx, local_start, _, _ = job

                if len(result) == 0:
                    continue

                start = (
                    local_start
                    + float(result[0].start_time)
                )
                end = (
                    local_start
                    + float(result[-1].end_time)
                )

                # Sanity checks.
                if (
                    not math.isfinite(start)
                    or not math.isfinite(end)
                    or end < start
                ):
                    continue

                refined[row_idx] = LineTime(
                    start=start,
                    end=end,
                    matched_hyp_start=(
                        coarse_times[
                            row_idx
                        ].matched_hyp_start
                    ),
                    matched_hyp_end=(
                        coarse_times[
                            row_idx
                        ].matched_hyp_end
                    ),
                    method="local_forced_aligner",
                )

            if progress_callback:
                progress_callback(
                    min(base + len(batch), len(jobs)),
                    len(jobs),
                )

        return refined


# ---------------------------------------------------------------------------
# Text/token alignment
# ---------------------------------------------------------------------------

def normalize_token(token: str) -> str:
    """
    Normalization only for sequence matching.

    Forced alignment itself always uses ORIGINAL text.
    """
    s = unicodedata.normalize(
        "NFKC",
        str(token),
    )
    return s.casefold().strip()


def tokenize_jsonl_rows(
    engine: QwenEngine,
    rows: Sequence[SubtitleRow],
    language: str,
) -> List[str]:
    all_tokens: List[str] = []

    for row in rows:
        row.token_start = len(all_tokens)

        if row.text and has_spoken_content(row.text):
            tokens = engine.tokenize_text(
                row.text,
                language,
            )
            all_tokens.extend(
                [
                    normalize_token(t)
                    for t in tokens
                    if normalize_token(t)
                ]
            )

        row.token_end = len(all_tokens)

    return all_tokens


def build_ref_to_hyp_mapping(
    ref_tokens: Sequence[str],
    hyp_tokens: Sequence[str],
) -> List[Optional[int]]:
    """
    Monotonic Levenshtein alignment.

    ref_tokens: exact JSONL transcript token sequence
    hyp_tokens: Qwen ASR token sequence

    The output maps each ref token index to one approximate ASR token index.
    Exact/equal blocks are 1:1. Replace blocks are projected proportionally.
    Deleted ref tokens remain None and are later interpolated.
    """
    mapping: List[Optional[int]] = [
        None
    ] * len(ref_tokens)

    opcodes = Levenshtein.opcodes(
        ref_tokens,
        hyp_tokens,
    )

    for op in opcodes:
        tag = op.tag
        i1, i2 = int(op.src_start), int(op.src_end)
        j1, j2 = int(op.dest_start), int(op.dest_end)

        ref_len = i2 - i1
        hyp_len = j2 - j1

        if tag == "equal":
            for k in range(
                min(ref_len, hyp_len)
            ):
                mapping[i1 + k] = j1 + k

        elif tag == "replace":
            if ref_len <= 0 or hyp_len <= 0:
                continue

            # Monotonic proportional projection inside a mismatched region.
            for k in range(ref_len):
                pos = (
                    (k + 0.5)
                    / ref_len
                )
                j = j1 + min(
                    hyp_len - 1,
                    int(pos * hyp_len),
                )
                mapping[i1 + k] = j

        elif tag == "delete":
            # ref tokens missing from ASR -> interpolate later.
            continue

        elif tag == "insert":
            # ASR hallucination/addition has no ref token.
            continue

    return mapping


def nearest_mapped_left(
    mapping: Sequence[Optional[int]],
    pos: int,
) -> Optional[int]:
    for i in range(pos, -1, -1):
        if mapping[i] is not None:
            return mapping[i]
    return None


def nearest_mapped_right(
    mapping: Sequence[Optional[int]],
    pos: int,
) -> Optional[int]:
    for i in range(
        pos,
        len(mapping),
    ):
        if mapping[i] is not None:
            return mapping[i]
    return None


def coarse_line_times(
    rows: Sequence[SubtitleRow],
    mapping: Sequence[Optional[int]],
    hyp_tokens: Sequence[TimedToken],
) -> List[LineTime]:
    output: List[LineTime] = []

    for row in rows:
        if row.token_end <= row.token_start:
            output.append(
                LineTime(
                    start=None,
                    end=None,
                    matched_hyp_start=None,
                    matched_hyp_end=None,
                    method="empty_text",
                )
            )
            continue

        mapped = [
            mapping[i]
            for i in range(
                row.token_start,
                row.token_end,
            )
            if mapping[i] is not None
        ]

        if mapped:
            j0 = max(
                0,
                min(mapped),
            )
            j1 = min(
                len(hyp_tokens) - 1,
                max(mapped),
            )

            output.append(
                LineTime(
                    start=hyp_tokens[j0].start,
                    end=hyp_tokens[j1].end,
                    matched_hyp_start=j0,
                    matched_hyp_end=j1,
                    method="asr_token_alignment",
                )
            )
            continue

        # Entire line was deleted/missed by ASR. Estimate using neighboring
        # mapped transcript tokens, then let the local ForcedAligner refine it.
        left = nearest_mapped_left(
            mapping,
            row.token_start - 1,
        )
        right = nearest_mapped_right(
            mapping,
            row.token_end,
        )

        if left is not None and right is not None:
            left_t = hyp_tokens[left].end
            right_t = hyp_tokens[right].start

            if right_t >= left_t:
                # Allocate the gap itself.
                output.append(
                    LineTime(
                        start=left_t,
                        end=right_t,
                        matched_hyp_start=left,
                        matched_hyp_end=right,
                        method="neighbor_gap_interpolation",
                    )
                )
                continue

        if left is not None:
            t = hyp_tokens[left].end
            output.append(
                LineTime(
                    start=t,
                    end=min(
                        t + 2.0,
                        hyp_tokens[-1].end,
                    ),
                    matched_hyp_start=left,
                    matched_hyp_end=left,
                    method="left_neighbor_fallback",
                )
            )
            continue

        if right is not None:
            t = hyp_tokens[right].start
            output.append(
                LineTime(
                    start=max(0.0, t - 2.0),
                    end=t,
                    matched_hyp_start=right,
                    matched_hyp_end=right,
                    method="right_neighbor_fallback",
                )
            )
            continue

        output.append(
            LineTime(
                start=None,
                end=None,
                matched_hyp_start=None,
                matched_hyp_end=None,
                method="unresolved",
            )
        )

    return output


def enforce_monotonic_line_times(
    times: Sequence[LineTime],
    min_duration: float = 0.04,
) -> List[LineTime]:
    """
    Final conservative sanity pass.

    Genuine pauses are preserved, but adjacent lines are not allowed to
    overlap. For every resolved line, this guarantees:

        current.start >= previous.end
        current.end >= current.start + min_duration

    Unresolved lines remain unchanged and do not reset the previous valid
    ending timestamp.
    """
    out: List[LineTime] = []
    prev_end = 0.0

    for item in times:
        if item.start is None or item.end is None:
            out.append(item)
            continue

        start = max(
            0.0,
            float(item.start),
            prev_end,
        )
        end = max(
            float(item.end),
            start + min_duration,
        )

        out.append(
            LineTime(
                start=start,
                end=end,
                matched_hyp_start=item.matched_hyp_start,
                matched_hyp_end=item.matched_hyp_end,
                method=item.method,
            )
        )
        prev_end = end

    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_alignment(
    args: argparse.Namespace,
    engine: Optional[QwenEngine] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> None:
    def progress(value: int, stage: str) -> None:
        if progress_callback:
            progress_callback(value, stage)

    validate_models()
    progress(23, "检查模型与输入")

    language = canonical_language(
        args.source_language
    )

    text_field = args.text_field
    time_field = (
        args.time_field
        if args.time_field
        else f"{text_field}_time"
    )

    rows = read_jsonl(
        args.jsonl,
        text_field=text_field,
    )

    LOGGER.info(
        "JSONL rows: %d | text_field=%s | time_field=%s",
        len(rows),
        text_field,
        time_field,
    )

    if engine is None:
        engine = QwenEngine(
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_inference_batch_size=args.max_inference_batch_size,
            max_new_tokens=args.max_new_tokens,
            use_flash_attention=args.flash_attn,
        )

    # Tokenize exact JSONL transcript before media processing.
    ref_tokens = tokenize_jsonl_rows(
        engine=engine,
        rows=rows,
        language=language,
    )
    progress(25, "逐行原文分词完成")

    LOGGER.info(
        "Reference transcript tokens: %d",
        len(ref_tokens),
    )

    if not ref_tokens:
        raise RuntimeError(
            f"Field {text_field!r} contains no alignable text."
        )

    with tempfile.TemporaryDirectory(
        prefix="qwen_jsonl_align_"
    ) as td:
        td_path = Path(td)
        normalized_wav = (
            td_path / "media_16k_mono.wav"
        )

        LOGGER.info(
            "Normalizing media with FFmpeg: %s",
            args.media,
        )
        progress(27, "正在提取并标准化音轨")
        normalize_media_with_ffmpeg(
            args.media,
            normalized_wav,
        )

        wav = load_wav(
            normalized_wav
        )

        duration = len(wav) / SAMPLE_RATE
        progress(30, "音轨标准化完成")
        LOGGER.info(
            "Media audio duration: %.3f sec",
            duration,
        )

        chunker = SpeechAwareChunker(
            preferred_sec=args.preferred_chunk_sec,
            hard_max_sec=args.hard_max_chunk_sec,
            min_chunk_sec=args.min_chunk_sec,
            vad_threshold=args.vad_threshold,
            min_silence_ms=args.min_silence_ms,
        )

        chunks = chunker.split(wav)
        progress(34, f"语音切分完成，共 {len(chunks)} 段")

        if not chunks:
            raise RuntimeError(
                "VAD found no speech in the media."
            )

        LOGGER.info(
            "Speech chunks: %d | mean %.2fs | max %.2fs",
            len(chunks),
            float(
                np.mean(
                    [x.duration_sec for x in chunks]
                )
            ),
            max(
                x.duration_sec
                for x in chunks
            ),
        )

        # Pass A: vLLM ASR + ASR-transcript forced timestamps.
        hyp_timed_tokens = engine.transcribe_chunks(
            wav=wav,
            chunks=chunks,
            language=language,
            batch_size=args.asr_batch_size,
            context=args.asr_context,
            progress_callback=lambda done, total: progress(
                35 + round(31 * done / max(1, total)),
                f"ASR 语音识别 {done}/{total} 段",
            ),
        )

        if not hyp_timed_tokens:
            raise RuntimeError(
                "ASR produced no timestamped tokens."
            )

        hyp_tokens = [
            x.norm
            for x in hyp_timed_tokens
        ]

        LOGGER.info(
            "ASR timestamp tokens: %d",
            len(hyp_tokens),
        )
        progress(68, "ASR 完成，正在全局匹配原文")

        # Global monotonic text alignment.
        t_align = time.time()
        mapping = build_ref_to_hyp_mapping(
            ref_tokens=ref_tokens,
            hyp_tokens=hyp_tokens,
        )
        progress(71, "全局单调文本匹配完成")
        LOGGER.info(
            "Global token alignment completed in %.2fs",
            time.time() - t_align,
        )

        mapped_count = sum(
            x is not None
            for x in mapping
        )
        LOGGER.info(
            "Mapped reference tokens: %d/%d (%.2f%%)",
            mapped_count,
            len(mapping),
            (
                100.0
                * mapped_count
                / max(1, len(mapping))
            ),
        )

        coarse = coarse_line_times(
            rows=rows,
            mapping=mapping,
            hyp_tokens=hyp_timed_tokens,
        )

        # Pass B: exact local FA with each original JSONL field string.
        if args.no_local_refine:
            final_times = coarse
            progress(91, "已完成逐行粗对齐")
        else:
            final_times = engine.refine_lines(
                wav=wav,
                rows=rows,
                coarse_times=coarse,
                language=language,
                batch_size=args.fa_batch_size,
                margin_sec=args.refine_margin_sec,
                max_window_sec=args.max_refine_window_sec,
                progress_callback=lambda done, total: progress(
                    73 + round(18 * done / max(1, total)),
                    f"ForcedAligner 精确校准 {done}/{total} 行",
                ),
            )
            progress(91, "ForcedAligner 精确校准完成")

        interpolated_ranges, interpolated = interpolate_missing_ranges(
            texts=[row.text for row in rows],
            ranges=[
                (item.start, item.end)
                if item.start is not None and item.end is not None
                else None
                for item in final_times
            ],
            total_duration=duration,
        )
        adjusted_times: List[LineTime] = []
        for row, original, value, was_interpolated in zip(
            rows, final_times, interpolated_ranges, interpolated
        ):
            if value is None:
                adjusted_times.append(original)
                continue
            if was_interpolated:
                method = (
                    "punctuation_gap_interpolation"
                    if not has_spoken_content(row.text)
                    else "unresolved_gap_interpolation"
                )
                adjusted_times.append(
                    LineTime(
                        start=value[0],
                        end=value[1],
                        matched_hyp_start=None,
                        matched_hyp_end=None,
                        method=method,
                    )
                )
            else:
                adjusted_times.append(original)
        final_times = adjusted_times
        progress(93, "已补全标点与未解析行时间")

        final_times = enforce_monotonic_line_times(
            final_times
        )

    write_aligned_jsonl(
        rows=rows,
        line_times=final_times,
        output_path=args.output,
        time_field=time_field,
        method_field=args.method_field,
    )
    progress(94, "对齐时间轴写入完成")

    unresolved = sum(
        1
        for x in final_times
        if x.start is None
        or x.end is None
    )
    refined = sum(
        1
        for x in final_times
        if x.method
        == "local_forced_aligner"
    )

    LOGGER.info(
        "Output: %s",
        args.output,
    )
    LOGGER.info(
        "Lines=%d | refined=%d | unresolved=%d",
        len(rows),
        refined,
        unresolved,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Align a spoken-source field in a JSONL subtitle file "
            "to long audio/video and append <field>_time=[start,end]."
        ),
    )

    p.add_argument(
        "--media",
        type=Path,
        required=True,
        help=(
            "Input audio/video, e.g. mp3/mp4/wav/wmv/"
            "m4a/flac/mkv/webm."
        ),
    )

    p.add_argument(
        "--jsonl",
        type=Path,
        required=True,
        help="Input JSONL subtitle file.",
    )

    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL with the added timestamp field.",
    )

    p.add_argument(
        "--text-field",
        default="src",
        help=(
            "JSON field containing text actually spoken in the audio."
        ),
    )

    p.add_argument(
        "--time-field",
        default=None,
        help=(
            "Output timestamp field. "
            "Default: <text-field>_time."
        ),
    )

    p.add_argument(
        "--method-field",
        default=None,
        help=(
            "Optional output field containing the alignment method used "
            "for each line."
        ),
    )

    p.add_argument(
        "--source-language",
        default="English",
        help="English / Chinese / Japanese (aliases supported).",
    )

    p.add_argument(
        "--asr-context",
        default="",
        help=(
            "Optional Qwen ASR context/hotwords. "
            "Use for names/technical terms if useful."
        ),
    )

    # A800 defaults
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.65,
        help=(
            "vLLM GPU-memory fraction. Leave headroom for "
            "the ForcedAligner on the same GPU."
        ),
    )

    p.add_argument(
        "--asr-batch-size",
        type=int,
        default=16,
    )

    p.add_argument(
        "--fa-batch-size",
        type=int,
        default=24,
        help="Batch size for local exact forced alignment.",
    )

    p.add_argument(
        "--max-inference-batch-size",
        type=int,
        default=32,
    )

    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
    )

    # Long-audio chunking
    p.add_argument(
        "--preferred-chunk-sec",
        type=float,
        default=50.0,
    )

    p.add_argument(
        "--hard-max-chunk-sec",
        type=float,
        default=90.0,
    )

    p.add_argument(
        "--min-chunk-sec",
        type=float,
        default=12.0,
    )

    p.add_argument(
        "--vad-threshold",
        type=float,
        default=0.50,
    )

    p.add_argument(
        "--min-silence-ms",
        type=int,
        default=300,
    )

    # Fine alignment
    p.add_argument(
        "--refine-margin-sec",
        type=float,
        default=1.25,
        help=(
            "Extra audio context around coarse line range before "
            "exact ForcedAligner refinement."
        ),
    )

    p.add_argument(
        "--max-refine-window-sec",
        type=float,
        default=45.0,
        help=(
            "Safety cap for one line's exact ForcedAligner audio window."
        ),
    )

    p.add_argument(
        "--no-local-refine",
        action="store_true",
        help=(
            "Skip second-pass exact line ForcedAligner; "
            "use ASR token mapping only."
        ),
    )

    p.add_argument(
        "--flash-attn",
        action="store_true",
        help=(
            "Use FlashAttention2 for ForcedAligner "
            "(requires flash-attn installed)."
        ),
    )

    p.add_argument(
        "--log-level",
        default="INFO",
        choices=[
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
        ],
    )

    return p


def main() -> None:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=getattr(
            logging,
            args.log_level,
        ),
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required."
        )

    run_alignment(args)


if __name__ == "__main__":
    # Important for vLLM multiprocessing/spawn.
    main()
