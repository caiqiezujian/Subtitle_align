from __future__ import annotations

import math
import unicodedata
from typing import Optional, Sequence, Tuple


TimeRange = Optional[Tuple[float, float]]


def has_spoken_content(text: str) -> bool:
    """Letters and numbers can be spoken; punctuation/symbol-only rows cannot."""
    return any(unicodedata.category(char)[0] in {"L", "N"} for char in text)


def interpolate_missing_ranges(
    texts: Sequence[str],
    ranges: Sequence[TimeRange],
    total_duration: float,
    edge_fallback_sec: float = 0.6,
) -> tuple[list[TimeRange], list[bool]]:
    """Fill unalignable runs from the surrounding resolved subtitle boundaries.

    A single missing row between two resolved rows receives exactly
    [previous.end, next.start]. Multiple consecutive rows share that gap evenly.
    Punctuation-only rows are always treated as missing even if a model emitted a
    spurious timestamp for them.
    """
    if len(texts) != len(ranges):
        raise ValueError("texts/ranges length mismatch")

    output: list[TimeRange] = []
    for text, value in zip(texts, ranges):
        valid = (
            value is not None
            and math.isfinite(float(value[0]))
            and math.isfinite(float(value[1]))
            and float(value[1]) >= float(value[0])
            and has_spoken_content(text)
        )
        output.append((float(value[0]), float(value[1])) if valid and value else None)

    interpolated = [False] * len(output)
    index = 0
    while index < len(output):
        if output[index] is not None:
            index += 1
            continue
        run_start = index
        while index < len(output) and output[index] is None:
            index += 1
        run_end = index
        count = run_end - run_start
        left = output[run_start - 1] if run_start > 0 else None
        right = output[run_end] if run_end < len(output) else None

        if left is not None and right is not None:
            gap_start = left[1]
            gap_end = max(gap_start, right[0])
        elif left is not None:
            gap_start = left[1]
            gap_end = min(total_duration, gap_start + edge_fallback_sec * count)
        elif right is not None:
            gap_end = right[0]
            gap_start = max(0.0, gap_end - edge_fallback_sec * count)
        else:
            continue

        step = max(0.0, gap_end - gap_start) / count
        for offset in range(count):
            start = gap_start + step * offset
            end = gap_end if offset == count - 1 else gap_start + step * (offset + 1)
            output[run_start + offset] = (start, end)
            interpolated[run_start + offset] = True

    return output, interpolated
