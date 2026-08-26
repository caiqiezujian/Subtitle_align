---
name: subtitle-alignment-ops
description: Submit, validate, inspect, and troubleshoot jobs for this repository's audio/video subtitle-alignment service. Use when preparing line-by-line transcripts, calling the service API, checking SRT/JSONL results, or diagnosing an alignment failure; do not use for translation or free-form transcription.
---

# Subtitle Alignment Ops

Use the service as a forced aligner: the transcript must contain the words actually spoken in the media, in the same order. Never substitute a translation. Do not merge, split, reorder, summarize, or creatively rewrite transcript lines.

## Prepare and submit

- Accept audio or video supported by FFmpeg and a `.txt`, `.srt`, `.jsonl`, `.json`, `.csv`, or `.tsv` transcript.
- Prefer one utterance per input line. Empty lines are harmless. For structured files, allow the service to detect `src`, `text`, `transcript`, `content`, `sentence`, `original`, `source`, `原文`, `字幕`, or `台词`; set `text_field` only when automatic selection would be ambiguous.
- Select Chinese, English, or Japanese to match the spoken audio.
- Enable `use_flash` only when the internal v4-flash endpoint is configured and the input has encoding, whitespace, or obvious formatting noise. v4-flash is a conservative cleanup fallback: it must return the same number of lines and must not receive media content.
- Use `asr_context` for names and domain terms, not as an instruction prompt.

Read [references/api.md](references/api.md) when making HTTP calls or interpreting job fields.

## Validate outputs

Download both artifacts after the job reaches `completed`.

- Confirm the SRT opens as UTF-8 in PotPlayer, cue numbers are continuous, timestamps are monotonic, and no cue has a negative or reversed duration.
- Confirm JSONL has one object per input line with `index`, `text`, `start`, `end`, `duration`, `status`, `method`, and any preserved `source` data.
- Report unresolved lines explicitly. Do not silently drop them from JSONL; unresolved lines are intentionally omitted only from SRT because SRT cannot represent a missing timestamp.

## Diagnose failures

Check `/api/health` first. A degraded response identifies missing FFmpeg or local model directories. Then inspect the job error and, when operating inside the server workspace, `data/jobs/<job-id>/alignment.log`.

- An input parsing failure: correct the file encoding, malformed structured data, or explicit field name and resubmit.
- No speech from VAD: confirm the selected media has an audible speech track.
- CUDA out of memory: keep one service worker and one GPU job, lower alignment batch settings in the engine, or disable FlashAttention only if its installation is incompatible.
- A v4-flash failure is non-fatal; the service should fall back to deterministic text cleanup.
- A service restart marks in-flight jobs failed. Resubmit rather than treating stale partial files as valid output.

Do not delete job data, expose API keys, change production configuration, or restart the service unless the user explicitly authorizes that operation.
