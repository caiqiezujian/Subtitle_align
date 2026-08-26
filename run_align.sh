#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="/data/yb/Code/Subtitle_align/zh_7_external/output/zh_external_7.jsonl"

mkdir -p "$(dirname "$OUTPUT")"

CUDA_VISIBLE_DEVICES=5 python "$SCRIPT_DIR/jsonl_forced_align.py" \
    --media /data/yb/Code/Subtitle_align/zh_7_external/zh_external_7.wav \
    --jsonl /data/yb/Code/Subtitle_align/zh_7_external/output_merged.jsonl \
    --text-field src \
    --source-language Chinese \
    --output "$OUTPUT"
