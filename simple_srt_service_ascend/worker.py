from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any


PROTOCOL_PREFIX = "__SUBALIGN_ASCEND__"


def emit(event: str, **payload: object) -> None:
    print(
        PROTOCOL_PREFIX
        + json.dumps(
            {"event": event, **payload}, ensure_ascii=False, separators=(",", ":")
        ),
        flush=True,
    )


def build_worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent Ascend subtitle worker")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--max-inference-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--attention-implementation", choices=["eager", "sdpa"], default="eager"
    )
    return parser


def build_ascend_engine(
    aligner: Any,
    torch: Any,
    *,
    device: str,
    max_inference_batch_size: int,
    max_new_tokens: int,
    attention_implementation: str,
) -> Any:
    """Load Qwen ASR + ForcedAligner on one NPU.

    The instance reuses the mature tokenization/alignment methods on
    ``QwenEngine`` but deliberately bypasses its CUDA/vLLM constructor.
    """
    engine = object.__new__(aligner.QwenEngine)

    model_kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "device_map": {"": device},
        "attn_implementation": attention_implementation,
    }
    forced_aligner_kwargs = dict(model_kwargs)

    logging.getLogger(__name__).info(
        "Loading ASR(Transformers) on %s: %s", device, aligner.ASR_MODEL_PATH
    )
    logging.getLogger(__name__).info(
        "Loading ForcedAligner on %s: %s", device, aligner.FORCED_ALIGNER_PATH
    )
    engine.asr = aligner.Qwen3ASRModel.from_pretrained(
        str(aligner.ASR_MODEL_PATH),
        forced_aligner=str(aligner.FORCED_ALIGNER_PATH),
        forced_aligner_kwargs=forced_aligner_kwargs,
        max_inference_batch_size=max_inference_batch_size,
        max_new_tokens=max_new_tokens,
        **model_kwargs,
    )

    if engine.asr.forced_aligner is None:
        raise RuntimeError("Qwen3ASRModel 未加载 ForcedAligner")

    engine.asr.model.eval()
    model_device = next(engine.asr.model.parameters()).device
    if model_device.type != "npu":
        raise RuntimeError(f"ASR 模型未加载到 NPU，实际设备为 {model_device}")
    engine.fa_processor = engine.asr.forced_aligner.aligner_processor
    torch.npu.synchronize()
    logging.getLogger(__name__).info(
        "Ascend models ready | device=%s | allocated=%.2f GiB",
        model_device,
        torch.npu.memory_allocated(model_device) / 1024**3,
    )
    return engine


def main() -> None:
    args = build_worker_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    command: dict[str, object] = {}
    try:
        import torch
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            raise RuntimeError("torch_npu 已导入，但 torch.npu.is_available() 为 False")
        torch.npu.set_device(args.device)

        import jsonl_forced_align as aligner

        aligner.validate_models()
        engine = build_ascend_engine(
            aligner,
            torch,
            device=args.device,
            max_inference_batch_size=args.max_inference_batch_size,
            max_new_tokens=args.max_new_tokens,
            attention_implementation=args.attention_implementation,
        )
    except Exception as exc:
        emit("startup_error", error=str(exc), traceback=traceback.format_exc())
        raise

    emit("ready")

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            command = json.loads(raw)
            if command.get("command") == "shutdown":
                emit("stopped")
                return
            if command.get("command") != "align":
                raise ValueError("Unknown NPU worker command")

            job_id = str(command["job_id"])
            log_path = Path(str(command["log_path"]))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(log_path, encoding="utf-8")
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
                )
            )
            logging.getLogger().addHandler(handler)
            try:
                job_args = aligner.build_parser().parse_args(command["argv"])
                aligner.run_alignment(
                    job_args,
                    engine=engine,
                    progress_callback=lambda progress, stage: emit(
                        "progress",
                        job_id=job_id,
                        progress=progress,
                        stage=stage,
                    ),
                )
            finally:
                logging.getLogger().removeHandler(handler)
                handler.close()
            emit("completed", job_id=job_id)
        except Exception as exc:
            emit(
                "job_error",
                job_id=str(command.get("job_id", "")),
                error=str(exc),
                traceback=traceback.format_exc(),
            )


if __name__ == "__main__":
    main()
