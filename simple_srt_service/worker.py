from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any


PROTOCOL_PREFIX = "__SUBALIGN_VLLM_ASCEND__"


def emit(event: str, **payload: object) -> None:
    print(
        PROTOCOL_PREFIX
        + json.dumps(
            {"event": event, **payload},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        flush=True,
    )


def build_worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Persistent qwen-asr vLLM-Ascend subtitle worker"
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-inference-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--attention-implementation",
        choices=["eager", "sdpa"],
        default="eager",
    )
    return parser


def build_ascend_vllm_engine(
    aligner: Any,
    torch: Any,
    *,
    device: str,
    gpu_memory_utilization: float,
    max_inference_batch_size: int,
    max_new_tokens: int,
    max_model_len: int,
    enforce_eager: bool,
    attention_implementation: str,
) -> Any:
    """Load ASR through vLLM-Ascend and ForcedAligner through torch-npu."""
    engine = object.__new__(aligner.QwenEngine)

    forced_aligner_kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "device_map": {"": device},
        "attn_implementation": attention_implementation,
    }

    logger = logging.getLogger(__name__)
    logger.info("Loading ASR(vLLM-Ascend): %s", aligner.ASR_MODEL_PATH)
    logger.info(
        "Loading ForcedAligner(torch-npu) on %s: %s",
        device,
        aligner.FORCED_ALIGNER_PATH,
    )
    engine.asr = aligner.Qwen3ASRModel.LLM(
        model=str(aligner.ASR_MODEL_PATH),
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
        max_inference_batch_size=max_inference_batch_size,
        max_new_tokens=max_new_tokens,
        forced_aligner=str(aligner.FORCED_ALIGNER_PATH),
        forced_aligner_kwargs=forced_aligner_kwargs,
    )

    if engine.asr.forced_aligner is None:
        raise RuntimeError("Qwen3ASRModel 未加载 ForcedAligner")

    forced_aligner_model = engine.asr.forced_aligner.model
    forced_aligner_model.eval()
    forced_aligner_device = next(forced_aligner_model.parameters()).device
    if forced_aligner_device.type != "npu":
        raise RuntimeError(
            "ForcedAligner 未加载到 NPU，实际设备为 "
            f"{forced_aligner_device}"
        )

    engine.fa_processor = engine.asr.forced_aligner.aligner_processor
    torch.npu.synchronize()
    logger.info(
        "Ascend vLLM models ready | forced_aligner_device=%s | "
        "torch_npu_allocated=%.2f GiB",
        forced_aligner_device,
        torch.npu.memory_allocated(forced_aligner_device) / 1024**3,
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

        # Importing qwen-asr through the shared aligner registers its vLLM ASR
        # implementation. The custom constructor below replaces only the
        # CUDA-specific device selection from QwenEngine.__init__.
        # Do not call torch.npu.set_device(), is_available(), synchronize(),
        # or allocate an NPU tensor before Qwen3ASRModel.LLM. vLLM 0.14 starts
        # an EngineCore subprocess, and inheriting an initialized NPU runtime
        # can make that subprocess fail while creating its default stream.
        import jsonl_forced_align as aligner

        aligner.validate_models()
        engine = build_ascend_vllm_engine(
            aligner,
            torch,
            device=args.device,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_inference_batch_size=args.max_inference_batch_size,
            max_new_tokens=args.max_new_tokens,
            max_model_len=args.max_model_len,
            enforce_eager=args.enforce_eager,
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
                raise ValueError("Unknown NPU vLLM worker command")

            job_id = str(command["job_id"])
            log_path = Path(str(command["log_path"]))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(log_path, encoding="utf-8")
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
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
