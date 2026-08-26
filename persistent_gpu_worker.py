from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path


PROTOCOL_PREFIX = "__SUBALIGN__"


def emit(event: str, **payload: object) -> None:
    message = {"event": event, **payload}
    print(
        PROTOCOL_PREFIX + json.dumps(message, ensure_ascii=False, separators=(",", ":")),
        flush=True,
    )


def build_worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent subtitle GPU worker")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--max-inference-batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--flash-attn", action="store_true")
    return parser


def main() -> None:
    worker_args = build_worker_parser().parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Import only inside the executable worker process. This keeps CUDA/vLLM
    # initialization isolated from the FastAPI process and satisfies vLLM's
    # multiprocessing __main__ requirement.
    import jsonl_forced_align as aligner

    try:
        aligner.validate_models()
        engine = aligner.QwenEngine(
            gpu_memory_utilization=worker_args.gpu_memory_utilization,
            max_inference_batch_size=worker_args.max_inference_batch_size,
            max_new_tokens=worker_args.max_new_tokens,
            use_flash_attention=worker_args.flash_attn,
        )
    except Exception as exc:
        emit("startup_error", error=str(exc), traceback=traceback.format_exc())
        raise

    emit("ready")

    for raw in sys.stdin:
        command: dict[str, object] = {}
        raw = raw.strip()
        if not raw:
            continue
        try:
            command = json.loads(raw)
            if command.get("command") == "shutdown":
                emit("stopped")
                return
            if command.get("command") != "align":
                raise ValueError("Unknown GPU worker command")

            job_id = str(command["job_id"])
            log_path = Path(command["log_path"])
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
