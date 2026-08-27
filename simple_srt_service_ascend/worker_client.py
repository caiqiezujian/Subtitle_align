from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from simple_srt_service_ascend.settings import AscendSettings


PROTOCOL_PREFIX = "__SUBALIGN_ASCEND__"


class PersistentAscendWorker:
    def __init__(self, settings: AscendSettings, project_root: Path) -> None:
        self.settings = settings
        self.project_root = project_root
        self.process: subprocess.Popen[str] | None = None
        self.responses: queue.Queue[dict[str, Any]] = queue.Queue()
        self.reader: threading.Thread | None = None
        self.status = "stopped"
        self.error: str | None = None
        self._command_lock = threading.Lock()
        self._stderr_handle = None

    @property
    def log_path(self) -> Path:
        return self.settings.data_dir / "npu-worker.log"

    @property
    def current_status(self) -> str:
        if (
            self.status == "ready"
            and self.process is not None
            and self.process.poll() is not None
        ):
            self.status = "error"
            self.error = f"NPU Worker exited with code {self.process.returncode}"
        return self.status

    def _tail_log(self, length: int = 6000) -> str:
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")[-length:]
        except OSError:
            return ""

    def start(self) -> None:
        if self.process and self.process.poll() is None:
            return

        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.status = "loading"
        self.error = None
        while not self.responses.empty():
            try:
                self.responses.get_nowait()
            except queue.Empty:
                break

        command = [
            sys.executable,
            "-u",
            "-m",
            "simple_srt_service_ascend.worker",
            "--device",
            self.settings.npu_device,
            "--max-inference-batch-size",
            str(self.settings.engine_max_inference_batch_size),
            "--max-new-tokens",
            str(self.settings.engine_max_new_tokens),
            "--attention-implementation",
            self.settings.engine_attention_implementation,
        ]

        env = os.environ.copy()
        env["ASCEND_RT_VISIBLE_DEVICES"] = self.settings.npu_visible_devices
        env["QWEN_MODEL_ROOT"] = str(self.settings.model_root)
        self._stderr_handle = self.log_path.open("a", encoding="utf-8", buffering=1)
        self.process = subprocess.Popen(
            command,
            cwd=self.project_root,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_handle,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self.reader = threading.Thread(target=self._read_stdout, daemon=True)
        self.reader.start()

        try:
            message = self.responses.get(
                timeout=self.settings.engine_startup_timeout_seconds
            )
        except queue.Empty as exc:
            self.status = "error"
            self.error = "NPU Worker 模型加载超时"
            self.stop(force=True)
            raise RuntimeError(f"{self.error}\n{self._tail_log()}") from exc

        if message.get("event") != "ready":
            self.status = "error"
            self.error = str(message.get("error") or "NPU Worker 启动失败")
            self.stop(force=True)
            raise RuntimeError(f"{self.error}\n{self._tail_log()}")
        self.status = "ready"

    def _read_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for raw in self.process.stdout:
            line = raw.rstrip("\r\n")
            if line.startswith(PROTOCOL_PREFIX):
                try:
                    self.responses.put(json.loads(line[len(PROTOCOL_PREFIX) :]))
                    continue
                except json.JSONDecodeError:
                    pass
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        self.responses.put(
            {
                "event": "worker_exited",
                "error": f"NPU Worker exited with code {self.process.poll()}",
            }
        )

    def run_job(
        self,
        job_id: str,
        argv: list[str],
        log_path: Path,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> None:
        with self._command_lock:
            if self.current_status != "ready" or self.process is None:
                raise RuntimeError(f"NPU Worker 不可用：{self.error or self.current_status}")
            assert self.process.stdin is not None
            self.process.stdin.write(
                json.dumps(
                    {
                        "command": "align",
                        "job_id": job_id,
                        "argv": argv,
                        "log_path": str(log_path),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            self.process.stdin.flush()

            while True:
                message = self.responses.get()
                event = message.get("event")
                if event == "progress" and message.get("job_id") == job_id:
                    if progress_callback:
                        progress_callback(
                            int(message.get("progress", 0)),
                            str(message.get("stage", "NPU 正在对齐")),
                        )
                    continue
                if event == "completed" and message.get("job_id") == job_id:
                    return
                if event in {"job_error", "worker_exited"}:
                    error = str(message.get("error") or "NPU Worker 任务失败")
                    trace = str(message.get("traceback") or "")
                    if event == "worker_exited":
                        self.status = "error"
                        self.error = error
                    raise RuntimeError(f"{error}\n{trace}\n{self._tail_log()}")

    def stop(self, force: bool = False) -> None:
        process = self.process
        if process is None:
            return
        if process.poll() is None and not force:
            try:
                assert process.stdin is not None
                process.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
                process.stdin.flush()
                process.wait(timeout=30)
            except Exception:
                force = True
        if process.poll() is None and force:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if self._stderr_handle:
            self._stderr_handle.close()
            self._stderr_handle = None
        self.status = "stopped" if not self.error else "error"

