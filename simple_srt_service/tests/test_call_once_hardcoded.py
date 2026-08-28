from pathlib import Path

import httpx
import pytest

from simple_srt_service import call_once_hardcoded as caller


def test_validate_paths_accepts_distinct_media_srt_and_output(tmp_path, monkeypatch):
    media = tmp_path / "demo.wav"
    srt = tmp_path / "demo.srt"
    output = tmp_path / "demo.aligned.srt"
    media.write_bytes(b"RIFF")
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")

    monkeypatch.setattr(caller, "MEDIA_PATH", media)
    monkeypatch.setattr(caller, "SRT_PATH", srt)
    monkeypatch.setattr(caller, "OUTPUT_PATH", output)

    assert caller.validate_paths() == (
        media.resolve(),
        srt.resolve(),
        output.resolve(),
    )


def test_check_health_requires_resident_ready_worker():
    class ReadyClient:
        def get(self, _url):
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "models_resident": True,
                    "worker": "ready",
                    "inference_backend": "qwen-asr-vllm",
                    "accelerator": "ascend-npu",
                    "npu_device": "npu:0",
                },
            )

    caller.check_health(ReadyClient())

    class LoadingClient:
        def get(self, _url):
            return httpx.Response(
                200,
                json={
                    "status": "degraded",
                    "models_resident": False,
                    "worker": "loading",
                },
            )

    with pytest.raises(SystemExit, match="服务尚未就绪"):
        caller.check_health(LoadingClient())


def test_read_error_prefers_fastapi_detail():
    response = httpx.Response(400, json={"detail": "SRT 解析失败"})
    assert caller.read_error(response) == "SRT 解析失败"
