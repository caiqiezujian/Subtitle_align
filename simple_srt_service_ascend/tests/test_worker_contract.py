from pathlib import Path


def test_ascend_worker_uses_transformers_backend_only():
    source = (
        Path(__file__).resolve().parents[1] / "worker.py"
    ).read_text(encoding="utf-8")

    assert "Qwen3ASRModel.from_pretrained" in source
    assert "Qwen3ASRModel.LLM" not in source
    assert "torch.npu.is_available" in source
    assert '"device_map": {"": device}' in source

