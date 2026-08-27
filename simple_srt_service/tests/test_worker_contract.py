from pathlib import Path
from types import SimpleNamespace

from simple_srt_service.check_environment import version_family
from simple_srt_service.worker import build_ascend_vllm_engine


SERVICE_DIR = Path(__file__).resolve().parents[1]


def test_worker_uses_vllm_for_asr_and_npu_for_forced_aligner():
    source = (SERVICE_DIR / "worker.py").read_text(encoding="utf-8")

    assert "Qwen3ASRModel.LLM" in source
    assert '"device_map": {"": device}' in source
    assert "torch.npu.set_device(args.device)" not in source
    assert "torch.npu.is_available()" not in source
    assert '"cuda:0"' not in source


def test_worker_client_sets_only_ascend_device_visibility():
    source = (SERVICE_DIR / "worker_client.py").read_text(encoding="utf-8")

    assert 'env["ASCEND_RT_VISIBLE_DEVICES"]' in source
    assert 'env.pop("CUDA_VISIBLE_DEVICES", None)' in source
    assert 'env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"' in source


def test_build_engine_routes_asr_to_vllm_and_aligner_to_npu():
    captured = {}

    class FakeModel:
        def eval(self):
            captured["eval"] = True

        def parameters(self):
            yield SimpleNamespace(device=SimpleNamespace(type="npu"))

    forced_aligner = SimpleNamespace(
        model=FakeModel(),
        aligner_processor=object(),
    )

    class FakeQwen3ASRModel:
        @classmethod
        def LLM(cls, **kwargs):
            captured["kwargs"] = kwargs
            return SimpleNamespace(forced_aligner=forced_aligner)

    class FakeNpu:
        @staticmethod
        def synchronize():
            captured["synchronized"] = True

        @staticmethod
        def memory_allocated(_device):
            return 1024**3

    aligner = SimpleNamespace(
        QwenEngine=type("QwenEngine", (), {}),
        Qwen3ASRModel=FakeQwen3ASRModel,
        ASR_MODEL_PATH=Path("/models/Qwen3-ASR-1.7B"),
        FORCED_ALIGNER_PATH=Path("/models/Qwen3-ForcedAligner-0.6B"),
    )
    torch = SimpleNamespace(bfloat16="bf16", npu=FakeNpu())

    build_ascend_vllm_engine(
        aligner,
        torch,
        device="npu:0",
        gpu_memory_utilization=0.85,
        max_inference_batch_size=4,
        max_new_tokens=1024,
        max_model_len=4096,
        enforce_eager=False,
        attention_implementation="eager",
    )

    kwargs = captured["kwargs"]
    assert kwargs["gpu_memory_utilization"] == 0.85
    assert kwargs["forced_aligner_kwargs"]["device_map"] == {"": "npu:0"}
    assert kwargs["enforce_eager"] is False
    assert captured["eval"] is True
    assert captured["synchronized"] is True


def test_version_family_accepts_release_candidates_and_empty_wheels():
    assert version_family("0.14.0") == (0, 14)
    assert version_family("0.14.0rc1") == (0, 14)
    assert version_family("0.14.0+empty") == (0, 14)
