from simple_srt_service.settings import build_settings


def test_build_settings_supports_ascend_vllm_options():
    settings = build_settings(
        {
            "server": {"port": 13000},
            "npu": {"visible_devices": "6", "logical_device_index": 0},
            "alignment_engine": {
                "language": "auto",
                "gpu_memory_utilization": 0.85,
                "max_inference_batch_size": 2,
                "asr_batch_size": 3,
                "forced_aligner_batch_size": 1,
                "max_model_len": 8192,
                "enforce_eager": True,
                "attention_implementation": "eager",
            },
            "models": {"root": "/models"},
            "storage": {"data_dir": "./test-data"},
        }
    )
    assert settings.server_port == 13000
    assert settings.npu_visible_devices == "6"
    assert settings.npu_device == "npu:0"
    assert settings.source_language == "auto"
    assert settings.engine_gpu_memory_utilization == 0.85
    assert settings.engine_max_inference_batch_size == 2
    assert settings.engine_asr_batch_size == 3
    assert settings.engine_forced_aligner_batch_size == 1
    assert settings.engine_max_model_len == 8192
    assert settings.engine_enforce_eager is True
    assert settings.data_dir.name == "test-data"


def test_build_settings_accepts_legacy_gpu_visible_devices():
    settings = build_settings({"gpu": {"visible_devices": "7"}})
    assert settings.npu_visible_devices == "7"
    assert settings.npu_device == "npu:0"
