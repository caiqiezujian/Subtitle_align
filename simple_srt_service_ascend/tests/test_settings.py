from simple_srt_service_ascend.settings import build_settings


def test_build_settings_for_ascend_transformers():
    settings = build_settings(
        {
            "server": {"port": 13000},
            "npu": {"visible_devices": "7", "logical_device_index": 0},
            "alignment_engine": {
                "language": "auto",
                "max_inference_batch_size": 2,
                "asr_batch_size": 3,
                "forced_aligner_batch_size": 1,
                "attention_implementation": "eager",
            },
            "models": {"root": "/models"},
            "storage": {"data_dir": "./test-data"},
        }
    )

    assert settings.server_port == 13000
    assert settings.npu_visible_devices == "7"
    assert settings.npu_device == "npu:0"
    assert settings.engine_max_inference_batch_size == 2
    assert settings.engine_asr_batch_size == 3
    assert settings.engine_forced_aligner_batch_size == 1
    assert settings.source_language == "auto"
    assert settings.data_dir.name == "test-data"

