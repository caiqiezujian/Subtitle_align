from simple_srt_service.settings import build_settings


def test_build_settings_supports_auto_language_and_relative_data_dir():
    settings = build_settings(
        {
            "server": {"port": 13000},
            "gpu": {"visible_devices": "0"},
            "alignment_engine": {"language": "auto"},
            "models": {"root": "/models"},
            "storage": {"data_dir": "./test-data"},
        }
    )
    assert settings.server_port == 13000
    assert settings.cuda_visible_devices == "0"
    assert settings.source_language == "auto"
    assert settings.data_dir.name == "test-data"
