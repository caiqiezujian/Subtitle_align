from pathlib import Path

import pytest

from app.config import load_config_file


def test_load_yaml_config(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "server:\n  port: 12045\ngpu:\n  visible_devices: '5'\n",
        encoding="utf-8",
    )
    value = load_config_file(path)
    assert value["server"]["port"] == 12045
    assert value["gpu"]["visible_devices"] == "5"


def test_yaml_config_top_level_must_be_mapping(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("- invalid\n- config\n", encoding="utf-8")
    with pytest.raises(ValueError, match="顶层必须是 YAML 对象"):
        load_config_file(path)
