import json

import pytest

from app.line_input import detect_spoken_language, parse_lines_json, write_lines_jsonl


def test_parse_lines_preserves_order_and_punctuation():
    lines = parse_lines_json('["第一句","，","third"]')
    assert lines == ["第一句", "，", "third"]


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ('{"text":"not-array"}', "JSON 字符串数组"),
        ("[]", "不能为空数组"),
        ('["ok",2]', "不是字符串"),
        ('["ok","  "]', "空字符串"),
    ],
)
def test_parse_lines_rejects_ambiguous_inputs(value, message):
    with pytest.raises(ValueError, match=message):
        parse_lines_json(value)


def test_detect_spoken_language():
    assert detect_spoken_language(["hello world"]) == "English"
    assert detect_spoken_language(["女排运动员"]) == "Chinese"
    assert detect_spoken_language(["今日は試合です"]) == "Japanese"


def test_write_lines_jsonl_keeps_source_indices(tmp_path):
    destination = tmp_path / "lines.jsonl"
    write_lines_jsonl(["第一句", "第二句"], destination)
    rows = [
        json.loads(line)
        for line in destination.read_text(encoding="utf-8").splitlines()
    ]
    assert rows == [
        {"text": "第一句", "source_index": 1},
        {"text": "第二句", "source_index": 2},
    ]
