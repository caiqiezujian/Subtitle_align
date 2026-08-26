import json

import pytest

from app.transcript import parse_transcript


def test_plain_text_ignores_blank_lines_and_preserves_physical_line():
    parsed = parse_transcript("第一句\n\n  第二句  \n".encode(), "demo.txt")
    assert parsed.detected_format == "txt"
    assert [line.text for line in parsed.lines] == ["第一句", "第二句"]
    assert parsed.lines[1].source["physical_line"] == 3


def test_jsonl_auto_detects_src_and_preserves_source():
    data = "\n".join(
        json.dumps(row, ensure_ascii=False)
        for row in [{"src": "你好", "speaker": "A"}, {"src": "世界", "speaker": "B"}]
    )
    parsed = parse_transcript(data.encode(), "demo.jsonl")
    assert parsed.text_field == "src"
    assert parsed.lines[0].source["speaker"] == "A"


def test_json_array_inside_data_key():
    payload = {"data": [{"text": "one"}, {"text": "two"}]}
    parsed = parse_transcript(json.dumps(payload).encode(), "unknown.dat")
    assert [line.text for line in parsed.lines] == ["one", "two"]


def test_srt_joins_multiline_and_keeps_input_timing():
    srt = "1\n00:00:01,200 --> 00:00:03,450\nHello\nworld\n"
    parsed = parse_transcript(srt.encode(), "demo.srt")
    assert parsed.lines[0].text == "Hello world"
    assert parsed.lines[0].source == {"input_start": 1.2, "input_end": 3.45}


def test_csv_auto_detects_text_column():
    parsed = parse_transcript("id,text\n1,hello\n2,world\n".encode(), "demo.csv")
    assert parsed.text_field == "text"
    assert len(parsed.lines) == 2


def test_requested_missing_field_is_actionable():
    with pytest.raises(ValueError, match="找不到指定文本字段"):
        parse_transcript(b'{"text":"hello"}\n', "demo.jsonl", "src")
