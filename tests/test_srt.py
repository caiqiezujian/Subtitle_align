import json

from app.srt import format_srt_time, rows_to_srt, write_outputs


def test_format_srt_time():
    assert format_srt_time(3661.007) == "01:01:01,007"
    assert format_srt_time(-1) == "00:00:00,000"


def test_rows_to_srt_skips_unresolved_and_renumbers():
    result = rows_to_srt(
        [
            {"text": "first", "start": 0.2, "end": 1.4},
            {"text": "missing", "start": None, "end": None},
            {"text": "third", "start": 2, "end": 3},
        ]
    )
    assert "1\n00:00:00,200 --> 00:00:01,400\nfirst" in result
    assert "2\n00:00:02,000 --> 00:00:03,000\nthird" in result
    assert "missing" not in result


def test_write_outputs_generates_jsonl_and_bom_srt(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        json.dumps(
            {
                "text": "你好",
                "source": {"src": "你好"},
                "time": [1.0, 2.25],
                "alignment_method": "local_forced_aligner",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    jsonl = tmp_path / "aligned.jsonl"
    srt = tmp_path / "aligned.srt"
    total, resolved = write_outputs(raw, jsonl, srt)
    assert (total, resolved) == (1, 1)
    item = json.loads(jsonl.read_text(encoding="utf-8"))
    assert item["duration"] == 1.25
    assert item["method"] == "local_forced_aligner"
    assert srt.read_bytes().startswith(b"\xef\xbb\xbf")
