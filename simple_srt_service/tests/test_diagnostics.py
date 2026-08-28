import json

from simple_srt_service.diagnostics import (
    parse_alignment_metrics,
    persist_request_diagnostics,
)


def test_parse_alignment_metrics():
    metrics = parse_alignment_metrics(
        "\n".join(
            [
                "Speech chunks: 3 | mean 42.50s | max 61.00s",
                "ASR timestamp tokens: 900",
                "Mapped reference tokens: 850/1000 (85.00%)",
            ]
        )
    )

    assert metrics["speech_chunks"]["count"] == 3
    assert metrics["asr_timestamp_tokens"] == 900
    assert metrics["mapped_reference_tokens"] == {
        "mapped": 850,
        "total": 1000,
        "percent": 85.0,
    }


def test_persist_request_diagnostics_keeps_small_artifacts_only(tmp_path):
    work_dir = tmp_path / "work"
    diagnostics_dir = tmp_path / "diagnostics"
    work_dir.mkdir()
    (work_dir / "media.wav").write_bytes(b"large-media-placeholder")
    (work_dir / "subtitle.srt").write_text("input", encoding="utf-8")
    (work_dir / "alignment.log").write_text(
        "Mapped reference tokens: 8/10 (80.00%)\n",
        encoding="utf-8",
    )
    (work_dir / "aligned.internal.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "index": 1,
                        "text": "hello",
                        "start": 0.0,
                        "end": 1.0,
                        "method": "local_forced_aligner",
                    }
                ),
                json.dumps(
                    {
                        "index": 2,
                        "text": "world",
                        "start": None,
                        "end": None,
                        "method": "unresolved",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (work_dir / "aligned.srt").write_text("result", encoding="utf-8")

    summary_path = persist_request_diagnostics(
        request_id="abc123",
        work_dir=work_dir,
        diagnostics_dir=diagnostics_dir,
        media_name="demo.wav",
        subtitle_name="demo.srt",
        language="English",
        status="completed",
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["total_lines"] == 2
    assert summary["aligned_lines"] == 1
    assert summary["unresolved_indices"] == [2]
    assert summary["local_forced_aligner_lines"] == 1
    assert summary["alignment_metrics"]["mapped_reference_tokens"]["percent"] == 80.0
    assert (diagnostics_dir / "abc123.alignment.log").is_file()
    assert (diagnostics_dir / "abc123.aligned.jsonl").is_file()
    assert (diagnostics_dir / "abc123.aligned.srt").is_file()
    assert not (diagnostics_dir / "media.wav").exists()
    assert not (diagnostics_dir / "subtitle.srt").exists()
