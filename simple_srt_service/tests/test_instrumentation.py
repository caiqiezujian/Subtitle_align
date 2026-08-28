import json
from pathlib import Path
from types import SimpleNamespace

from simple_srt_service.instrumentation import DetailedAlignmentTrace


class FakeEngine:
    def __init__(self):
        self.asr = SimpleNamespace(forced_aligner=SimpleNamespace())

    def tokenize_text(self, text, language):
        return text.casefold().split()

    def transcribe_chunks(self, **_kwargs):
        return [
            SimpleNamespace(
                text="Hello",
                norm="hello",
                start=0.1,
                end=0.5,
                chunk_index=0,
            ),
            SimpleNamespace(
                text="world",
                norm="world",
                start=0.6,
                end=1.0,
                chunk_index=0,
            ),
        ]

    def refine_lines(self, *, coarse_times, **_kwargs):
        return [
            SimpleNamespace(
                start=item.start + 0.01,
                end=item.end + 0.01,
                matched_hyp_start=item.matched_hyp_start,
                matched_hyp_end=item.matched_hyp_end,
                method="local_forced_aligner",
            )
            for item in coarse_times
        ]


def test_detailed_trace_captures_each_alignment_boundary(tmp_path):
    asr_model = tmp_path / "Qwen3-ASR-1.7B"
    fa_model = tmp_path / "Qwen3-ForcedAligner-0.6B"
    asr_model.mkdir()
    fa_model.mkdir()
    (asr_model / "config.json").write_text('{"model":"asr"}', encoding="utf-8")
    (fa_model / "config.json").write_text('{"model":"fa"}', encoding="utf-8")

    row = SimpleNamespace(
        index=0,
        physical_line_number=1,
        text="Hello world",
        token_start=0,
        token_end=2,
    )
    coarse = SimpleNamespace(
        start=0.1,
        end=1.0,
        matched_hyp_start=0,
        matched_hyp_end=1,
        method="asr_token_alignment",
    )
    final = SimpleNamespace(
        start=0.11,
        end=1.01,
        matched_hyp_start=0,
        matched_hyp_end=1,
        method="local_forced_aligner",
    )

    aligner = SimpleNamespace(
        ASR_MODEL_PATH=asr_model,
        FORCED_ALIGNER_PATH=fa_model,
        tokenize_jsonl_rows=lambda engine, rows, language: ["hello", "world"],
        build_ref_to_hyp_mapping=lambda ref_tokens, hyp_tokens: [0, 1],
        coarse_line_times=lambda rows, mapping, hyp_tokens: [coarse],
        write_aligned_jsonl=lambda rows, line_times, **kwargs: None,
    )
    trace = DetailedAlignmentTrace(
        work_dir=tmp_path,
        job_id="request-1",
        engine_options={"backend": "test"},
    )

    chunk = SimpleNamespace(
        index=0,
        start_sample=0,
        end_sample=16000,
        start_sec=0.0,
        end_sec=1.0,
        duration_sec=1.0,
    )
    engine = FakeEngine()
    with trace.instrument(aligner, engine) as traced:
        assert aligner.tokenize_jsonl_rows(
            engine=traced, rows=[row], language="English"
        ) == ["hello", "world"]
        traced.transcribe_chunks(chunks=[chunk])
        mapping = aligner.build_ref_to_hyp_mapping(
            ref_tokens=["hello", "world"],
            hyp_tokens=["hello", "world"],
        )
        aligner.coarse_line_times(
            rows=[row],
            mapping=mapping,
            hyp_tokens=[],
        )
        refined = traced.refine_lines(rows=[row], coarse_times=[coarse])
        aligner.write_aligned_jsonl(rows=[row], line_times=[final])
        assert refined[0].method == "local_forced_aligner"
    trace.finish(status="completed")

    summary = json.loads((tmp_path / "trace.summary.json").read_text(encoding="utf-8"))
    assert summary["counts"]["asr_timestamp_tokens"] == 2
    assert summary["counts"]["mapped_reference_tokens"] == 2
    assert summary["counts"]["mapping_percent"] == 100.0
    assert summary["final_method_counts"] == {"local_forced_aligner": 1}

    timeline = [
        json.loads(line)
        for line in (tmp_path / "trace.asr-timeline.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert timeline[0]["text"] == "Hello"
    assert timeline[1]["start"] == 0.6

    stages = [
        json.loads(line)
        for line in (tmp_path / "trace.line-stages.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert stages[0]["coarse"]["method"] == "asr_token_alignment"
    assert stages[0]["refined"]["method"] == "local_forced_aligner"
    assert stages[0]["final"]["start"] == 0.11

    runtime = json.loads((tmp_path / "trace.runtime.json").read_text(encoding="utf-8"))
    assert runtime["models"]["asr"]["file_count"] == 1
    assert runtime["models"]["asr"]["files"][0]["sha256"]
