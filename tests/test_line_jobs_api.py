import asyncio
import json
from io import BytesIO

from fastapi import UploadFile

from app import main


def test_line_job_uses_resident_queue_and_auto_language(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    media_path = upload_dir / "media.wav"
    transcript_path = upload_dir / "transcript.jsonl"
    captured = {}

    class FakeState:
        id = "job-123"

        @staticmethod
        def public_dict():
            return {"id": "job-123", "status": "queued"}

    def fake_create(media_name, transcript_name, options):
        captured["media_name"] = media_name
        captured["transcript_name"] = transcript_name
        captured["options"] = options
        return FakeState(), media_path, transcript_path

    async def fake_enqueue(job_id, options):
        captured["enqueued"] = (job_id, options)

    monkeypatch.setattr(main.manager, "create", fake_create)
    monkeypatch.setattr(main.manager, "enqueue", fake_enqueue)
    monkeypatch.setattr(main.manager, "_update", lambda *args, **kwargs: None)

    media = UploadFile(filename="demo.wav", file=BytesIO(b"RIFF-demo"))
    lines = ["我们张常宁", "刚才的速度是91，", "哎呀。"]
    response = asyncio.run(
        main.create_line_job(media=media, lines=json.dumps(lines, ensure_ascii=False))
    )

    assert response == {"id": "job-123", "status": "queued"}
    assert captured["media_name"] == "demo.wav"
    assert captured["transcript_name"] == "lines.jsonl"
    assert captured["options"].language == "Chinese"
    assert captured["options"].text_field == "text"
    assert captured["options"].use_flash is False
    assert captured["options"].local_refine is True
    assert captured["enqueued"][0] == "job-123"
    assert media_path.read_bytes() == b"RIFF-demo"
    rows = [
        json.loads(line)
        for line in transcript_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["text"] for row in rows] == lines


def test_line_job_route_is_registered():
    route = next(route for route in main.app.routes if route.path == "/api/line-jobs")
    assert route.methods == {"POST"}
