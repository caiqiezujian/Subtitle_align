import asyncio
from inspect import signature
from io import BytesIO

from fastapi import UploadFile

from simple_srt_service_ascend import main
from simple_srt_service_ascend.main import align, app


def test_align_endpoint_has_exactly_two_inputs():
    assert list(signature(align).parameters) == ["media", "srt"]
    route = next(route for route in app.routes if getattr(route, "path", None) == "/align")
    assert route.methods == {"POST"}


def test_align_returns_only_srt_and_cleans_workspace(tmp_path, monkeypatch):
    expected = b"\xef\xbb\xbf1\n00:00:00,000 --> 00:00:01,000\nhello\n"
    monkeypatch.setattr(main, "request_root", tmp_path)
    monkeypatch.setattr(main, "run_alignment", lambda *args: expected)
    media = UploadFile(filename="demo.wav", file=BytesIO(b"RIFF-demo"))
    srt = UploadFile(
        filename="demo.srt",
        file=BytesIO(b"1\n00:00:00,000 --> 00:00:01,000\nhello\n"),
    )

    response = asyncio.run(main.align(media=media, srt=srt))

    assert response.body == expected
    assert response.media_type == "application/x-subrip; charset=utf-8"
    assert "demo.aligned.srt" in response.headers["content-disposition"]
    assert list(tmp_path.iterdir()) == []

