import json

import httpx

from app.config import Settings
from app.llm_adapter import FlashAssistant, extract_json_object


def test_extract_json_object_accepts_common_internal_model_wrappers():
    expected = {"lines": ["第一句"]}
    assert extract_json_object(json.dumps(expected, ensure_ascii=False)) == expected
    assert extract_json_object('```json\n{"lines":["第一句"]}\n```') == expected
    assert extract_json_object('结果如下：\n{"lines":["第一句"]}\n完毕') == expected


def test_v4_flash_uses_internal_contract(monkeypatch, tmp_path):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '```json\n{"lines":["你好"]}\n```'}}
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    original_client = httpx.Client

    def fake_client(*args, **kwargs):
        captured["trust_env"] = kwargs.get("trust_env")
        captured["verify"] = kwargs.get("verify")
        kwargs["transport"] = transport
        return original_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", fake_client)
    config = Settings(
        data_dir=tmp_path,
        flash_enabled=True,
        flash_base_url="https://internal.example/v1",
        flash_api_key="secret-key",
        flash_model="dsv4",
        flash_retry_count=0,
        flash_verify_ssl=False,
    )
    result = FlashAssistant(config)._normalize_batch(["你好"])

    assert result == ["你好"]
    assert captured["url"] == "https://internal.example/v1/chat/completions"
    assert captured["authorization"] == "Bearer secret-key"
    assert captured["body"]["model"] == "dsv4"
    assert captured["body"]["max_tokens"] == 8192
    assert "response_format" not in captured["body"]
    assert captured["trust_env"] is False
    assert captured["verify"] is False
