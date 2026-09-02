from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_background_start_waits_for_real_health() -> None:
    script = (ROOT / "start_background.sh").read_text(encoding="utf-8")

    assert "/api/health" in script
    assert 'payload.get("status") != "ok"' in script
    assert 'payload.get("gpu_worker") != "ready"' in script
    assert 'payload.get("models_resident") is not True' in script
    assert "SUBALIGN_STARTUP_WAIT_SECONDS" in script
    assert "sleep 2" in script


def test_foreground_start_uses_unbuffered_logs() -> None:
    script = (ROOT / "start.sh").read_text(encoding="utf-8")

    assert "PYTHONUNBUFFERED=1" in script
