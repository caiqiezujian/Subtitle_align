from pathlib import Path

from app.config import Settings
from app.gpu_worker_client import PersistentGpuWorker


def test_gpu_worker_stays_alive_and_serves_multiple_jobs(tmp_path: Path):
    fixture_root = Path(__file__).parent / "fixtures"
    settings = Settings(
        data_dir=tmp_path,
        engine_startup_timeout_seconds=10,
        engine_flash_attention=False,
    )
    worker = PersistentGpuWorker(settings, fixture_root)
    worker.start()
    try:
        assert worker.current_status == "ready"
        assert worker.process is not None
        first_pid = worker.process.pid
        worker.run_job("job-1", [], tmp_path / "job-1.log")
        worker.run_job("job-2", [], tmp_path / "job-2.log")
        assert worker.process.pid == first_pid
        assert worker.process.poll() is None
    finally:
        worker.stop()
    assert worker.current_status == "stopped"
