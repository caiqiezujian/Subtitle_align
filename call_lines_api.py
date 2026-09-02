from __future__ import annotations

import json
import mimetypes
import os
import time
from pathlib import Path

import httpx


# ======================== 只需要修改这里 ========================
SERVER_URL = "http://10.170.23.103:12045"
MEDIA_PATH = Path("/data/yb/Test/demo.wav")
OUTPUT_SRT = Path("/data/yb/Test/demo.aligned.srt")
OUTPUT_JSONL = Path("/data/yb/Test/demo.aligned.jsonl")
API_KEY = os.getenv("SUBALIGN_API_KEY", "")

LINES = [
    "我们张常宁",
    "刚才的速度是91，",
    "这已经是非常顶级的女排运动员的速度了。",
    "我们一般人可能就是40或者是50的样子。",
    "把杆击倒的这人呢，",
    "杆击倒的属于失误，",
    "他不可能拍到77的。",
    "哎呀。",
    "这把我们自尊给撕了。",
]
# ===============================================================

POLL_INTERVAL_SECONDS = 2
MAX_WAIT_SECONDS = 2 * 60 * 60


def error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or "服务未返回错误详情"
    if isinstance(payload, dict) and payload.get("detail"):
        return str(payload["detail"])
    return json.dumps(payload, ensure_ascii=False)


def require_success(response: httpx.Response, action: str) -> None:
    if response.is_success:
        return
    raise SystemExit(
        f"{action}失败（HTTP {response.status_code}）：{error_detail(response)}"
    )


def save_download(response: httpx.Response, destination: Path) -> None:
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    temporary.write_bytes(response.content)
    temporary.replace(destination)
    print(f"已保存：{destination}")


def main() -> None:
    media_path = MEDIA_PATH.expanduser().resolve()
    if not media_path.is_file():
        raise SystemExit(f"找不到音频或视频：{media_path}")
    if not LINES or any(not isinstance(line, str) or not line.strip() for line in LINES):
        raise SystemExit("LINES 必须是非空字符串数组")

    base_url = SERVER_URL.rstrip("/")
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    media_type = mimetypes.guess_type(media_path.name)[0] or "application/octet-stream"
    timeout = httpx.Timeout(MAX_WAIT_SECONDS, connect=30.0)

    with httpx.Client(timeout=timeout, trust_env=False, headers=headers) as client:
        health = client.get(f"{base_url}/api/health")
        require_success(health, "健康检查")
        health_payload = health.json()
        if health_payload.get("status") != "ok":
            raise SystemExit(
                "服务尚未就绪：" + json.dumps(health_payload, ensure_ascii=False)
            )

        print(f"正在提交 {len(LINES)} 行文本和 {media_path.name}")
        with media_path.open("rb") as media_file:
            response = client.post(
                f"{base_url}/api/line-jobs",
                files={"media": (media_path.name, media_file, media_type)},
                data={"lines": json.dumps(LINES, ensure_ascii=False)},
            )
        require_success(response, "任务提交")
        job = response.json()
        job_id = str(job["id"])
        print(f"任务已提交：{job_id}")

        deadline = time.monotonic() + MAX_WAIT_SECONDS
        last_progress: tuple[object, object, object] | None = None
        while True:
            if time.monotonic() >= deadline:
                raise SystemExit(f"等待任务 {job_id} 超过 {MAX_WAIT_SECONDS} 秒")
            status_response = client.get(f"{base_url}/api/jobs/{job_id}")
            require_success(status_response, "查询任务")
            job = status_response.json()
            current = (job.get("status"), job.get("progress"), job.get("stage"))
            if current != last_progress:
                print(f"{current[1]}% | {current[0]} | {current[2]}")
                last_progress = current
            if job.get("status") == "completed":
                break
            if job.get("status") == "failed":
                raise SystemExit(f"字幕对齐失败：{job.get('error') or '未知错误'}")
            time.sleep(POLL_INTERVAL_SECONDS)

        srt_response = client.get(f"{base_url}/api/jobs/{job_id}/download/srt")
        require_success(srt_response, "下载 SRT")
        save_download(srt_response, OUTPUT_SRT)

        jsonl_response = client.get(f"{base_url}/api/jobs/{job_id}/download/jsonl")
        require_success(jsonl_response, "下载 JSONL")
        save_download(jsonl_response, OUTPUT_JSONL)


if __name__ == "__main__":
    main()
