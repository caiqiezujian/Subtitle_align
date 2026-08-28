from __future__ import annotations

import json
import mimetypes
from pathlib import Path

import httpx


# ======================== 只需要修改这里 ========================
SERVER_URL = "http://10.170.23.103:12046"
MEDIA_PATH = Path("/data/yb/Test/en_external_12.wav")
SRT_PATH = Path("/data/yb/Test/english.srt")
OUTPUT_PATH = Path("/data/yb/Test/english.aligned.srt")

# 长音视频可能需要较长时间；当前设置为 2 小时。
TIMEOUT_SECONDS = 2 * 60 * 60
# ===============================================================


def read_error(response: httpx.Response) -> str:
    """尽量从服务响应中提取清晰的错误信息。"""
    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError):
        return response.text.strip() or "服务没有返回错误详情"
    if isinstance(payload, dict) and payload.get("detail"):
        return str(payload["detail"])
    return json.dumps(payload, ensure_ascii=False)


def validate_paths() -> tuple[Path, Path, Path]:
    media_path = MEDIA_PATH.expanduser().resolve()
    srt_path = SRT_PATH.expanduser().resolve()
    output_path = OUTPUT_PATH.expanduser().resolve()

    if not media_path.is_file():
        raise SystemExit(f"找不到音视频文件：{media_path}")
    if not srt_path.is_file():
        raise SystemExit(f"找不到 SRT 文件：{srt_path}")
    if srt_path.suffix.casefold() != ".srt":
        raise SystemExit(f"字幕文件必须以 .srt 结尾：{srt_path}")
    if output_path == srt_path:
        raise SystemExit("输出路径不能与输入 SRT 路径相同")

    return media_path, srt_path, output_path


def check_health(client: httpx.Client) -> None:
    health_url = f"{SERVER_URL.rstrip('/')}/health"
    try:
        response = client.get(health_url)
    except httpx.HTTPError as exc:
        raise SystemExit(f"无法连接字幕服务健康检查：{exc}") from exc

    if response.status_code != 200:
        raise SystemExit(
            f"健康检查失败（HTTP {response.status_code}）：{read_error(response)}"
        )
    try:
        status = response.json()
    except (json.JSONDecodeError, ValueError) as exc:
        raise SystemExit("健康检查没有返回有效 JSON") from exc

    if (
        status.get("status") != "ok"
        or status.get("models_resident") is not True
        or status.get("worker") != "ready"
    ):
        raise SystemExit("服务尚未就绪：" + json.dumps(status, ensure_ascii=False))

    print(
        "服务已就绪："
        f"backend={status.get('inference_backend')}，"
        f"accelerator={status.get('accelerator')}，"
        f"device={status.get('npu_device')}"
    )


def main() -> None:
    media_path, srt_path, output_path = validate_paths()
    temporary_output = output_path.with_name(output_path.name + ".part")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output.unlink(missing_ok=True)

    align_url = f"{SERVER_URL.rstrip('/')}/align"
    media_type = mimetypes.guess_type(media_path.name)[0]
    media_type = media_type or "application/octet-stream"

    print("准备发起一次字幕对齐请求")
    print(f"服务地址：{align_url}")
    print(f"音视频：{media_path}")
    print(f"原字幕：{srt_path}")
    print(f"输出文件：{output_path}")
    print("请求发出后会同步等待服务完成，不会重复提交。")

    request_timeout = httpx.Timeout(TIMEOUT_SECONDS, connect=30.0)
    try:
        # 内网调用不读取 HTTP_PROXY/HTTPS_PROXY，避免请求误走代理。
        with httpx.Client(timeout=request_timeout, trust_env=False) as client:
            check_health(client)
            with media_path.open("rb") as media_file, srt_path.open("rb") as srt_file:
                response = client.post(
                    align_url,
                    files={
                        "media": (media_path.name, media_file, media_type),
                        "srt": (
                            srt_path.name,
                            srt_file,
                            "application/x-subrip",
                        ),
                    },
                )
    except httpx.TimeoutException as exc:
        raise SystemExit(
            f"等待服务超过 {TIMEOUT_SECONDS} 秒，未生成输出文件：{exc}"
        ) from exc
    except httpx.HTTPError as exc:
        raise SystemExit(f"请求字幕服务失败，未生成输出文件：{exc}") from exc

    print(f"服务返回：HTTP {response.status_code}")
    print(f"响应大小：{len(response.content)} 字节")
    if response.status_code != 200:
        raise SystemExit(
            f"字幕对齐失败（HTTP {response.status_code}）：{read_error(response)}"
        )

    content_type = response.headers.get("content-type", "")
    if "application/x-subrip" not in content_type.casefold():
        raise SystemExit(
            "服务返回的不是 SRT，"
            f"Content-Type={content_type or '未提供'}"
        )

    try:
        srt_text = response.content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SystemExit("服务返回内容不是有效的 UTF-8 SRT") from exc
    if "-->" not in srt_text:
        raise SystemExit("服务返回内容中没有 SRT 时间轴，拒绝保存")

    temporary_output.write_bytes(response.content)
    temporary_output.replace(output_path)
    print(f"对齐完成，文件已经保存：{output_path}")
    print(f"最终文件大小：{output_path.stat().st_size} 字节")


if __name__ == "__main__":
    main()
