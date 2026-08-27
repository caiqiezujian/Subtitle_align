from __future__ import annotations

import json
import mimetypes
from pathlib import Path

import httpx


# ======================== 只需要修改这里 ========================
ALIGN_URL = "http://10.170.23.103:12045/align"
MEDIA_PATH = Path("/data/yb/Test/en_external_12.wav")
SRT_PATH = Path("/data/yb/Test/english.srt")
OUTPUT_PATH = Path("/data/yb/Test/demo.aligned.srt")

# 长音视频可能需要较长时间；这里设置为 2 小时。
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


def main() -> None:
    media_path = MEDIA_PATH.expanduser().resolve()
    srt_path = SRT_PATH.expanduser().resolve()
    output_path = OUTPUT_PATH.expanduser().resolve()
    temporary_output = output_path.with_name(output_path.name + ".part")

    if not media_path.is_file():
        raise SystemExit(f"找不到音视频文件：{media_path}")
    if not srt_path.is_file():
        raise SystemExit(f"找不到 SRT 文件：{srt_path}")
    if srt_path.suffix.casefold() != ".srt":
        raise SystemExit(f"字幕文件必须以 .srt 结尾：{srt_path}")
    if output_path == srt_path:
        raise SystemExit("输出路径不能与输入 SRT 路径相同")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output.unlink(missing_ok=True)
    media_type = mimetypes.guess_type(media_path.name)[0] or "application/octet-stream"

    print("准备发起一次字幕对齐请求")
    print(f"服务地址：{ALIGN_URL}")
    print(f"音视频：{media_path}")
    print(f"原字幕：{srt_path}")
    print("请求发出后会一直等待服务完成，这是一次请求，不会重复提交。")

    timeout = httpx.Timeout(TIMEOUT_SECONDS, connect=30.0)
    try:
        # trust_env=False：不读取 HTTP_PROXY/HTTPS_PROXY，直接访问内部服务。
        with httpx.Client(timeout=timeout, trust_env=False) as client:
            with media_path.open("rb") as media_file, srt_path.open("rb") as srt_file:
                response = client.post(
                    ALIGN_URL,
                    files={
                        "media": (media_path.name, media_file, media_type),
                        "srt": (srt_path.name, srt_file, "application/x-subrip"),
                    },
                )
    except httpx.TimeoutException as exc:
        raise SystemExit(
            f"等待服务返回超过 {TIMEOUT_SECONDS} 秒，未生成输出文件：{exc}"
        ) from exc
    except httpx.HTTPError as exc:
        raise SystemExit(f"请求字幕服务失败，未生成输出文件：{exc}") from exc

    print(f"服务返回：HTTP {response.status_code}")
    print(f"响应大小：{len(response.content)} 字节")
    if response.status_code != 200:
        raise SystemExit(f"字幕对齐失败：{read_error(response)}")

    content_type = response.headers.get("content-type", "")
    if "application/x-subrip" not in content_type.casefold():
        raise SystemExit(f"服务返回的不是 SRT，Content-Type={content_type or '未提供'}")

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
