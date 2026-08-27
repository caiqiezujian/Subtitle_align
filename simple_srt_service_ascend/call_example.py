from __future__ import annotations

import argparse
import json
import mimetypes
from pathlib import Path

import httpx


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="调用 D910B 极简字幕对齐服务并保存返回的 SRT。"
    )
    parser.add_argument("media", type=Path, help="本地音频或视频文件")
    parser.add_argument("srt", type=Path, help="与媒体原文对应的 SRT 文件")
    parser.add_argument(
        "--server",
        default="http://127.0.0.1:12045",
        help="服务地址",
    )
    parser.add_argument("--output", type=Path, help="输出 SRT 路径")
    parser.add_argument(
        "--timeout",
        type=float,
        default=7200,
        help="单次对齐请求超时秒数",
    )
    return parser


def error_message(response: httpx.Response) -> str:
    try:
        value = response.json()
    except (json.JSONDecodeError, ValueError):
        return response.text.strip() or "服务未返回错误详情"
    if isinstance(value, dict) and value.get("detail"):
        return str(value["detail"])
    return json.dumps(value, ensure_ascii=False)


def main() -> None:
    args = build_parser().parse_args()
    media_path = args.media.expanduser().resolve()
    srt_path = args.srt.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else srt_path.with_name(f"{srt_path.stem}.aligned.srt")
    )

    if not media_path.is_file():
        raise SystemExit(f"找不到音视频文件：{media_path}")
    if not srt_path.is_file():
        raise SystemExit(f"找不到 SRT 文件：{srt_path}")
    if srt_path.suffix.casefold() != ".srt":
        raise SystemExit(f"字幕文件必须以 .srt 结尾：{srt_path}")
    if output_path == srt_path:
        raise SystemExit("输出路径不能覆盖输入 SRT")

    server = args.server.rstrip("/")
    timeout = httpx.Timeout(args.timeout, connect=15.0)
    try:
        with httpx.Client(timeout=timeout) as client:
            health = client.get(f"{server}/health")
            health.raise_for_status()
            status = health.json()
            if status.get("status") != "ok" or not status.get("models_resident"):
                raise SystemExit(
                    "服务尚未就绪：" + json.dumps(status, ensure_ascii=False)
                )

            media_type = (
                mimetypes.guess_type(media_path.name)[0]
                or "application/octet-stream"
            )
            with media_path.open("rb") as media_handle, srt_path.open(
                "rb"
            ) as srt_handle:
                response = client.post(
                    f"{server}/align",
                    files={
                        "media": (media_path.name, media_handle, media_type),
                        "srt": (srt_path.name, srt_handle, "application/x-subrip"),
                    },
                )
    except httpx.HTTPError as exc:
        raise SystemExit(f"无法调用字幕服务：{exc}") from exc

    if response.status_code != 200:
        raise SystemExit(
            f"字幕对齐失败（HTTP {response.status_code}）：{error_message(response)}"
        )
    if "application/x-subrip" not in response.headers.get("content-type", ""):
        raise SystemExit(
            "服务返回的不是 SRT："
            + response.headers.get("content-type", "unknown content-type")
        )
    try:
        text = response.content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SystemExit("服务返回的 SRT 不是 UTF-8 编码") from exc
    if "-->" not in text:
        raise SystemExit("服务返回内容中没有有效的 SRT 时间轴")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)
    print(f"对齐完成：{output_path}")


if __name__ == "__main__":
    main()

