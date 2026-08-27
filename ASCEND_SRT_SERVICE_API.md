# 音视频字幕对齐服务调用文档

## 1. 服务信息

| 项目 | 内容 |
| --- | --- |
| 服务地址 | `http://10.170.23.103:12045` |
| 健康检查 | `GET /health` |
| 字幕对齐 | `POST /align` |
| 接口文档 | `http://10.170.23.103:12045/docs` |
| 请求格式 | `multipart/form-data` |
| 返回格式 | UTF-8 编码的 SRT 文件 |

该服务部署在公司内网，仅供能够访问 `10.170.23.103` 的内部客户端调用。
当前接口无需传递 API Key。

## 2. 功能说明

调用方上传：

1. 一个音频或视频文件；
2. 一个与媒体内容对应的原文 SRT 字幕文件。

服务会根据音视频中的实际语音重新计算每条字幕的开始和结束时间，并直接返回
一个新的 SRT 文件。返回文件可以用于播放器字幕展示或后续数据处理。

字幕文本必须是媒体中实际说出的原文，且顺序必须与音视频一致。请勿上传译文、
摘要或打乱顺序的字幕。

## 3. 健康检查

调用前建议先确认服务和常驻模型已经就绪：

```bash
curl -s "http://10.170.23.103:12045/health"
```

正常结果类似：

```json
{
  "status": "ok",
  "models_resident": true,
  "worker": "ready",
  "inference_backend": "transformers",
  "accelerator": "ascend-npu",
  "npu_visible_devices": "5",
  "npu_device": "npu:0"
}
```

只有当 `status` 为 `ok`、`models_resident` 为 `true`、`worker` 为 `ready`
时才建议提交任务。

## 4. 对齐接口

### 请求

```text
POST http://10.170.23.103:12045/align
Content-Type: multipart/form-data
```

必须使用以下两个字段名：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `media` | 文件 | 是 | 音频或视频文件，格式需能被 FFmpeg 读取 |
| `srt` | 文件 | 是 | 原文 SRT，文件名必须以 `.srt` 结尾 |

不要把本地路径作为普通字符串提交。使用 curl 时，路径前的 `@` 表示读取并上传
该文件，因此 `@` 必须保留。

### 返回

成功时：

```text
HTTP 200
Content-Type: application/x-subrip; charset=utf-8
```

响应正文就是完整的 SRT 文件，不是 JSON。调用方应当把响应内容直接保存为
`.srt` 文件。

该接口是同步接口：连接会一直等待，直到对齐完成或发生错误。长音视频可能需要
较长时间，客户端建议将读取超时设置为 2 小时或更长。

## 5. curl 调用示例

### Linux

```bash
curl -fS -X POST "http://10.170.23.103:12045/align" \
  -F "media=@/data/test/demo.mp4" \
  -F "srt=@/data/test/demo.srt" \
  -o /data/test/demo.aligned.srt
```

如果文件就在当前目录：

```bash
curl -fS -X POST "http://10.170.23.103:12045/align" \
  -F "media=@demo.mp4" \
  -F "srt=@demo.srt" \
  -o demo.aligned.srt
```

### Windows PowerShell

PowerShell 中建议明确使用 `curl.exe`：

```powershell
curl.exe -fS -X POST "http://10.170.23.103:12045/align" `
  -F "media=@D:\test\demo.mp4" `
  -F "srt=@D:\test\demo.srt" `
  -o "D:\test\demo.aligned.srt"
```

## 6. Python 调用示例

安装客户端依赖：

```bash
python3 -m pip install httpx
```

示例代码：

```python
from pathlib import Path

import httpx


SERVER = "http://10.170.23.103:12045"
MEDIA_PATH = Path("demo.mp4")
SRT_PATH = Path("demo.srt")
OUTPUT_PATH = Path("demo.aligned.srt")

timeout = httpx.Timeout(7200.0, connect=15.0)

with MEDIA_PATH.open("rb") as media_file, SRT_PATH.open("rb") as srt_file:
    response = httpx.post(
        f"{SERVER}/align",
        files={
            "media": (MEDIA_PATH.name, media_file, "application/octet-stream"),
            "srt": (SRT_PATH.name, srt_file, "application/x-subrip"),
        },
        timeout=timeout,
    )

if response.status_code != 200:
    raise RuntimeError(
        f"对齐失败：HTTP {response.status_code}，{response.text}"
    )

if "application/x-subrip" not in response.headers.get("content-type", ""):
    raise RuntimeError("服务返回的不是 SRT 文件")

OUTPUT_PATH.write_bytes(response.content)
print(f"对齐完成：{OUTPUT_PATH.resolve()}")
```

## 7. 输入要求

- SRT 必须包含合法的序号、时间轴和字幕文本；原有时间可以不准确。
- 字幕文本必须对应音视频中实际说出的原文，不能使用译文代替。
- 字幕顺序必须与音视频中的语音顺序一致。
- 一条 SRT 字幕建议对应一句话或一个自然语义片段。
- 中英文标点行可以保留，服务会尽量根据前后字幕补齐时间。
- 音视频必须包含清晰可用的语音轨道。
- 默认单个上传文件最大为 4096 MB。

## 8. 输出检查

调用完成后可以快速检查输出：

```bash
ls -lh demo.aligned.srt
head -20 demo.aligned.srt
```

建议确认：

- SRT 序号连续；
- 时间戳按顺序递增；
- 开始时间不晚于结束时间；
- 文件可以作为字幕加载到 PotPlayer、VLC 等播放器；
- 字幕文本数量和顺序与输入一致。

## 9. 常见错误

### 无法连接服务

```text
Connection refused / Could not connect
```

先检查网络是否能访问服务器，再调用健康检查：

```bash
curl -v "http://10.170.23.103:12045/health"
```

### HTTP 400

通常表示：

- `srt` 字段不是 `.srt` 文件；
- SRT 内容为空；
- SRT 编码或格式无法解析；
- 上传字段名不是 `media` 和 `srt`。

### HTTP 413

上传文件超过服务允许的最大大小。

### HTTP 422

通常表示缺少 `media` 或 `srt` 字段，或者请求没有使用
`multipart/form-data`。

### HTTP 500

服务端在音轨解析、ASR 或字幕对齐过程中失败。请保留以下信息并联系服务维护人：

- 调用时间；
- HTTP 状态码和响应内容；
- 音视频格式、时长和文件大小；
- SRT 行数和文本语言。

### 调用长时间没有返回

服务使用单个常驻 NPU Worker 串行处理对齐任务。多人同时提交时，后提交的请求
需要排队等待。不要因为暂时没有返回而立即重复提交同一个大文件，否则会增加
队列压力。

## 10. 最简调用命令

调用方只需要替换三个文件路径：

```bash
curl -fS -X POST "http://10.170.23.103:12045/align" \
  -F "media=@输入音视频路径" \
  -F "srt=@输入字幕路径" \
  -o "输出字幕路径"
```

