# 昇腾 vLLM 字幕对齐服务接口调用说明

## 1. 服务信息

| 项目 | 内容 |
| --- | --- |
| 服务地址 | `http://10.170.23.103:12046` |
| 健康检查 | `GET /health` |
| 字幕对齐 | `POST /align` |
| 在线接口文档 | `http://10.170.23.103:12046/docs` |
| 请求格式 | `multipart/form-data` |
| 成功响应 | UTF-8 编码的 SRT 文件 |
| 推理后端 | Qwen3-ASR + qwen-asr + vLLM-Ascend |

该服务部署在公司内网，当前无需提供 API Key。

## 2. 功能与输入要求

调用方上传一个音频或视频文件，以及一个原文 SRT。服务根据媒体中的实际语音，
重新计算每条字幕的开始和结束时间，并直接返回新的 SRT 文件。

输入必须满足：

- SRT 文本是媒体中实际说出的原文，不能使用译文或摘要；
- 字幕文本顺序必须与媒体中的语音顺序一致；
- SRT 文件名必须以 `.srt` 结尾；
- 音视频格式必须能够被 FFmpeg 读取；
- 一条字幕建议是一句话或一个自然语义片段；
- 只有标点符号的字幕行可以保留，服务会根据相邻行补齐时间；
- 默认单个上传文件最大为 4096 MB。

## 3. 健康检查

提交文件前建议先检查服务：

```bash
curl --noproxy '*' -sS "http://10.170.23.103:12046/health"
```

正常响应类似：

```json
{
  "status": "ok",
  "models_resident": true,
  "worker": "ready",
  "inference_backend": "qwen-asr-vllm",
  "accelerator": "ascend-npu",
  "npu_visible_devices": "6",
  "npu_device": "npu:0",
  "gpu_memory_utilization": 0.85
}
```

只有同时满足下面三个条件时才建议提交任务：

- `status` 等于 `ok`；
- `models_resident` 等于 `true`；
- `worker` 等于 `ready`。

## 4. 字幕对齐接口

### 请求

```text
POST http://10.170.23.103:12046/align
Content-Type: multipart/form-data
```

请求必须包含且仅需要下面两个文件字段：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `media` | 文件 | 是 | 音频或视频文件 |
| `srt` | 文件 | 是 | 与媒体内容对应的原文 SRT |

字段名必须准确写成 `media` 和 `srt`。

### 成功响应

```text
HTTP 200
Content-Type: application/x-subrip; charset=utf-8
```

响应正文就是完整的 SRT 文件，不是 JSON。调用方应将响应字节直接保存为
`.srt` 文件。

响应头包含 `X-Request-ID`。当结果质量异常或服务返回错误时，请把该 ID 提供给
服务维护人员，以便在 `simple_srt_service/data/diagnostics/` 中定位本次请求的
独立日志、ASR token 时间轴、原文映射、逐行校准阶段和质量摘要。诊断文件包含
字幕及 ASR 文本，只应由服务维护人员访问。

这是同步接口：每次调用只会提交一个任务，连接会持续等待，直到对齐完成或发生
错误。长音视频建议把客户端读取超时设置为 2 小时或更长。不要因为暂时没有返回
就重复提交同一文件。

## 5. curl 调用

Linux：

```bash
curl --noproxy '*' --fail-with-body \
  -X POST "http://10.170.23.103:12046/align" \
  -F "media=@/data/yb/Test/en_external_12.wav" \
  -F "srt=@/data/yb/Test/english.srt" \
  -o /data/yb/Test/english.aligned.srt
```

`@` 必须保留，它表示读取并上传本地文件。没有 `@` 时，curl 只会把路径文字发送
给服务。

Windows PowerShell：

```powershell
curl.exe --noproxy "*" --fail-with-body `
  -X POST "http://10.170.23.103:12046/align" `
  -F "media=@D:\Test\demo.mp4" `
  -F "srt=@D:\Test\demo.srt" `
  -o "D:\Test\demo.aligned.srt"
```

## 6. 固定参数 Python 脚本

项目已经提供：

```text
simple_srt_service/call_once_hardcoded.py
```

只需修改脚本顶部的四项配置：

```python
SERVER_URL = "http://10.170.23.103:12046"
MEDIA_PATH = Path("/data/yb/Test/en_external_12.wav")
SRT_PATH = Path("/data/yb/Test/english.srt")
OUTPUT_PATH = Path("/data/yb/Test/english.aligned.srt")
```

运行：

```bash
cd /data/yb/Subtitle_align-main
python3 simple_srt_service/call_once_hardcoded.py
```

脚本会先检查 `/health`，然后只提交一次对齐请求。返回内容通过状态码、媒体类型、
UTF-8 解码和 SRT 时间轴检查后，才会写入最终输出文件。

## 7. 常见响应

| 状态码 | 含义 | 处理建议 |
| --- | --- | --- |
| 200 | 对齐成功 | 将响应保存为 `.srt` |
| 400 | SRT 后缀、编码、内容或格式错误 | 检查输入 SRT |
| 413 | 上传文件超过服务限制 | 检查文件大小和服务配置 |
| 422 | 缺少字段或字段名错误 | 确认使用 `media` 和 `srt` |
| 500 | 服务端音轨解析、ASR 或对齐失败 | 保存响应详情并联系维护人员 |

连接被拒绝时先运行健康检查。请求长时间没有返回时，可能是正在处理长音视频，
或者前面已有任务排队；不要直接再次提交。

## 8. 输出验证

调用完成后可以检查：

```bash
ls -lh /data/yb/Test/english.aligned.srt
head -20 /data/yb/Test/english.aligned.srt
```

建议确认：

- 文件不是空文件；
- 字幕序号连续；
- 时间戳按顺序递增；
- 每条字幕的开始时间不晚于结束时间；
- 字幕文本数量和顺序与输入保持一致；
- 可以在 PotPlayer 或 VLC 中随音视频正常展示。
