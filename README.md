# 声轨刻度：音视频字幕对齐服务

一个可部署在内网 GPU 服务器上的完整字幕对齐项目。用户上传音频/视频和“每行一句”的原文，服务使用 Qwen3-ASR + Qwen3-ForcedAligner 完成长音频双阶段对齐，并同时生成：

- `SRT`：UTF-8 BOM 编码，可直接加载到 PotPlayer。
- `JSONL`：保留逐行文本、毫秒级起止时间、时长、状态、对齐方法和输入来源字段。

前端、API、GPU 任务队列、输入适配、v4-flash 辅助清洗、部署文件和项目内 Codex Skill 均已包含在仓库中。

## 支持的输入

媒体可以是 FFmpeg 能解码的常见格式，例如 MP3、WAV、M4A、FLAC、MP4、MKV、MOV、WMV、WEBM。

字幕支持：

| 格式 | 规则 |
|---|---|
| TXT | 每个非空物理行是一句原文 |
| SRT | 读取字幕文本，原时间作为 `source` 保存后重新对齐 |
| JSONL | 每行一个对象，自动寻找原文字段 |
| JSON | 对象数组，或 `lines/items/data/subtitles/segments` 数组 |
| CSV / TSV | 使用表头，自动寻找原文字段 |

结构化文件优先识别 `src`、`text`、`transcript`、`content`、`sentence`、`original`、`source`、`原文`、`字幕`、`台词`。也可以在页面或 API 中明确传入字段名。

> 对齐输入必须是音频里实际说出的原文，不能使用译文。当前对外开放中文、英文、日文，与现有对齐脚本保持一致。

## 项目结构

```text
app/
  main.py          FastAPI 与下载接口
  jobs.py          单 GPU 任务队列、状态持久化与对齐进程
  transcript.py    TXT/SRT/JSONL/JSON/CSV/TSV 自动适配
  llm_adapter.py   可选的 v4-flash OpenAI-compatible 适配器
  srt.py           标准 JSONL 与 PotPlayer SRT 输出
  static/          响应式前端
jsonl_forced_align.py  原有 Qwen3 对齐核心（增加了方法字段输出）
.codex/skills/subtitle-alignment-ops/  项目内 Codex Skill
deploy/            systemd 与 Nginx 示例
tests/             不需要 GPU 的输入/输出单元测试
```

## 本地启动（已有 Qwen 环境）

官方 Qwen3-ASR 项目推荐独立 Python 3.12 环境，并通过 `qwen-asr[vllm]` 启用 vLLM 后端；本服务保留了原脚本的 vLLM 推理方式。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-aligner.txt
cp .env.example .env
set -a && source .env && set +a
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

打开 `http://服务器地址:8000`。接口文档在 `/docs`，健康检查在 `/api/health`。

模型目录应为：

```text
$QWEN_MODEL_ROOT/
  Qwen3-ASR-1.7B/
  Qwen3-ForcedAligner-0.6B/
```

项目默认读取 `/data/yb/Code/models`。如需下载，国内服务器可以使用 ModelScope：

```bash
modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir /data/yb/Code/models/Qwen3-ASR-1.7B
modelscope download --model Qwen/Qwen3-ForcedAligner-0.6B --local_dir /data/yb/Code/models/Qwen3-ForcedAligner-0.6B
```

## Docker 部署

Dockerfile 基于 Qwen 官方 `qwenllm/qwen3-asr:latest` 镜像。服务器需先安装 NVIDIA 驱动、Docker 与 NVIDIA Container Toolkit。

```bash
cp .env.example .env
# 编辑 .env；也可通过 HOST_MODEL_ROOT 指定宿主机模型目录
HOST_MODEL_ROOT=/data/yb/Code/models docker compose up -d --build
docker compose logs -f subtitle-align
```

生产环境仍只运行一个 Uvicorn worker。GPU 并发由 `SUBALIGN_MAX_CONCURRENT_JOBS` 控制，默认 `1`；不要用多个 Web worker 横向复制内存队列，否则会重复争抢显存。需要多 GPU/多实例时，应把队列升级为 Redis/Celery 一类的外部持久队列。

`deploy/subtitle-align.service` 和 `deploy/nginx.conf` 提供非 Docker 的 systemd 与 Nginx 参考。部署公网前应配置 HTTPS、网关登录或至少设置 `SUBALIGN_API_KEY`。

## v4-flash 接入

v4-flash 必须提供 OpenAI-compatible `POST /chat/completions` 接口。示例：

```dotenv
V4_FLASH_ENABLED=true
V4_FLASH_BASE_URL=http://内部模型地址/v1
V4_FLASH_API_KEY=替换为真实密钥
V4_FLASH_MODEL=dsv4
V4_FLASH_TIMEOUT_SECONDS=120
V4_FLASH_RETRY_COUNT=2
V4_FLASH_VERIFY_SSL=false
```

调用方式与 `tranx-mtqe` 一致：向 `$V4_FLASH_BASE_URL/chat/completions` 发送 OpenAI-compatible 请求，使用 Bearer Key（Key 为空则不发送鉴权头）、`model=dsv4`、`max_tokens=8192`，禁用系统代理，并允许通过 `V4_FLASH_VERIFY_SSL=false` 适配内部自签证书。旧的 `V4_ENABLED/V4_BASE_URL/V4_API_KEY/V4_MODEL` 变量仍兼容，但新部署建议使用 `V4_FLASH_*`。

它只处理逐行文本，不会接收音视频。提示词强制要求“不翻译、不改写、不增删、不合并、不拆分”，并兼容纯 JSON、Markdown JSON 围栏和前后附带说明三种返回形式。返回行数不一致、空行、超时或接口异常时会自动重试，最终仍失败则回退到确定性清洗，任务继续执行。

## API 示例

```bash
curl -X POST http://127.0.0.1:8000/api/jobs \
  -H "X-API-Key: $SUBALIGN_API_KEY" \
  -F "media=@demo.wav" \
  -F "transcript=@demo.txt" \
  -F "language=Chinese" \
  -F "use_flash=false"

curl -H "X-API-Key: $SUBALIGN_API_KEY" \
  http://127.0.0.1:8000/api/jobs/任务ID
```

返回 `completed` 后使用响应中的 `download_urls` 下载文件。

JSONL 每行示例：

```json
{"index":1,"text":"你好，欢迎回来。","start":1.12,"end":2.64,"duration":1.52,"status":"aligned","method":"local_forced_aligner","source":{"physical_line":1}}
```

## 测试

输入解析和输出格式测试不加载 GPU 模型：

```bash
pip install -r requirements-dev.txt
pytest
```

实际 GPU 验收建议使用一段 20–60 秒、人工知道台词的音频，确认：任务完成、SRT 在 PotPlayer 正常显示、JSONL 行数等于原文非空行数、时间单调且未解析行被明确标为 `unresolved`。

## 推送到私有 GitCode

当前目录还未绑定远程仓库。创建空白私有仓库后，在本项目目录执行：

```bash
git init
git add .
git commit -m "feat: build subtitle alignment service"
git branch -M main
git remote add origin <你的私有 GitCode 地址>
git push -u origin main
```

不要提交 `.env`、API Key、模型权重和 `data/` 任务文件；这些路径已经写入 `.gitignore`。
