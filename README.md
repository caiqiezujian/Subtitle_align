# 声轨刻度：音视频字幕对齐服务

一个可部署在内网 GPU 服务器上的完整字幕对齐项目。用户上传音频/视频和“每行一句”的原文，服务使用 Qwen3-ASR + Qwen3-ForcedAligner 完成长音频双阶段对齐，并同时生成：

- `SRT`：UTF-8 BOM 编码，可直接加载到 PotPlayer。
- `JSONL`：保留逐行文本、毫秒级起止时间、时长、状态、对齐方法和输入来源字段。

前端、API、GPU 任务队列、输入适配、v4-flash 辅助清洗、部署文件和项目内 Codex Skill 均已包含在仓库中。

## D910B / Ascend 独立版本

仓库中的 `simple_srt_service_ascend/` 是为昇腾 D910B 单独整理的
Transformers + torch-npu 版本：接口只接收 `media` 和 `srt`，只返回对齐后的
SRT，两个模型在启动时加载并常驻 NPU。它与现有 NVIDIA/vLLM 服务相互独立，
部署前请阅读 `simple_srt_service_ascend/README.md`，不要在仍承载其他业务的
vLLM 生产容器中直接修改 Transformers 版本。

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
  jobs.py          单 GPU 任务队列、状态持久化与常驻 Worker 调度
  gpu_worker_client.py  常驻 GPU Worker 的进程通信客户端
  transcript.py    TXT/SRT/JSONL/JSON/CSV/TSV 自动适配
  llm_adapter.py   可选的 v4-flash OpenAI-compatible 适配器
  srt.py           标准 JSONL 与 PotPlayer SRT 输出
  static/          响应式前端
jsonl_forced_align.py  原有 Qwen3 对齐核心（增加了方法字段输出）
persistent_gpu_worker.py  启动时加载并持续持有 ASR/ForcedAligner
.codex/skills/subtitle-alignment-ops/  项目内 Codex Skill
deploy/            systemd 与 Nginx 示例
tests/             不需要 GPU 的输入/输出单元测试
config.example.yaml  唯一配置模板
start_server.py      读取 config.yaml 并启动服务
start.sh             容器内一条命令启动
```

## 已有 Qwen Docker 容器：最简启动

不需要创建虚拟环境，也不需要手动 `export`。进入已经能运行原后端代码的容器后：

```bash
cd /data/yb/Code/Subtitle_align
git pull origin main
python3 -m pip install -r requirements-web.txt
cp config.example.yaml config.yaml
vim config.yaml
bash start.sh
```

以后修改配置只编辑 `config.yaml`，启动只执行 `bash start.sh`。打开 `http://服务器地址:12045`，接口文档在 `/docs`，健康检查在 `/api/health`。

`config.yaml` 中统一配置：

```yaml
server:
  host: "0.0.0.0"
  port: 12045
  workers: 1

gpu:
  visible_devices: "5"
  max_concurrent_jobs: 1

alignment_engine:
  gpu_memory_utilization: 0.65
  max_inference_batch_size: 32
  max_new_tokens: 2048
  flash_attention: false
  startup_timeout_seconds: 900

models:
  root: "/data/yb/Code/models"

storage:
  data_dir: "/data/yb/Code/Subtitle_align/data"

v4_flash:
  enabled: true
  base_url: "http://内部模型地址/v1"
  api_key: "替换为真实Key"
  model: "dsv4"
  timeout_seconds: 120
  retry_count: 2
  verify_ssl: false
```

`config.yaml` 已被 Git 忽略，不会把真实 Key 推送到仓库。环境变量仍作为可选的兼容覆盖方式存在，但普通部署不需要使用。

服务启动时会立即拉起独立 GPU Worker，加载 Qwen3-ASR 和 ForcedAligner，并保持模型常驻 GPU。启动日志出现 `Persistent GPU worker is ready` 后页面才开始接受访问；后续任务复用同一模型实例，不会重复加载和释放显存。

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
cp config.example.yaml config.yaml
vim config.yaml
docker compose up -d --build
docker compose logs -f subtitle-align
```

生产环境仍只运行一个 Uvicorn worker。GPU 并发由 `SUBALIGN_MAX_CONCURRENT_JOBS` 控制，默认 `1`；不要用多个 Web worker 横向复制内存队列，否则会重复争抢显存。需要多 GPU/多实例时，应把队列升级为 Redis/Celery 一类的外部持久队列。

`deploy/subtitle-align.service` 和 `deploy/nginx.conf` 提供非 Docker 的 systemd 与 Nginx 参考。部署公网前应配置 HTTPS、网关登录或至少设置 `config.yaml` 中的 `security.api_key`。

## v4-flash 接入

v4-flash 必须提供 OpenAI-compatible `POST /chat/completions` 接口。示例：

```yaml
v4_flash:
  enabled: true
  base_url: "http://内部模型地址/v1"
  api_key: "替换为真实密钥"
  model: "dsv4"
  timeout_seconds: 120
  retry_count: 2
  verify_ssl: false
```

调用方式与 `tranx-mtqe` 一致：向 `base_url + /chat/completions` 发送 OpenAI-compatible 请求，使用 Bearer Key（Key 为空则不发送鉴权头）、`model=dsv4`、`max_tokens=8192`，禁用系统代理，并允许通过 `verify_ssl: false` 适配内部自签证书。

它只处理逐行文本，不会接收音视频。提示词强制要求“不翻译、不改写、不增删、不合并、不拆分”，并兼容纯 JSON、Markdown JSON 围栏和前后附带说明三种返回形式。返回行数不一致、空行、超时或接口异常时会自动重试，最终仍失败则回退到确定性清洗，任务继续执行。

任务进度来自真实流水线事件：输入解析、v4-flash 分批清洗、音轨标准化、VAD 切分、ASR 分段、全局文本匹配、ForcedAligner 分批精修、缺失行插值和文件生成，不再使用固定的 22% 等待值。

纯标点或符号行不送入 ForcedAligner。若它位于两条成功字幕之间，会使用“上一条结束时间 → 下一条开始时间”；连续多条无法对齐的字幕会均分这段空档。JSONL 的 `method` 会标记为 `punctuation_gap_interpolation` 或 `unresolved_gap_interpolation`。

## API 示例

```bash
curl -X POST http://127.0.0.1:12045/api/jobs \
  -H "X-API-Key: 你在config.yaml里设置的服务访问Key" \
  -F "media=@demo.wav" \
  -F "transcript=@demo.txt" \
  -F "language=Chinese" \
  -F "use_flash=false"

curl -H "X-API-Key: 你在config.yaml里设置的服务访问Key" \
  http://127.0.0.1:12045/api/jobs/任务ID
```

返回 `completed` 后使用响应中的 `download_urls` 下载文件。

## 页面内同步验收

任务完成后，页面会自动打开“同步播放验收”工作台：

- 直接播放本次上传的原始音频或视频，不需要再次上传或占用服务器下载带宽。
- 自动读取最终生成的 SRT，在播放器画面上同步显示当前字幕。
- 右侧按时间顺序展示完整字幕轨，点击任意一句即可跳转并播放对应位置。
- 当前字幕会随播放自动高亮，同时支持字幕开关、0.75–2 倍速和全屏验收。
- 页面在宽屏使用播放器与字幕轨双栏布局，中等窗口和手机自动切换为上下布局。

页面播放能力取决于浏览器自身支持的媒体编码。MP4、WebM、MP3、WAV 等常见格式可以直接验收；少数 MKV、WMV 或特殊编码仍可正常参与服务器对齐，但建议下载 SRT 后使用 PotPlayer 复核。

JSONL 每行示例：

```json
{"index":1,"text":"你好，欢迎回来。","start":1.12,"end":2.64,"duration":1.52,"status":"aligned","method":"local_forced_aligner","source":{"physical_line":1}}
```

## 测试

输入解析和输出格式测试不加载 GPU 模型：

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m pytest
```

实际 GPU 验收建议使用一段 20–60 秒、人工知道台词的音频，确认：任务完成、SRT 在 PotPlayer 正常显示、JSONL 行数等于原文非空行数、时间单调且未解析行被明确标为 `unresolved`。

不要提交 `config.yaml`、API Key、模型权重和 `data/` 任务文件；这些路径已经写入 `.gitignore`。
