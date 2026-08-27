# D910B 极简 SRT 对齐服务

这是现有 NVIDIA/vLLM 服务的独立 Ascend 版本。它不会修改或替换
`simple_srt_service/`，两套服务可以在不同容器中分别部署。

## 接口

```text
POST /align
media: 音频或视频
srt:   原文 SRT
返回:  application/x-subrip SRT 文件
```

服务启动时使用 Transformers 将以下两个模型加载到 D910B，并由一个常驻
NPU Worker 复用：

```text
Qwen3-ASR-1.7B
Qwen3-ForcedAligner-0.6B
```

该版本不使用 vLLM，也不会安装 `qwen-asr[vllm]`。

## 为什么使用独立容器

当前验证组合将 `qwen-asr` 和 Transformers 锁定为：

```text
qwen-asr==0.0.6
transformers==4.57.6
```

这会改变现有 vLLM 0.18 容器中的 Transformers 5.4.0。建议复制现有 D910B
容器或基于相同 CANN/torch-npu 镜像制作一个字幕服务专用容器，不要直接改动
还要承载其他业务的生产容器。Docker 本身已经提供环境隔离，不要求额外创建
Python 虚拟环境。

基础镜像必须已经提供相互匹配的 CANN、torch 和 torch-npu。不要通过本项目的
requirements 文件安装或替换这三项。

## 安装

进入项目：

```bash
cd /data/yb/Code/Subtitle_align
```

安装系统依赖：

```bash
apt-get update
apt-get install -y ffmpeg libsndfile1 sox
```

先查看 Python 包变更计划：

```bash
python3 -m pip install --dry-run -r simple_srt_service_ascend/requirements.txt
```

确认不会替换 `torch` 或 `torch-npu` 后再安装：

```bash
python3 -m pip install -r simple_srt_service_ascend/requirements.txt
python3 -m pip check
```

如果安装计划准备替换 torch/torch-npu，请停止安装并保留输出，不要使用
`--force-reinstall` 或 `--no-deps` 强行绕过。

## 配置

```bash
cp simple_srt_service_ascend/config.example.yaml \
  simple_srt_service_ascend/config.yaml
vim simple_srt_service_ascend/config.yaml
```

`models.root` 必须是两个模型共同的父目录：

```text
/data/yb/Code/models/
├── Qwen3-ASR-1.7B/
└── Qwen3-ForcedAligner-0.6B/
```

如果容器只映射了一张物理 NPU，通常使用：

```yaml
npu:
  visible_devices: "0"
  logical_device_index: 0
```

首次建议保持较小批次：

```yaml
alignment_engine:
  max_inference_batch_size: 4
  asr_batch_size: 4
  forced_aligner_batch_size: 4
  max_new_tokens: 1024
  attention_implementation: "eager"
```

## 启动

前台启动并观察首次模型加载：

```bash
bash simple_srt_service_ascend/start.sh
```

确认启动正常后，可改为后台常驻：

```bash
nohup bash simple_srt_service_ascend/start.sh \
  > simple-srt-service-ascend.log 2>&1 &
```

查看日志：

```bash
tail -f simple-srt-service-ascend.log
```

NPU Worker 的模型加载和任务日志：

```bash
tail -f simple_srt_service_ascend/data/npu-worker.log
```

健康检查：

```bash
curl -s http://127.0.0.1:12045/health | python3 -m json.tool
```

正常结果包含：

```json
{
  "status": "ok",
  "models_resident": true,
  "inference_backend": "transformers",
  "accelerator": "ascend-npu"
}
```

## 调用

```bash
curl -X POST http://127.0.0.1:12045/align \
  -F "media=@demo.mp4" \
  -F "srt=@demo.srt" \
  -o demo.aligned.srt
```

## 停止和重启

```bash
pkill -f "python3 -m simple_srt_service_ascend.run"
nohup bash simple_srt_service_ascend/start.sh \
  > simple-srt-service-ascend.log 2>&1 &
```

模型路径、NPU 编号、批次或 attention 配置改变后必须重启，因为这些参数只在
常驻 Worker 加载模型时读取。

## 资源与并发

- Uvicorn 固定为一个 Worker，避免重复加载两套模型。
- `/align` 使用进程内锁串行进入 NPU Worker。
- 多个客户端可以提交请求，但会排队处理。
- Transformers 不预留 vLLM KV Cache，因此不再使用 `gpu_memory_utilization`。
- NPU 显存不足时，先把 ASR 和 ForcedAligner 批次从 4 降到 2 或 1。
- 不要安装 NVIDIA `flash-attn`；首次使用 `eager` attention 验证兼容性。

