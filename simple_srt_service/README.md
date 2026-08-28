# 昇腾 vLLM 极简 SRT 对齐服务

该目录提供与原 GPU 版相同的极简接口：上传一个音频/视频和一个原文
SRT，响应直接返回重新对齐后的 SRT。

推理后端：

- Qwen3-ASR：`qwen-asr` 的 `Qwen3ASRModel.LLM()`，底层使用 vLLM-Ascend。
- Qwen3-ForcedAligner：使用 PyTorch + torch-npu 加载到同一张昇腾卡。
- 两个模型在服务启动时加载并常驻，后续请求不会重复加载。

## 运行环境

该版本按下面的环境路线适配：

- vLLM 0.14.x
- vllm-ascend 0.14.x（包括 0.14.0rc 版本）
- 彼此匹配的 torch 与 torch-npu
- 已安装 `qwen-asr`；不要求通过 `qwen-asr[vllm]` 安装，因为 vLLM 已由
  Ascend 镜像提供
- FFmpeg

不要执行 `pip install -U vllm`，也不要让 `qwen-asr[vllm]` 覆盖 Ascend
镜像中已经匹配的 vLLM、torch 和 torch-npu。

## 配置

首次运行：

```bash
cd /data/yb/Subtitle_align-main
cp simple_srt_service/config.example.yaml simple_srt_service/config.yaml
vim simple_srt_service/config.yaml
```

使用物理 NPU 6 时：

```yaml
server:
  host: "0.0.0.0"
  port: 12046

npu:
  visible_devices: "6"
  logical_device_index: 0

alignment_engine:
  gpu_memory_utilization: 0.85
  max_inference_batch_size: 4
  asr_batch_size: 4
  forced_aligner_batch_size: 4
  max_new_tokens: 1024
  max_model_len: 4096
  enforce_eager: false
  attention_implementation: "eager"
  startup_timeout_seconds: 1800
  language: "auto"

models:
  root: "/data/yb/model"
```

`ASCEND_RT_VISIBLE_DEVICES=6` 会把物理卡 6 隔离给 Worker，因此模型内部使用
`npu:0`。`gpu_memory_utilization` 是 vLLM 的通用参数名，在这里控制的是 NPU
HBM，而不是 NVIDIA GPU。

本机真实 `config.yaml` 已被 `.gitignore` 忽略，不会上传或被 `git pull`
覆盖。旧配置里的 `gpu.visible_devices` 仍可读取，但建议改成上面的 `npu`。

## 启动

前台启动并直接观察日志：

```bash
cd /data/yb/Subtitle_align-main
bash simple_srt_service/start.sh
```

启动脚本会先检查 FFmpeg、模型目录、NPU、qwen-asr、vLLM 0.14、
vllm-ascend 0.14，以及 torch/torch-npu 是否匹配。预检通过后才加载模型。

后台启动：

```bash
cd /data/yb/Subtitle_align-main
nohup bash simple_srt_service/start.sh \
  > simple_srt_service/service.log 2>&1 &
echo $! > simple_srt_service/service.pid
```

查看服务日志和模型 Worker 日志：

```bash
tail -f simple_srt_service/service.log
tail -f simple_srt_service/data/npu-vllm-worker.log
```

看到下面的日志表示模型已经常驻：

```text
Ascend vLLM SRT alignment service is ready
```

停止：

```bash
kill "$(cat simple_srt_service/service.pid)"
```

## 健康检查与调用

```bash
curl --noproxy '*' http://127.0.0.1:12046/health
```

正常结果包含：

```json
{
  "status": "ok",
  "models_resident": true,
  "inference_backend": "qwen-asr-vllm",
  "accelerator": "ascend-npu"
}
```

调用：

```bash
curl --noproxy '*' --fail-with-body \
  -X POST http://127.0.0.1:12046/align \
  -F "media=@/data/yb/Test/en_external_12.wav" \
  -F "srt=@/data/yb/Test/english.srt" \
  -o /data/yb/Test/english.aligned.srt
```

接口文档：`http://服务器IP:12046/docs`

提供给调用方的完整接口说明见 `simple_srt_service/API_CALL_GUIDE.md`。如果希望
直接在代码中写死服务地址和三个文件路径后运行，可以使用：

```bash
python3 simple_srt_service/call_once_hardcoded.py
```

同一张 NPU 一次只执行一个请求，其他请求自动排队。若启动时出现 ACL Graph
编译问题，将 `enforce_eager` 改成 `true`；若 ForcedAligner 加载时 HBM 不足，
先把 `gpu_memory_utilization` 从 `0.85` 降到 `0.75`。

若曾经启动失败，再次启动前先用 `npu-smi info` 检查卡上是否还存在旧的
`EngineCore` 或 Python 进程。旧进程仍占用 NPU 时，应先停止对应的旧服务进程，
确认卡上资源释放后再重启。本服务会强制 vLLM Worker 使用 `spawn`，并避免在
EngineCore 创建前初始化 torch-npu，以防止 `507899 / create stream failed`。
