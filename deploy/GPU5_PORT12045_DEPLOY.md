# Docker 内使用 GPU 5、12045 端口

适用情况：已经进入一个能够正常运行原 Qwen 后端代码的 Docker 容器。无需创建虚拟环境，无需逐项 `export`。

## 1. 更新代码

```bash
cd /data/yb/Code/Subtitle_align
git pull origin main
```

已有 Qwen、vLLM、Torch 环境时，只需要补充 Web 服务依赖：

```bash
python3 -m pip install -r requirements-web.txt
```

## 2. 创建并修改 config.yaml

首次运行复制一次：

```bash
cp config.example.yaml config.yaml
vim config.yaml
```

推荐配置：

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
  max_upload_mb: 4096

security:
  api_key: ""
  allow_origins: []

v4_flash:
  enabled: true
  base_url: "http://内部-v4-flash-地址/v1"
  api_key: "替换为真实Key"
  model: "dsv4"
  timeout_seconds: 120
  retry_count: 2
  verify_ssl: false
```

GPU 编号说明：

- Docker 启动时使用 `--gpus all`：`visible_devices` 填物理卡编号 `"5"`。
- Docker 启动时已经使用 `--gpus '"device=5"'`：物理卡 5 在容器内通常映射为逻辑卡 0，此时填 `"0"`。

用下面命令查看容器内可见编号：

```bash
nvidia-smi -L
```

`config.yaml` 已被 Git 忽略，可以填写真实内部 Key，不会在后续 `git pull` 时被覆盖。

## 3. 启动服务

前台启动：

```bash
bash start.sh
```

后台启动：

```bash
nohup bash start.sh > subtitle-align.log 2>&1 &
```

查看日志：

```bash
tail -f subtitle-align.log
```

停止后台进程：

```bash
pkill -f "python start_server.py"
```

所有 Uvicorn 参数由 `config.yaml` 的 `server` 部分读取。程序会拒绝 `workers` 大于 1，防止重复占用 GPU 显存。

启动阶段会加载 ASR 和 ForcedAligner。看到日志 `Persistent GPU worker is ready` 后，模型已经常驻显存；以后提交任务会直接复用，不再重复加载模型。

## 4. 确认 Docker 端口

如果容器使用 `--network host`，无需额外映射，直接访问：

```text
http://服务器IP:12045
```

如果容器使用普通 bridge 网络，创建容器时必须包含：

```bash
-p 12045:12045
```

已经运行的容器不能动态增加端口映射，需要按原来的启动参数重建容器并补上该参数。

容器内检查：

```bash
curl http://127.0.0.1:12045/api/health
ss -lntp | grep 12045
```

宿主机还应根据实际环境放行 TCP 12045。Ubuntu：

```bash
sudo ufw allow 12045/tcp
```

CentOS/RHEL：

```bash
sudo firewall-cmd --permanent --add-port=12045/tcp
sudo firewall-cmd --reload
```

## 5. 验证和更新

健康检查：

```bash
curl -s http://127.0.0.1:12045/api/health | python3 -m json.tool
```

正常应看到：

- `status: "ok"`
- `ffmpeg: true`
- `models.asr: true`
- `models.forced_aligner: true`
- `v4_flash: true`
- `gpu_worker: "ready"`
- `models_resident: true`
- `max_concurrent_jobs: 1`

提交实际任务后检查 GPU 5：

```bash
watch -n 1 nvidia-smi -i 5
```

以后更新只有三步：

```bash
cd /data/yb/Code/Subtitle_align
git pull origin main
bash start.sh
```

`config.yaml` 和 `data/` 均不会被 Git 覆盖。
