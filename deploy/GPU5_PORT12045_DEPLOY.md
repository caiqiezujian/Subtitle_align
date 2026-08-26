# GPU 5 + 12045 端口完整部署指南

本文用于在 Linux GPU 服务器上部署本项目，固定使用物理 GPU 5，并通过 TCP `12045` 对外提供服务。

假设：

- 项目目录：`/data/yb/Code/Subtitle_align`
- 模型目录：`/data/yb/Code/models`
- 系统可以执行 `nvidia-smi`
- 两个模型已经下载到本地

如果实际路径不同，请替换文中的路径。

## 1. 拉取代码并准备系统环境

首次部署：

```bash
cd /data/yb/Code
git clone https://github.com/caiqiezujian/Subtitle_align.git
cd Subtitle_align
```

已经 Clone 过时：

```bash
cd /data/yb/Code/Subtitle_align
git pull origin main
```

确认 GPU 5 存在：

```bash
nvidia-smi -i 5
```

安装 FFmpeg。Ubuntu/Debian：

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg python3-venv
```

CentOS/RHEL 请使用服务器已经配置的 FFmpeg 软件源；安装后确认：

```bash
ffmpeg -version
```

官方 Qwen3-ASR 推荐使用独立 Python 3.12 环境。已有可工作的 Qwen/vLLM 环境时，优先继续使用该环境，避免重复安装或改变 CUDA 依赖。

新建虚拟环境的示例：

```bash
cd /data/yb/Code/Subtitle_align
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-aligner.txt
```

如果 FlashAttention 2 与服务器 GPU、CUDA 和 PyTorch 版本兼容，可选择安装：

```bash
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

FlashAttention 不是服务启动的必需条件。没有安装时，不要在页面勾选对应选项。

## 2. 配置 GPU 5、模型和 v4-flash

复制环境变量模板：

```bash
cd /data/yb/Code/Subtitle_align
cp .env.example .env
chmod 600 .env
vim .env
```

推荐配置：

```dotenv
# 只让程序使用物理 GPU 5。程序内部显示 cuda:0 属于正常映射。
CUDA_VISIBLE_DEVICES=5

# 模型根目录；下面必须有两个对应的模型文件夹。
QWEN_MODEL_ROOT=/data/yb/Code/models

# 上传文件、日志中间产物和结果保存位置。
SUBALIGN_DATA_DIR=/data/yb/Code/Subtitle_align/data
SUBALIGN_MAX_UPLOAD_MB=4096

# 单卡只运行一个任务，避免并发加载模型导致显存不足。
SUBALIGN_MAX_CONCURRENT_JOBS=1

# 内网服务建议设置；设置后网页高级设置和 API 调用都需要此 Key。
SUBALIGN_API_KEY=
SUBALIGN_ALLOW_ORIGINS=

# 公司内部 v4-flash；按实际 URL 和 Key 替换。
V4_FLASH_ENABLED=true
V4_FLASH_BASE_URL=http://内部-v4-flash-地址/v1
V4_FLASH_API_KEY=替换为真实Key
V4_FLASH_MODEL=dsv4
V4_FLASH_TIMEOUT_SECONDS=120
V4_FLASH_RETRY_COUNT=2

# 内部 HTTPS 使用自签证书时设为 false；正式可信证书可设为 true。
V4_FLASH_VERIFY_SSL=false
```

不要把 `.env` 提交到 Git。项目已经在 `.gitignore` 中排除了它。

检查模型目录：

```bash
test -d /data/yb/Code/models/Qwen3-ASR-1.7B && echo "ASR model OK"
test -d /data/yb/Code/models/Qwen3-ForcedAligner-0.6B && echo "Aligner model OK"
```

两个目录都必须存在：

```text
/data/yb/Code/models/
├── Qwen3-ASR-1.7B/
└── Qwen3-ForcedAligner-0.6B/
```

## 3. 在 12045 端口启动服务

### 3.1 首次前台启动

先在前台运行一次，便于观察错误：

```bash
cd /data/yb/Code/Subtitle_align
source .venv/bin/activate

set -a
source .env
set +a

python -m uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 12045 \
  --workers 1
```

必须保持 `--workers 1`。多个 Uvicorn 进程会各自建立任务队列，可能同时加载模型并争抢 GPU 5 的显存。

服务成功启动后，另开一个终端检查：

```bash
curl http://127.0.0.1:12045/api/health
```

确认无误后按 `Ctrl+C` 停止前台服务，再配置 systemd。

### 3.2 使用 systemd 长期运行

先确认 Python 的绝对路径：

```bash
cd /data/yb/Code/Subtitle_align
source .venv/bin/activate
which python
```

创建服务文件：

```bash
sudo vim /etc/systemd/system/subtitle-align.service
```

写入以下内容；如果用户名、项目目录或 Python 路径不同，必须按实际情况替换：

```ini
[Unit]
Description=Subtitle Alignment Service on GPU 5
After=network.target

[Service]
Type=simple
User=yb
Group=yb
WorkingDirectory=/data/yb/Code/Subtitle_align
EnvironmentFile=/data/yb/Code/Subtitle_align/.env
ExecStart=/data/yb/Code/Subtitle_align/.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 12045 --workers 1
Restart=always
RestartSec=5
TimeoutStopSec=30
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
```

如果服务器用户不是 `yb`，查看当前用户名并替换 `User` 和 `Group`：

```bash
whoami
id -gn
```

启动并设置开机自启：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now subtitle-align
sudo systemctl status subtitle-align --no-pager
```

实时查看日志：

```bash
sudo journalctl -u subtitle-align -f
```

常用管理命令：

```bash
sudo systemctl restart subtitle-align
sudo systemctl stop subtitle-align
sudo systemctl start subtitle-align
```

## 4. 开放 TCP 12045 端口

先确认服务确实监听：

```bash
ss -lntp | grep 12045
```

Ubuntu 使用 UFW：

```bash
sudo ufw allow 12045/tcp
sudo ufw reload
sudo ufw status
```

CentOS/RHEL 使用 firewalld：

```bash
sudo firewall-cmd --permanent --add-port=12045/tcp
sudo firewall-cmd --reload
sudo firewall-cmd --list-ports
```

服务器如果位于云平台，还需要在云安全组、机房网络策略或公司防火墙中放行入站 TCP `12045`。建议只允许公司内网网段访问，不要无保护地开放到公网。

浏览器访问：

```text
http://服务器IP:12045
```

API 文档：

```text
http://服务器IP:12045/docs
```

## 5. 验证、使用和日常更新

### 5.1 健康检查

```bash
curl -s http://127.0.0.1:12045/api/health | python -m json.tool
```

正常情况下应满足：

- `status` 为 `ok`
- `ffmpeg` 为 `true`
- `models.asr` 为 `true`
- `models.forced_aligner` 为 `true`
- 配置 v4-flash 后，`v4_flash` 为 `true`
- `max_concurrent_jobs` 为 `1`

`v4_flash: true` 只代表服务配置已经启用且 URL 存在。首次正式任务仍需要确认内部 URL、Key 和网络连通性正确。

### 5.2 验证 GPU 5

提交一个实际任务后运行：

```bash
watch -n 1 nvidia-smi -i 5
```

应当看到该卡出现 Python/vLLM 进程和显存占用。由于 `CUDA_VISIBLE_DEVICES=5` 会进行设备映射，程序日志中出现 `cuda:0` 不代表使用了物理卡 0。

### 5.3 API 提交示例

未设置 `SUBALIGN_API_KEY`：

```bash
curl -X POST http://127.0.0.1:12045/api/jobs \
  -F "media=@/path/to/demo.wav" \
  -F "transcript=@/path/to/demo.txt" \
  -F "language=Chinese" \
  -F "use_flash=true"
```

设置了 `SUBALIGN_API_KEY`：

```bash
curl -X POST http://127.0.0.1:12045/api/jobs \
  -H "X-API-Key: 你的服务访问Key" \
  -F "media=@/path/to/demo.wav" \
  -F "transcript=@/path/to/demo.txt" \
  -F "language=Chinese" \
  -F "use_flash=true"
```

接口返回任务 ID 后查询进度：

```bash
curl http://127.0.0.1:12045/api/jobs/任务ID
```

如配置了访问 Key，查询和下载接口同样需要 `X-API-Key` 请求头。

### 5.4 更新代码

```bash
cd /data/yb/Code/Subtitle_align
git pull origin main
source .venv/bin/activate
pip install -r requirements-aligner.txt
sudo systemctl restart subtitle-align
sudo systemctl status subtitle-align --no-pager
```

`.env` 和 `data/` 不会被 Git Pull 覆盖。

### 5.5 常见问题

#### 健康检查显示 degraded

检查 FFmpeg 和模型路径：

```bash
which ffmpeg
grep QWEN_MODEL_ROOT .env
ls -la /data/yb/Code/models/Qwen3-ASR-1.7B
ls -la /data/yb/Code/models/Qwen3-ForcedAligner-0.6B
```

#### 端口无法从其他电脑访问

依次检查：

```bash
systemctl status subtitle-align --no-pager
ss -lntp | grep 12045
curl http://127.0.0.1:12045/api/health
```

本机可以访问、其他电脑不能访问，通常是系统防火墙、云安全组或公司网络策略未放行。

#### GPU 显存不足

- 确认 `--workers 1`
- 确认 `SUBALIGN_MAX_CONCURRENT_JOBS=1`
- 使用 `nvidia-smi -i 5` 检查是否有其他程序占用
- 不兼容时不要勾选 FlashAttention 2
- 必要时降低 `jsonl_forced_align.py` 中的批次默认值后重新测试

#### v4-flash 不可用

检查配置是否加载：

```bash
sudo systemctl show subtitle-align --property=EnvironmentFiles
grep '^V4_FLASH_' /data/yb/Code/Subtitle_align/.env | sed 's/API_KEY=.*/API_KEY=***已隐藏***/'
```

不要把真实 Key 输出到聊天、日志或 Git 仓库。v4-flash 最终失败时，服务会回退到确定性清洗并继续对齐；详细原因可在 systemd 日志中查看。

#### 查看单个任务日志

```bash
find /data/yb/Code/Subtitle_align/data/jobs -name alignment.log -type f -printf '%T@ %p\n' \
  | sort -nr \
  | head
```

任务目录结构：

```text
data/jobs/<任务ID>/
├── job.json
├── alignment.log
├── source.normalized.jsonl
├── uploads/
└── results/
    ├── aligned.srt
    └── aligned.jsonl
```
