# GPU 文本数组字幕对齐接口

该接口属于原有 12045 完整平台，复用同一个常驻 GPU Worker，不启动第二份模型，
调用时不需要打开前端页面。

## 后台启动

```bash
cd /data/yb/Subtitle_align-main
bash start_background.sh
tail -f subtitle-align.log
```

正常启动后检查：

```bash
curl --noproxy '*' http://127.0.0.1:12045/api/health
```

## 输入格式

`POST /api/line-jobs` 使用 `multipart/form-data`：

- `media`：音频或视频文件。
- `lines`：JSON 字符串数组，服务保持行顺序，不翻译、不拆分、不合并。

服务根据文本自动选择 Chinese、English 或 Japanese，并固定启用逐行
ForcedAligner 精校，不调用 v4-flash。

提交示例：

```bash
curl --noproxy '*' --fail-with-body \
  -X POST http://127.0.0.1:12045/api/line-jobs \
  -F 'media=@/data/yb/Test/demo.wav' \
  -F 'lines=["我们张常宁","刚才的速度是91，","哎呀。"]'
```

接口立即返回任务 `id`。使用现有接口查询和下载：

```bash
JOB_ID="替换为返回的任务ID"
curl --noproxy '*' "http://127.0.0.1:12045/api/jobs/$JOB_ID"
curl --noproxy '*' --fail-with-body \
  "http://127.0.0.1:12045/api/jobs/$JOB_ID/download/srt" \
  -o /data/yb/Test/demo.aligned.srt
curl --noproxy '*' --fail-with-body \
  "http://127.0.0.1:12045/api/jobs/$JOB_ID/download/jsonl" \
  -o /data/yb/Test/demo.aligned.jsonl
```

查询状态只是读取进度，不会重复执行任务。异步任务方式可以避免长音频经过网关时
发生 504。

## Python 一次调用

仓库根目录的 `call_lines_api.py` 已写入完整示例。只需修改文件顶部的服务地址、
音视频路径、输出路径和 `LINES`：

```bash
cd /data/yb/Subtitle_align-main
python3 call_lines_api.py
```

脚本会提交一次任务、持续查询状态，并自动下载 SRT 和 JSONL。

如果 `config.yaml` 配置了 `security.api_key`，调用时需要发送
`X-API-Key`。Python 示例从环境变量读取，不要将真实 Key 写入代码：

```bash
export SUBALIGN_API_KEY='你们的Key'
python3 call_lines_api.py
```
