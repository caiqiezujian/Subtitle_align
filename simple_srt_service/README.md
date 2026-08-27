# 极简 SRT 对齐服务

这是一个独立的极简接口层，不修改现有完整服务，也不改变已经验证过的对齐算法。

它只有一个业务接口：

```text
POST /align
```

请求只有两个文件字段：

- `media`：音频或视频。
- `srt`：音频中实际原文对应的 SRT。

响应只有一个文件：重新对齐后的 SRT。

## 配置

```bash
cp simple_srt_service/config.example.yaml simple_srt_service/config.yaml
```

修改 `simple_srt_service/config.yaml` 中的 GPU 编号和模型目录。

## 启动

```bash
nohup bash simple_srt_service/start.sh > simple-srt-service.log 2>&1 &
```

查看日志：

```bash
tail -f simple-srt-service.log
```

看到 `Simple SRT alignment service is ready` 表示模型已加载并常驻显存。

## 调用

```bash
curl -X POST http://127.0.0.1:12045/align \
  -F "media=@/data/demo.mp4" \
  -F "srt=@/data/demo.srt" \
  -o /data/demo.aligned.srt
```

接口文档：`http://服务器IP:12045/docs`

健康检查：

```bash
curl http://127.0.0.1:12045/health
```

同一张 GPU 卡一次只执行一个对齐请求，其他请求会自动等待。SRT 语言默认自动判断，也可以在 `config.yaml` 中固定为 `Chinese`、`English` 或 `Japanese`。

完整服务与极简服务默认都使用 12045 端口，二者不要同时启动。
