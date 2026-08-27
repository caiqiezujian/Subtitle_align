# 服务器更新命令

确认没有正在运行的对齐任务，然后按照顺序逐条执行。

## 1. 进入项目目录

```bash
cd /data/yb/Code/Subtitle_align
```

## 2. 拉取最新代码

```bash
git pull --ff-only origin main
```

## 3. 停止旧服务

```bash
pkill -f "python3 start_server.py"
```

## 4. 启动新服务

```bash
nohup bash start.sh > subtitle-align.log 2>&1 &
```

## 5. 查看启动日志

```bash
tail -f subtitle-align.log
```

看到 `Persistent GPU worker is ready` 表示模型已经加载完成。按 `Ctrl + C` 只会退出日志查看，不会停止服务。

## 6. 检查服务状态

```bash
curl -s http://127.0.0.1:12045/api/health | python3 -m json.tool
```

浏览器访问 `http://服务器IP:12045`，按 `Ctrl + F5` 刷新页面。

`config.yaml` 已被 Git 忽略，更新代码不会覆盖服务器配置和内部 Key。
