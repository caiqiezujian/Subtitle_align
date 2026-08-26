FROM qwenllm/qwen3-asr:latest

WORKDIR /srv/subtitle-align
COPY requirements-web.txt ./
RUN python -m pip install --no-cache-dir -r requirements-web.txt

COPY . .
RUN cp config.example.yaml config.yaml && mkdir -p /srv/subtitle-align/data

ENV PYTHONUNBUFFERED=1 \
    SUBALIGN_DATA_DIR=/srv/subtitle-align/data

EXPOSE 12045
CMD ["python", "start_server.py"]
