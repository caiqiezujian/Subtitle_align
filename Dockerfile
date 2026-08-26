FROM qwenllm/qwen3-asr:latest

WORKDIR /srv/subtitle-align
COPY requirements-web.txt ./
RUN python -m pip install --no-cache-dir -r requirements-web.txt

COPY . .
RUN mkdir -p /srv/subtitle-align/data

ENV PYTHONUNBUFFERED=1 \
    SUBALIGN_DATA_DIR=/srv/subtitle-align/data

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
