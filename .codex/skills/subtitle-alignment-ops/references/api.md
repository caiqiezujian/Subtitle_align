# Subtitle alignment API

Default local address: `http://127.0.0.1:8000`. If `SUBALIGN_API_KEY` is set, send it in the `X-API-Key` header. Never print the key.

## Health

`GET /api/health` returns service version, FFmpeg availability, both model-directory checks, v4-flash availability, and GPU queue concurrency.

## Create a job

`POST /api/jobs` uses `multipart/form-data`:

- `media`: required audio/video file.
- `transcript`: required TXT/SRT/JSONL/JSON/CSV/TSV file.
- `language`: `Chinese`, `English`, or `Japanese`.
- `text_field`: optional structured-file text field.
- `use_flash`: optional boolean.
- `asr_context`: optional terms/names.
- `flash_attention`: optional boolean.
- `local_refine`: boolean; normally keep `true`.

The response is HTTP 202 and includes `id`, `status`, `progress`, and `stage`.

## Poll and download

Poll `GET /api/jobs/{id}` about every two seconds. Stop on `completed` or `failed`. Completed jobs expose `download_urls` for `srt` and `jsonl`.

Download with:

- `GET /api/jobs/{id}/download/srt`
- `GET /api/jobs/{id}/download/jsonl`

Do not infer completion from progress alone; use the terminal job status.
