from __future__ import annotations

import json
import sys


PREFIX = "__SUBALIGN__"


def emit(event: str, **values: object) -> None:
    print(PREFIX + json.dumps({"event": event, **values}), flush=True)


emit("ready")
for raw in sys.stdin:
    command = json.loads(raw)
    if command["command"] == "shutdown":
        emit("stopped")
        break
    emit("completed", job_id=command["job_id"])
