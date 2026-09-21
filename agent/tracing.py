import json
import time
import uuid
from pathlib import Path

class Tracer:
    def __init__(self, log_dir:"logs", session_id:str | None = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
        self.path = self.log_dir / f"{self.session_id}.jsonl"

    def log(self, event: str, **fields) -> None:
        record = {"ts": time.time(), "event": event, **fields}
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")