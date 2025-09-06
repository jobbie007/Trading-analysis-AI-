import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional


_LOG_DIR = os.getenv("DASH_LOG_DIR", os.path.join(os.path.dirname(__file__), "logs"))
_LOG_FILE = os.getenv("DASH_LOG_FILE", os.path.join(_LOG_DIR, "entropy.jsonl"))


def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def log_event(event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
    """Append a single JSON line with timestamp and event_type.
    Keep it robust: never raise to callers.
    """
    try:
        # Allow disabling logs via env toggle
        if (os.getenv("DASH_DISABLE_LOGS", "").strip().lower() in ("1", "true", "yes")):
            return
        _ensure_dir(_LOG_DIR)
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "payload": payload or {},
        }
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        # Swallow logging errors silently
        pass


class Timer:
    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self._t0
        return False
