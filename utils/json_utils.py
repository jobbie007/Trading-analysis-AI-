from __future__ import annotations

import json
from typing import Any


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Return the first complete JSON object embedded in `text`.

    Attempts fast-path full parse, then scans for balanced braces.
    """
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = text.find("{")
    while start != -1 and start < len(text):
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start : i + 1]
                    try:
                        obj = json.loads(chunk)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        break
        start = text.find("{", start + 1)
    return None
