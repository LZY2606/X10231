from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_fingerprint(parts: dict[str, Any]) -> str:
    encoded = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
