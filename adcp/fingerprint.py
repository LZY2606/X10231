"""Stable content fingerprints used in derivation identities."""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def fingerprint(prefix: str, obj: Any) -> str:
    digest = hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()
    return "%s_%s" % (prefix, digest[:16])
