"""Content fingerprints with canonical, deterministic JSON encoding.

Fingerprints identify *content*, so the same derivation content must always
produce the same id, even after export -> delete local database -> import.
Floats are rounded to a fixed number of significant decimal places so that
platform float noise does not split otherwise identical nodes.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

_SIG_DIGITS = 9


def _canonical(obj: Any) -> Any:
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return repr(obj)
        # Round to a fixed number of decimal places after stabilising -0.0.
        value = round(obj, _SIG_DIGITS)
        if value == 0.0:
            value = 0.0
        return value
    if isinstance(obj, dict):
        return {str(k): _canonical(obj[k]) for k in sorted(obj)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    return obj


def canonical_json(obj: Any) -> str:
    """Deterministic JSON used both for hashing and for exported bundles."""
    return json.dumps(
        _canonical(obj), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def fingerprint(prefix: str, payload: Any) -> str:
    """Return ``prefix_<sha256[:16]>`` for the canonical payload."""
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:16]}"
