"""Deterministic canonical serialization and content fingerprints.

Fingerprints are stable across processes because every byte is derived from
canonical JSON (sorted keys, no insignificant whitespace) and SHA-256.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> str:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def fingerprint(prefix: str, obj: Any) -> str:
    """Return ``prefix`` + first 16 hex chars of the SHA-256 of canonical JSON."""
    digest = hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:16]}"
