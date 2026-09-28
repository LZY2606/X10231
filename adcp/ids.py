"""Stable content identifiers.

Every displayed value, source record, event and derived node carries an id
that is a content hash of a canonical representation, so identity/order survive
export -> delete local DB -> import.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

PREFIXES = {
    "beam": "beam", "corr": "corr", "amp": "amp", "badflag": "badf",
    "att": "att", "dec": "decl", "mount": "mnt", "geo": "geom",
    "config": "cfg", "cell": "cell", "bt": "bt",
    "event": "ev", "commit": "cm", "branch": "br",
    "ninst": "ni", "nship": "ns", "nearth": "ne", "nqual": "nq",
    "nattres": "na", "ncand4": "c4", "ncand3": "c3", "ncell": "nc",
    "archive": "arc", "decision": "dec",
}


def canonical(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, float round-trip."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False)


def digest(prefix: str, payload: Any) -> str:
    body = hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{body}"
