"""Content-addressed derived fingerprints.

Every derived node id is sha256 of: node kind, canonical (sorted) JSON of its
direct source revisions and upstream node ids, and all rule parameters -
beam geometry id, transform order, magnetic-declination version, mounting
version, thresholds.  Identical inputs reuse the id; any rule/source change
produces a new id.  This makes "why still valid / why recomputed" decidable
from id equality alone and keeps identities stable across export/import.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False, default=str)


def node_id(kind: str, inputs: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> str:
    payload = {"kind": kind, "inputs": inputs, "params": params or {}}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]
