"""Stable identity and derived fingerprints.

Identity vs fingerprint
-----------------------
- ``stable_id`` identifies *what* a display slot is (ping 12, cell 3, beam 2).
  It survives export/import and never changes when a source is revised.
- ``value_id`` identifies the exact value at that slot (hash of content plus
  dependency fingerprints). It changes exactly when the value changes.

Coordinate transform order, beam geometry version and declination version are
all explicit inputs, so they necessarily enter every downstream fingerprint.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, Sequence


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def fingerprint(payload: Any) -> str:
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return "sha256:" + digest[:16]


def derived_fingerprint(rule: str, inputs: Dict[str, Any], upstream: Sequence[str]) -> str:
    return fingerprint(
        {
            "rule": rule,
            "inputs": inputs,
            "upstream_fingerprints": list(upstream),
        }
    )


def make_stable_id(parts: Iterable[str]) -> str:
    return "/".join(str(part) for part in parts)


def value_id(rule: str, inputs: Dict[str, Any], upstream_fingerprints: Sequence[str]) -> str:
    return derived_fingerprint(rule, inputs, upstream_fingerprints)
