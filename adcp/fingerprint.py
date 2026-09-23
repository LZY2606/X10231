"""Stable hashes for derived values and dependency sets."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Sequence


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any, length: int = 16) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]


def derived_value_id(dataset: str, ping: str, cell: str, stage: str, candidate: str) -> str:
    raw = f"dv:{dataset}:{ping}:{cell}:{stage}:{candidate}"
    return "dv_" + stable_hash(raw, 20)


def derivation_fingerprint(inputs: Sequence[dict], rule: dict) -> str:
    material = {"rule": rule, "inputs": sorted((i.get("record_id"), i.get("version_hash")) for i in inputs)}
    return "fp_" + stable_hash(material, 24)
