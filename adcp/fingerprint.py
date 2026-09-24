"""Deterministic derived-fingerprints.

A derived object's fingerprint includes every source/rule version it depends
on: the beam geometry version, transform-order version, magnetic-declination
version, mounting-angle version, attitude bracket, quality thresholds, the
exact transform ORDER, and the family (four/three beam). Replaying the same
inputs under the same rule versions yields the identical fingerprint.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def fingerprint(parts: dict[str, Any]) -> str:
    return "fp_" + hashlib.sha256(_canonical(parts).encode("utf-8")).hexdigest()[:16]
