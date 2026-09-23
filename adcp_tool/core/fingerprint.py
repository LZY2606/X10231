"""Stable derived fingerprints.

The fingerprint captures *basis* only: rule/version of the transform, the
exact input values and their own fingerprints.  Identity fields (node ids,
human labels, timestamps of creation) are deliberately excluded so the same
basis always hashes identically.  Coordinate transform order, beam geometry
and magnetic-declination version are part of the basis by construction.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint(rule: str, basis: Any) -> str:
    payload = canonical({"rule": rule, "basis": basis})
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def short(fp: str, n: int = 12) -> str:
    return fp[len("sha256:"):][:n] if fp.startswith("sha256:") else fp[:n]
