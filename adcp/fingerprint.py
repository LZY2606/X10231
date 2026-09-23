"""Stable derived fingerprints.

A fingerprint covers the rule version and *every* input that can change a
derived value: raw record identity+value, upstream node fingerprints and
the parameter versions (coordinate transform order, beam geometry id,
magnetic-declination version, mounting-angle version).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def fingerprint(**components: Any) -> str:
    blob = json.dumps(components, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]
