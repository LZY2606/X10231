"""Derived fingerprints.

The fingerprint binds the coordinate-transform order, the beam geometry
version, the magnetic-declination version, the installation-angle version,
the attitude window and the exact source record versions feeding a derived
node. Changing any one of them changes the fingerprint, so a viewer can
tell why a node is still valid or must be recomputed.
"""
from __future__ import annotations

import hashlib
import json


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def fingerprint(kind: str, inputs: dict, rule_versions: dict) -> str:
    payload = {"kind": kind, "inputs": inputs, "rules": rule_versions}
    digest = hashlib.sha256(_canon(payload).encode("utf-8")).hexdigest()
    return "sha256:" + digest[:16]


def rule_versions(*, beam_geometry: str, declination: str, install: str,
                  transform_order: str, quality_policy: str) -> dict:
    return {
        "beam_geometry": beam_geometry,
        "declination": declination,
        "install": install,
        "transform_order": transform_order,
        "quality_policy": quality_policy,
    }
