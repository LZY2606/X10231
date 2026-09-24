"""Derived fingerprint.

The fingerprint binds a derivation to: transform order, beam geometry,
declination version, mounting angle version, QC thresholds and the engine
version. Any change produces a different fingerprint, so results computed
under different rules are never confused.
"""
from __future__ import annotations

import hashlib
import json

ENGINE_VERSION = "adcp-engine/1.0"


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint_rules(rules: dict) -> str:
    material = {
        "engine": ENGINE_VERSION,
        "transform_order": rules["transform_order"],
        "beam_geometry": rules["beam_geometry"],
        "beam_angle_deg": rules["beam_angle_deg"],
        "declination_version": rules["declination_version"],
        "declination_deg": rules["declination_deg"],
        "mounting_angle_deg": rules["mounting_angle_deg"],
        "qc": rules["qc"],
    }
    return hashlib.sha256(canonical(material).encode("utf-8")).hexdigest()[:16]
