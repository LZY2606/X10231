"""Derived fingerprints.

Every derived value carries a fingerprint covering: rule version, transform
order, beam geometry version, magnetic declination version, mounting angle
version, processing config version and the exact input (id, version) pairs.
Two results with different fingerprints are never treated as the same value.
"""

from __future__ import annotations

import hashlib
import json


def fingerprint(stage, rule_version, transform_order, geometry_version,
                declination_version, mounting_angle_version, config_version,
                input_refs):
    payload = {
        "stage": stage,
        "rule_version": rule_version,
        "transform_order": list(transform_order),
        "geometry_version": geometry_version,
        "declination_version": declination_version,
        "mounting_angle_version": mounting_angle_version,
        "config_version": config_version,
        "inputs": sorted([f"{i}@{v}" for i, v in input_refs]),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return "fp:" + digest[:16], payload
