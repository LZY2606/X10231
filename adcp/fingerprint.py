"""Derived fingerprints.

Every derived value carries a fingerprint covering the transform order, beam
geometry version, magnetic declination version and mounting-angle version, so
a rule/config change is always visible in the derived identity.
"""
from __future__ import annotations

import hashlib
import json


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def short_hash(obj) -> str:
    return hashlib.sha1(canonical(obj).encode()).hexdigest()[:16]


def pipeline_fingerprint(transform_order: str, geometry_version: str,
                         declination_version: str,
                         mounting_angle_version: str) -> str:
    return short_hash({
        "transform_order": transform_order,
        "beam_geometry_version": geometry_version,
        "magnetic_declination_version": declination_version,
        "mounting_angle_version": mounting_angle_version,
    })
