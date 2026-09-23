"""Derived fingerprints.

Every derived value carries a fingerprint that commits to:
  * the rule version,
  * the coordinate transform order (e.g. HPR vs RPH),
  * the beam geometry,
  * the magnetic-declination value *and* its model version,
  * the mounting angle,
  * the identities of the source records it was derived from.

Changing any of these inputs yields a different fingerprint, which is how
the dependency graph decides what is still valid and what must recompute.
"""
from __future__ import annotations

import hashlib
import json


def derived_fingerprint(*, rule_version: str, transform_order: str,
                        beam_geometry: dict, declination_deg: float,
                        declination_version: str, mounting_angle_deg: float,
                        source_ids: list[str], stage: str) -> str:
    payload = {
        "rule_version": rule_version,
        "transform_order": transform_order,
        "beam_geometry": beam_geometry,
        "declination_deg": declination_deg,
        "declination_version": declination_version,
        "mounting_angle_deg": mounting_angle_deg,
        "source_ids": sorted(source_ids),
        "stage": stage,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
