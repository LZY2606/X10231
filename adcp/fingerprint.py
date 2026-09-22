"""Derived fingerprints.

Every derived value carries a fingerprint that commits to the transform
order, the beam geometry, and the versions of the magnetic declination and
mounting-angle records used.  Changing any of them changes the fingerprint,
which is how the dependency graph decides what must be recomputed.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict


def derived_fingerprint(*, transform_order: str, beam_angle_deg: float,
                        beam_azimuths: Dict[int, float], declination_version: int,
                        mounting_angle_version: int, stage: str,
                        source_record_ids: list) -> str:
    payload: Dict[str, Any] = {
        "transform_order": transform_order,
        "beam_geometry": {
            "beam_angle_deg": beam_angle_deg,
            "beam_azimuths_deg": {str(k): v for k, v in sorted(beam_azimuths.items())},
        },
        "declination_version": declination_version,
        "mounting_angle_version": mounting_angle_version,
        "stage": stage,
        "sources": sorted(source_record_ids),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "fp-" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
