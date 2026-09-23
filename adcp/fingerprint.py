"""Content-addressed derivation fingerprints.

A fingerprint covers every input the derived value depends on:

* exact versions of the upstream source records (beam geometry, the beam
  velocities used, bad-beam flags, correlation/amplitude gates, attitude,
  magnetic declination and transducer install angles — all separate);
* coordinate transform order, beam geometry rule version and the exact
  magnetic-declination version;
* quality gates and the code rule versions.

Two derivations with identical fingerprints are the same computation and
must reuse the stored value; any change in any listed input yields a new
fingerprint.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from . import geometry, attitude, transforms, quality


def rule_versions() -> dict[str, str]:
    return {
        "geometry": geometry.GEOMETRY_RULE_VERSION,
        "attitude": attitude.ATTITUDE_RULE_VERSION,
        "transforms": transforms.TRANSFORM_RULE_VERSION,
        "quality": quality.QUALITY_RULE_VERSION,
    }


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)


def fingerprint(payload: dict[str, Any]) -> str:
    body = {"rules": rule_versions(), "payload": payload}
    digest = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    return "fp_" + digest[:16]
