"""Deterministic fingerprints for derived results.

Every derived node's fingerprint covers exactly the things that can change
its value:

* the rule/algorithm version (e.g. ``hpr-v1`` vs ``hpr-v2``);
* the beam geometry version (angle + sign convention);
* the magnetic-declination version and the mount/install-angle version;
* the quality-threshold version;
* stable identities (not row numbers) of every upstream source value;
* the inputs themselves (canonical numeric values + units/datum).

Changing coordinate-transform order, beam geometry, or the declination
version therefore produces a *different* fingerprint, never a silent in-place
overwrite.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


RULE_VERSIONS = {
    "instrument_solve": "janus-lstsq-v1",
    "ship_transform": "mount-rz-v1",
    "earth_transform": "hpr-v1",
    "earth_transform_alt": "hpr-v2",
    "attitude_interp": "gap-aware-circular-v1",
    "quality": "hierarchy-v1",
    "bottom_depth": "slant-cos-median-v1",
    "cell_placement": "faredge-blanking-v1",
}


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)


def fingerprint(rule: str, payload: dict[str, Any]) -> str:
    material = {"rule_version": rule, "payload": payload}
    digest = hashlib.sha1(_canonical(material).encode("utf-8")).hexdigest()
    return f"fp_{digest[:16]}"


def stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha1(
        _canonical(list(parts)).encode("utf-8")
    ).hexdigest()
    return f"{prefix}_{digest[:16]}"
