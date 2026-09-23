"""Stable identities and the machine-readable principle catalogue."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from . import RULE_VERSION


def canonical_json(obj: Any) -> str:
    """Deterministic JSON used for hashing and export byte stability."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_id(prefix: str, payload: Any) -> str:
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:16]}"


def derivation_fingerprint(
    *,
    slot: str,
    rule_version: str,
    transform_order: str,
    geometry: dict,
    thresholds: dict,
    declination_version: str,
    mount_version: str,
    inputs: dict,
) -> str:
    """Fingerprint every rule choice that can change a derived value."""
    return content_id(
        "fp",
        {
            "rule_version": rule_version,
            "slot": slot,
            "transform_order": transform_order,
            "beam_geometry": geometry,
            "thresholds": thresholds,
            "declination_version": declination_version,
            "mount_version": mount_version,
            "inputs": inputs,
        },
    )


# Principles emitted into machine-readable events so the UI and tests can both
# assert why a verdict was reached.
PRINCIPLES: dict[str, str] = {
    "P-3v4-separate": "Three-beam and four-beam solutions occupy distinct slots and are never reported as one value.",
    "P-no-cross-gap": "Attitude interpolation does not cross a missing sample or a declared time gap.",
    "P-order-fingerprinted": "Coordinate transform order is part of the derivation fingerprint.",
    "P-geometry-fingerprinted": "Transducer beam geometry is part of the derivation fingerprint.",
    "P-decl-versioned": "Magnetic declination is versioned and its version is part of the fingerprint.",
    "P-mount-versioned": "Installation angles are versioned separately from transducer geometry.",
    "P-multiple-candidates": "When no solution is unique, every legal candidate is retained.",
    "P-cell-excluded": "Invalid or sidelobe-contaminated depth cells are excluded with an explicit reason.",
    "P-history-immutable": "Replaying an old event restores its dependency versions, never overwrites history.",
    "P-atomic-pointer": "A commit pointer moves only after all derived outputs are written.",
    "P-raw-preserved": "Imported values keep original representation, unit and datum; unconvertible values stay in place.",
}


def event(
    event_type: str,
    subject: str,
    *,
    principles: list[str] | None = None,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "type": event_type,
        "subject": subject,
        "principles": principles or [],
        "detail": detail or {},
        "rule_version": RULE_VERSION,
    }
