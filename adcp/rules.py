"""Versioned processing rules and deterministic fingerprints.

Anything that can change a derived value has a version and participates in a
fingerprint: quality thresholds, beam geometry, rotation order, magnetic
declination and the rule-set revision itself.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from .geometry import TransducerGeometry, rdi_4_beam
from .quality import QualityThresholds


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(prefix: str, payload) -> str:
    return f"{prefix}_{sha256_text(canonical(payload))[:16]}"


def geometry_fingerprint(geo: TransducerGeometry) -> str:
    return fingerprint("geom", {
        "name": geo.name,
        "azimuths_deg": list(geo.azimuths_deg),
        "beam_angle_deg": geo.beam_angle_deg,
        "convex": geo.convex,
    })


@dataclass(frozen=True)
class RuleSet:
    version: str
    rotation_order: str = "HPR"
    declination_deg: float = 0.0
    mount_heading_deg: float = 0.0
    mount_pitch_deg: float = 0.0
    mount_roll_deg: float = 0.0
    max_attitude_gap_s: float = 2.0
    thresholds: QualityThresholds = field(default_factory=QualityThresholds)
    geometry: TransducerGeometry = field(default_factory=rdi_4_beam)

    def rule_fingerprint(self) -> str:
        return fingerprint("rules", {
            "version": self.version,
            "rotation_order": self.rotation_order,
            "declination_deg": self.declination_deg,
            "mount_heading_deg": self.mount_heading_deg,
            "mount_pitch_deg": self.mount_pitch_deg,
            "mount_roll_deg": self.mount_roll_deg,
            "max_attitude_gap_s": self.max_attitude_gap_s,
            "thresholds": self.thresholds.fingerprint_dict(),
            "geometry": geometry_fingerprint(self.geometry_key()),
        })

    def geometry_key(self) -> TransducerGeometry:
        return self.geometry
