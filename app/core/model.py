"""Domain model: separated source records and versioned derivation config."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from .units import Quantity
from .transforms import TransducerGeometry, order_fingerprint


@dataclass
class BeamVelocity:
    id: str
    ping: str
    time_s: float
    cell: int
    depth_m: Quantity
    beams: list[Quantity]  # exactly geometry.beam_count entries
    unit: str = "m/s"


@dataclass
class Correlation:
    id: str
    ping: str
    cell: int
    beams: list[Quantity]  # counts / dimensionless
    unit: str = "counts"


@dataclass
class Amplitude:
    id: str
    ping: str
    cell: int
    beams: list[Quantity]
    unit: str = "dB"


@dataclass
class BadBeamFlag:
    id: str
    ping: str
    cell: int
    flags: list[int]  # 1 = bad / excluded, one per beam


@dataclass
class AttitudeRecord:
    id: str
    ping: str
    time_s: float
    heading: Quantity  # deg
    pitch: Quantity    # deg
    roll: Quantity     # deg


@dataclass
class Declination:
    """Magnetic declination version: magnetic -> true degrees."""

    id: str
    value_deg: float
    source: str
    valid_from: str


@dataclass
class MountAngles:
    """Installation angles version (degrees) applied instrument->ship."""

    id: str
    dx: float
    dy: float
    dz: float
    source: str
    valid_from: str


@dataclass
class BottomTrack:
    id: str
    ping: str
    time_s: float
    depth_m: Quantity
    beams: list[Quantity]  # radial bottom velocities per beam


@dataclass
class GeometryConfig:
    id: str
    beam_count: int
    theta_deg: float
    first_azimuth_deg: float
    azimuth_step_deg: float
    beam_signs: list[float]

    def geometry(self) -> TransducerGeometry:
        return TransducerGeometry(
            beam_count=self.beam_count,
            theta_deg=self.theta_deg,
            first_azimuth_deg=self.first_azimuth_deg,
            azimuth_step_deg=self.azimuth_step_deg,
            beam_signs=tuple(self.beam_signs),
        )


@dataclass
class RuleConfig:
    """Versioned rule set. Any change here changes derived fingerprints."""

    transform_order: str = "HPR_RDI"
    geom_id: str = "geom-standard"
    declination_id: str = ""
    mount_id: str = ""
    correlation_min_counts: float = 64.0
    amplitude_min_db: float = 40.0
    max_attitude_gap_s: float = 5.0
    w_ambiguity_limit_mps: float = 0.05

    def rule_fingerprint(self, geom: TransducerGeometry, order: str) -> str:
        return "|".join([
            geom.fingerprint(),
            order_fingerprint(order),
            f"decl={self.declination_id}",
            f"mount={self.mount_id}",
            f"corr_min={self.correlation_min_counts}",
            f"amp_min={self.amplitude_min_db}",
            f"gap={self.max_attitude_gap_s}",
            f"wlim={self.w_ambiguity_limit_mps}",
        ])


def record_kind(record: Any) -> str:
    return type(record).__name__
