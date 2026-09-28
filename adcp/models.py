"""Domain records for ADCP evidence.

Beam measurements, correlation, amplitude, bad-beam flags, attitude, bottom
track, declination and mount/install configuration are deliberately separate
record types with separate streams. Each record has a stable stream identity
(ping index, cell index, beam index / timestamp) and keeps its raw import
representation.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, asdict
from typing import Any

from .units import Quantity


class TransformOrder(str, enum.Enum):
    """Heading/pitch/roll application order, part of the derivation fingerprint.

    The string lists the rotation order applied to a level-frame vector to get
    a ship-frame vector (first-applied -> last-applied).
    """

    RPH = "RPH"  # roll, then pitch, then heading (heading applied last)
    HPR = "HPR"  # heading, then pitch, then roll (roll applied last)


@dataclass
class TransducerConfig:
    """Beam geometry. Beam unit vectors are derived from this configuration."""

    beam_angle_deg: float  # angle from vertical
    beam_azimuths_deg: list[float]  # azimuth of each beam, measured from +x toward +y
    convex: bool = True  # convex Janus: beams splay outward; concave points inward
    n_beams: int = 4
    frequency_hz: Quantity | None = None
    config_id: str = "geom-v1"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.frequency_hz is not None:
            data["frequency_hz"] = self.frequency_hz.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransducerConfig":
        data = dict(data)
        freq = data.get("frequency_hz")
        if isinstance(freq, dict):
            data["frequency_hz"] = Quantity.from_dict(freq)
        data["beam_azimuths_deg"] = list(data["beam_azimuths_deg"])
        return cls(**data)


@dataclass
class CellConfig:
    """Depth-cell geometry, in metres."""

    bin_size_m: float
    blank_transmit_m: float
    n_cells: int
    cell1_mid_m: float

    def bin_center(self, cell: int) -> float:
        return self.cell1_mid_m + cell * self.bin_size_m


@dataclass
class BeamVelocityRecord:
    ping: int
    cell: int
    beam: int
    velocity: Quantity  # canonical m/s; may be unconvertible


@dataclass
class CorrelationRecord:
    ping: int
    cell: int
    beam: int
    correlation: Quantity  # counts 0..255


@dataclass
class AmplitudeRecord:
    ping: int
    cell: int
    beam: int
    amplitude: Quantity  # counts


@dataclass
class BadBeamFlagRecord:
    ping: int
    cell: int
    beam: int
    flagged: bool
    reason: str = ""


@dataclass
class AttitudeRecord:
    """One heading/pitch/roll sample at a timestamp (epoch seconds)."""

    timestamp_s: float
    heading: Quantity  # deg, magnetic as recorded
    pitch: Quantity  # deg
    roll: Quantity  # deg


@dataclass
class BottomTrackRecord:
    ping: int
    beam: int
    range: Quantity  # m
    velocity: Quantity  # m/s along-beam bottom-track velocity


@dataclass
class DeclinationRecord:
    """Magnetic declination version. Effective over [valid_from_s, valid_to_s)."""

    declination_id: str
    declination_deg: float
    valid_from_s: float | None = None
    valid_to_s: float | None = None
    source: str = ""


@dataclass
class MountRecord:
    """Installation/alignment angles, versioned independently from declination."""

    mount_id: str
    heading_bias_deg: float = 0.0  # installation angle added to sensor heading
    alignment_pitch_deg: float = 0.0  # instrument -> ship alignment
    alignment_roll_deg: float = 0.0
    valid_from_s: float | None = None
    valid_to_s: float | None = None
    note: str = ""


@dataclass
class PingMeta:
    ping: int
    timestamp_s: float
    ensemble_number: int | None = None


# ---------------------------------------------------------------------------
# Derived layer
# ---------------------------------------------------------------------------


@dataclass
class BeamQuality:
    ping: int
    cell: int
    beam: int
    level: str  # good | marginal | bad | excluded
    reasons: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.level in ("good", "marginal")

    @property
    def hard_bad(self) -> bool:
        return self.level in ("bad", "excluded")


@dataclass
class Vector3:
    x: float
    y: float
    z: float
    frame: str  # inst | ship | earth
    units: str = "m/s"

    def as_array(self):
        import numpy as np

        return np.array([self.x, self.y, self.z], dtype=float)


@dataclass
class VelocityDerivation:
    """One derived velocity at a stage for ping/cell.

    Solutions are kept distinct by ``n_beams``: a 3-beam solution never shares
    an identity with a 4-beam solution.
    """

    ping: int
    cell: int
    stage: str  # instrument | ship | earth
    n_beams: int
    velocity: Vector3 | None
    error_velocity_mps: float | None
    quality_levels: dict[int, str]
    excluded_beams: list[int]
    candidate_id: str | None  # which omitted-beam candidate this is
    fingerprint: str
    status: str  # ok | ambiguous | no_solution | excluded
    diagnostics: list[str] = field(default_factory=list)
    rule_versions: dict[str, str] = field(default_factory=dict)


@dataclass
class AttitudeAtPing:
    ping: int
    timestamp_s: float
    heading_deg: float
    pitch_deg: float
    roll_deg: float
    bracketing_indices: tuple[int | None, int | None]
    status: str  # interpolated | bracketed | gap | outside | unconvertible
    diagnostics: list[str] = field(default_factory=list)
