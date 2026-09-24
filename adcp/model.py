"""Domain model: how raw measurements are represented before derivation.

Every raw measurement is stored *separately* and keeps its original unit
and representation:

* beam velocities          kind=``beam_velocity`` (per ping/bin/beam)
* correlation              kind=``correlation``
* amplitude                kind=``amplitude``
* raw bad-beam flags       kind=``bad_beam_flag``
* heading/pitch/roll       kind=``attitude`` (one triple per time stamp)
* magnetic declination     kind=``declination`` (versioned parameter)
* mount angles             kind=``mount_angles`` (versioned parameter)
* transducer geometry      kind=``geometry`` (versioned parameter)
* quality rules            kind=``quality_rules`` (versioned parameter)
* transform order          kind=``transform_order`` (versioned parameter)
* ping/bin geometry        kind=``profile_config``
* bottom track velocities  kind=``bottom_track`` (per ping/beam)
* bin invalid marks        kind=``bin_invalid``
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .fingerprint import fingerprint


@dataclass
class RawRecord:
    uid: str
    kind: str
    ping_index: int | None
    bin_index: int | None
    beam_index: int | None
    t: float | None
    raw_value: Any
    unit: str | None
    datum: str | None
    source: str
    note: str | None = None

    def content(self) -> dict:
        return {
            "uid": self.uid,
            "kind": self.kind,
            "ping_index": self.ping_index,
            "bin_index": self.bin_index,
            "beam_index": self.beam_index,
            "t": self.t,
            "raw_value": self.raw_value,
            "unit": self.unit,
            "datum": self.datum,
            "source": self.source,
            "note": self.note,
        }

    @property
    def record_id(self) -> str:
        return fingerprint("raw", self.content())


@dataclass
class PingBundle:
    """All raw material that belongs to one ping, assembled from records."""

    ping_index: int
    t: float
    bins: int
    radial: list[list[float | None]]          # [bin][beam] m/s
    correlation: list[list[float | None]]
    amplitude: list[list[float | None]]
    raw_bad: list[list[bool]]
    bin_invalid: list[bool]
    bin_invalid_reason: list[str | None]
    bottom_track_radial: list[float | None]
    bottom_track_corr: list[float | None]
    bottom_track_amp: list[float | None]
    bottom_track_bad: list[bool]
    diagnostics: list[str] = field(default_factory=list)
    unconvertible: list[dict] = field(default_factory=list)
