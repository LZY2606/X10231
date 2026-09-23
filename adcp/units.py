"""Unit / coordinate-frame tagged quantities.

Raw imports keep their original representation (e.g. integer mm/s from the
ADCP ensemble).  Normalisation to SI happens only in the derived layer.
Values that cannot be converted stay in place and carry a diagnostic; they
are never silently dropped to make plotting easier.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# conversion factors to the canonical unit of each dimension
_TO_CANONICAL = {
    "mm/s": ("m/s", 0.001),
    "cm/s": ("m/s", 0.01),
    "m/s": ("m/s", 1.0),
    "deg": ("deg", 1.0),
    "rad": ("deg", 57.29577951308232),
    "counts": ("counts", 1.0),
    "dB": ("dB", 1.0),
    "m": ("m", 1.0),
    "cm": ("m", 0.01),
    "s": ("s", 1.0),
}

FRAMES = ("beam", "instrument", "ship", "earth")


@dataclass
class Quantity:
    """A domain quantity that always carries unit and coordinate frame."""

    value: float | None
    unit: str
    frame: str = "beam"
    raw: str | None = None            # original textual representation
    diagnostic: str | None = None     # set when the value cannot be normalised

    def to_canonical(self) -> "Quantity":
        """Return the normalised quantity, or self with a diagnostic."""
        if self.value is None:
            return Quantity(None, self.unit, self.frame, self.raw,
                            self.diagnostic or "missing-value")
        conv = _TO_CANONICAL.get(self.unit)
        if conv is None:
            # unconvertible: stay in place, attach a diagnostic
            return Quantity(self.value, self.unit, self.frame, self.raw,
                            f"unknown-unit:{self.unit}")
        canon_unit, factor = conv
        return Quantity(self.value * factor, canon_unit, self.frame, self.raw,
                        self.diagnostic)

    @property
    def convertible(self) -> bool:
        return self.unit in _TO_CANONICAL and self.value is not None


@dataclass
class CellEvidence:
    """Per depth-cell quality evidence for one ping."""

    ping: int
    cell: int
    depth_m: float
    correlation: dict[int, float] = field(default_factory=dict)  # beam -> counts
    amplitude: dict[int, float] = field(default_factory=dict)    # beam -> dB
    bad_beams: list[int] = field(default_factory=list)
    excluded: bool = False
    reasons: list[str] = field(default_factory=list)
