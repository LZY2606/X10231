"""Quantities, units and coordinate datums.

Every domain value keeps the text that was imported (``raw``), the unit and
datum it arrived with, plus an optional canonical value produced only in the
derivation layer. A value that cannot be converted stays in place with a
diagnostic instead of being dropped for plotting convenience.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


class ConversionError(ValueError):
    """Raised when an imported value cannot be normalized to its SI unit."""


# Linear multiplier -> canonical SI unit.
UNIT_SCALE: dict[tuple[str, str], float] = {
    ("velocity", "m/s"): 1.0,
    ("velocity", "cm/s"): 0.01,
    ("velocity", "mm/s"): 0.001,
    ("velocity", "knot"): 0.5144444444444445,
    ("angle", "deg"): 1.0,
    ("angle", "rad"): 180.0 / math.pi,
    ("angle", "gon"): 0.9,
    ("depth", "m"): 1.0,
    ("depth", "cm"): 0.01,
    ("time", "s"): 1.0,
    ("time", "ms"): 0.001,
    ("count", "count"): 1.0,
    ("correlation", "count"): 1.0,
    ("amplitude", "count"): 1.0,
    ("magnetic_declination", "deg"): 1.0,
}

# Frame / datum vocabulary. These are never silently remapped.
BEAM_FRAME = "beam"                      # individual beam radial axis
INSTRUMENT_FRAME = "instrument_xyz"      # right handed: x=fwd,y=stbd,z=down
SHIP_FRAME = "ship_fsu"                  # forward, starboard, up (EN-aligned yaw)
EARTH_ENU = "earth_enu"                  # east, north, up
FRD_FRAME = "ship_frd"                   # forward, starboard, down (rot step)
MAGNETIC_HEADING = "heading_magnetic_deg"
TRUE_HEADING = "heading_true_deg"


@dataclass
class Quantity:
    """An imported measurement with provenance of its representation."""

    raw: str
    kind: str
    unit: str
    value: Optional[float] = None
    canonical: Optional[float] = None
    canonical_unit: Optional[str] = None
    datum: Optional[str] = None
    diagnostic: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.diagnostic is None and self.canonical is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Quantity":
        return cls(**data)


def canonical_unit(kind: str) -> str:
    return {
        "velocity": "m/s",
        "angle": "deg",
        "depth": "m",
        "time": "s",
        "count": "count",
        "correlation": "count",
        "amplitude": "count",
        "magnetic_declination": "deg",
    }[kind]


def parse_quantity(
    raw: Any,
    kind: str,
    unit: str,
    datum: Optional[str] = None,
    bounds: Optional[tuple[float, float]] = None,
) -> Quantity:
    """Parse an imported cell without losing the original text.

    ``raw`` is preserved verbatim. Missing markers (``-``, ``NA``, blank) are
    kept as-is without a canonical value, flagged ``missing`` rather than zero.
    """
    text = "" if raw is None else str(raw).strip()
    q = Quantity(raw=text, kind=kind, unit=unit, datum=datum)
    if text == "" or text.upper() in {"NA", "N/A", "NAN", "NULL", "-", "--"}:
        q.diagnostic = "missing"
        return q
    try:
        q.value = float(text)
    except ValueError:
        q.diagnostic = f"unparseable:{text!r}"
        return q
    key = (kind, unit)
    if key not in UNIT_SCALE:
        q.diagnostic = f"unknown_unit:{kind}/{unit}"
        return q
    if not math.isfinite(q.value):
        q.diagnostic = "non_finite"
        return q
    q.canonical = q.value * UNIT_SCALE[key]
    q.canonical_unit = canonical_unit(kind)
    if bounds is not None and not (bounds[0] <= q.canonical <= bounds[1]):
        q.diagnostic = f"out_of_bounds:{q.canonical}{q.canonical_unit}"
    return q


def unwrap_angle_deg(series: list[Optional[float]]) -> list[Optional[float]]:
    """Unwrap a heading series in degrees, preserving ``None`` as a gap.

    Gaps split the series: unwrapping never interpolates across them.
    """
    out: list[Optional[float]] = []
    last: Optional[float] = None
    for value in series:
        if value is None:
            out.append(None)
            last = None
            continue
        if last is None:
            out.append(float(value))
        else:
            delta = (value - last + 180.0) % 360.0 - 180.0
            out.append(last + delta)
        last = out[-1]
    return out


def normalize_heading_deg(value: float) -> float:
    """Map any unwrapped heading onto [0, 360) so 359.5deg -> 0.5deg is local."""
    return value % 360.0


def interpolate_gap_aware(
    times: list[float],
    values: list[Optional[float]],
    query_times: list[float],
    max_gap_s: Optional[float] = None,
) -> tuple[list[Optional[float]], list[Optional[str]]]:
    """Linear interpolation that never crosses a missing sample or a gap.

    Returns interpolated values and per-query diagnostics.
    """
    known = [(t, v) for t, v in zip(times, values) if v is not None]
    out: list[Optional[float]] = []
    diag: list[Optional[str]] = []
    for qt in query_times:
        before = [pair for pair in known if pair[0] <= qt]
        after = [pair for pair in known if pair[0] >= qt]
        if before and after and before[-1][0] == qt:
            out.append(before[-1][1]); diag.append(None); continue
        if not before or not after:
            out.append(None); diag.append("attitude_outside_support"); continue
        t0, v0 = before[-1]
        t1, v1 = after[0]
        span = t1 - t0
        if max_gap_s is not None and span > max_gap_s:
            out.append(None)
            diag.append(f"attitude_gap:{span:.3f}s>{max_gap_s:g}s")
            continue
        frac = 0.0 if span == 0 else (qt - t0) / span
        out.append(v0 + (v1 - v0) * frac)
        diag.append(None)
    return out, diag
