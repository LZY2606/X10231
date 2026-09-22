"""Quantity parsing: every domain value carries units or a coordinate datum.

Importers keep the raw representation.  Only the derivation layer normalizes.
A value that cannot be converted stays in place with a diagnostic instead of
being dropped to make a plot possible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import math

# canonical (SI-ish) units used after normalization
VELOCITY = "m/s"
ANGLE = "deg"
TIME = "s"
LENGTH = "m"
TIME_UNIX = "unix_s"

_DIM_ALIASES = {
    "velocity": {"m/s": 1.0, "m s-1": 1.0, "cm/s": 0.01, "mm/s": 0.001,
                 "knot": 0.514444444, "kn": 0.514444444, "msec": 1.0},
    "angle": {"deg": 1.0, "degree": 1.0, "degrees": 1.0, "rad": 180.0 / math.pi,
              "radian": 180.0 / math.pi},
    "length": {"m": 1.0, "meter": 1.0, "meters": 1.0, "cm": 0.01,
               "mm": 0.001, "ft": 0.3048},
    "time": {"s": 1.0, "sec": 1.0, "second": 1.0, "ms": 0.001,
             "unix_s": 1.0, "iso8601": None},
}

_ALIAS = {}
for dim, units in _DIM_ALIASES.items():
    for u in units:
        _ALIAS[u.lower()] = dim


@dataclass
class Quantity:
    raw: str                      # original representation, always retained
    unit: str                     # original unit / datum label
    dim: Optional[str] = None     # recognized dimension, else None
    value: Optional[float] = None
    valid: bool = False
    diagnostic: Optional[str] = None
    canonical_unit: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "raw": self.raw, "unit": self.unit, "dim": self.dim,
            "value": self.value, "valid": self.valid,
            "diagnostic": self.diagnostic, "canonical_unit": self.canonical_unit,
        }


def _parse_iso(raw: str) -> Optional[float]:
    """Parse a small subset of ISO-8601 timestamps to unix seconds (UTC)."""
    from datetime import datetime, timezone
    text = raw.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def parse(raw, unit: str, dim: Optional[str] = None,
          valid_range: Optional[Tuple[float, float]] = None) -> Quantity:
    """Parse a raw cell preserving its representation.

    `dim` may be forced by the importer (e.g. a beam radial channel); when
    omitted it is derived from the unit string.  Unconvertible values return
    an invalid Quantity with a diagnostic and are never silently dropped.
    """
    raw_s = "" if raw is None else str(raw)
    u = (unit or "").strip()
    inferred_dim = dim
    if inferred_dim is None:
        inferred_dim = _ALIAS.get(u.lower())
    q = Quantity(raw=raw_s, unit=u, dim=inferred_dim)
    if inferred_dim is None:
        q.diagnostic = "unknown_unit:%s" % u
        return q
    table = _DIM_ALIASES[inferred_dim]
    factor = table.get(u.lower())
    if factor is None:
        q.diagnostic = "unsupported_unit:%s" % u
        return q
    text = raw_s.strip()
    if text == "" or text.lower() in ("nan", "na", "n/a", "null", "--"):
        q.diagnostic = "missing_value"
        return q
    try:
        if u.lower() == "iso8601":
            parsed = _parse_iso(text)
            if parsed is None:
                raise ValueError("bad iso8601")
        else:
            parsed = float(text)
    except (ValueError, OverflowError):
        q.diagnostic = "unparseable_value:%s" % raw_s
        return q
    if not math.isfinite(parsed):
        q.diagnostic = "non_finite"
        return q
    if valid_range is not None and not (valid_range[0] <= parsed <= valid_range[1]):
        q.diagnostic = "out_of_range:%g..%g" % valid_range
        return q
    if factor is not None:
        value = parsed * factor
    else:
        value = parsed
    q.value = value
    q.valid = True
    canonical = {
        "velocity": VELOCITY, "angle": ANGLE,
        "length": LENGTH, "time": TIME if u.lower() != "unix_s" else TIME_UNIX,
    }[inferred_dim]
    q.canonical_unit = canonical
    return q
