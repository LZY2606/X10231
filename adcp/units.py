"""Units and coordinate datums.

A :class:`Quantity` keeps both the raw textual representation (as imported) and
a normalized SI value. Values that cannot be converted keep ``normalized is
None`` and carry a diagnostic; they are never dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Linear conversion: canonical = value * slope + offset
# angle is handled specially because of wrap semantics.
_TO_CANONICAL: dict[tuple[str, str], float] = {
    ("velocity", "mm/s"): 0.001,
    ("velocity", "cm/s"): 0.01,
    ("velocity", "m/s"): 1.0,
    ("length", "mm"): 0.001,
    ("length", "cm"): 0.01,
    ("length", "m"): 1.0,
    ("time", "s"): 1.0,
    ("time", "ms"): 0.001,
    ("angle", "deg"): 1.0,
    ("angle", "rad"): 180.0 / 3.141592653589793,
    ("angle", "degrees_true"): 1.0,
    ("angle", "degrees_mag"): 1.0,
    ("dimensionless", "count"): 1.0,
    ("dimensionless", "dB"): 1.0,
}

CANONICAL_UNIT = {
    "velocity": "m/s",
    "length": "m",
    "time": "s",
    "angle": "deg",
    "dimensionless": "count",
}


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True)
class Quantity:
    """A measured value plus its unit/datum, retaining raw representation."""

    value: Optional[float]
    unit: str
    dimension: str
    raw: str
    normalized: Optional[float] = None
    diagnostic: Optional[Diagnostic] = None

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "unit": self.unit,
            "dimension": self.dimension,
            "raw": self.raw,
            "normalized": self.normalized,
            "diagnostic": self.diagnostic.to_dict() if self.diagnostic else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Quantity":
        diag = d.get("diagnostic")
        return cls(
            value=d.get("value"),
            unit=d["unit"],
            dimension=d["dimension"],
            raw=d.get("raw", ""),
            normalized=d.get("normalized"),
            diagnostic=Diagnostic(**diag) if diag else None,
        )


def parse_quantity(
    raw: str | float | int | None,
    unit: str,
    dimension: str,
    *,
    allow_missing: bool = True,
) -> Quantity:
    """Parse and normalize a raw value in ``unit`` of ``dimension``.

    Unparseable or missing values stay at the origin with a diagnostic instead
    of raising, so the derived layer can record why they were excluded.
    """
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        if allow_missing:
            return Quantity(
                value=None,
                unit=unit,
                dimension=dimension,
                raw="" if raw is None else str(raw),
                normalized=None,
                diagnostic=Diagnostic("missing_value", f"empty {dimension} in {unit}"),
            )
        return Quantity(
            value=None,
            unit=unit,
            dimension=dimension,
            raw="" if raw is None else str(raw),
            normalized=None,
            diagnostic=Diagnostic("missing_value", f"required {dimension} missing"),
        )
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return Quantity(
            value=None,
            unit=unit,
            dimension=dimension,
            raw=str(raw),
            normalized=None,
            diagnostic=Diagnostic("unparseable_value", f"cannot parse {raw!r} as {dimension}"),
        )
    slope = _TO_CANONICAL.get((dimension, unit))
    if slope is None:
        return Quantity(
            value=value,
            unit=unit,
            dimension=dimension,
            raw=str(raw),
            normalized=None,
            diagnostic=Diagnostic(
                "unknown_unit", f"no normalization for {dimension}/{unit}"
            ),
        )
    if value != value:  # NaN guard
        return Quantity(
            value=value,
            unit=unit,
            dimension=dimension,
            raw=str(raw),
            normalized=None,
            diagnostic=Diagnostic("nan_value", f"NaN {dimension}"),
        )
    return Quantity(
        value=value,
        unit=unit,
        dimension=dimension,
        raw=str(raw),
        normalized=value * slope,
    )


def convert(q: Quantity, target_unit: str) -> float:
    """Convert an already normalized quantity to ``target_unit``.

    Raises ``ValueError`` when the conversion is impossible; callers at the
    derived layer convert that into a diagnostic instead of dropping data.
    """
    if q.normalized is None:
        raise ValueError(f"quantity has no normalized value: {q.diagnostic}")
    if target_unit == CANONICAL_UNIT[q.dimension]:
        return q.normalized
    slope = _TO_CANONICAL.get((q.dimension, target_unit))
    if slope is None or slope == 0:
        raise ValueError(f"cannot convert {q.unit} to {target_unit}")
    return q.normalized / slope


# Coordinate datums (not numeric "units"; they label reference frames).
DATUMS = {
    "BEAM": "beam frame, one axis per transducer",
    "INSTRUMENT": "instrument frame X=beam1/starboard side, Y=forward, Z=up",
    "SHIP": "ship frame after mounting angle correction",
    "EARTH_MAG": "earth frame relative to magnetic north",
    "EARTH_TRUE": "ENU earth frame relative to true north",
}
