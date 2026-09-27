"""Units and quantities.

All domain quantities carry a unit or datum.  Imported source records keep
their original representation untouched; conversion only happens in the
derivation layer.  A value that cannot be converted stays at its original
location with a diagnostic instead of being silently dropped.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# linear conversions to SI base unit
LINEAR = {
    "m/s": (1.0, "m s^-1"),
    "mm/s": (0.001, "m s^-1"),
    "cm/s": (0.01, "m s^-1"),
    "m": (1.0, "m"),
    "cm": (0.01, "m"),
    "mm": (0.001, "m"),
    "m per bin": (1.0, "m"),
    "count": (1.0, "count"),
    "%": (1.0, "percent"),
    "dB": (1.0, "dB"),
    "s": (1.0, "s"),
}


class UnitError(ValueError):
    pass


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str
    datum: str | None = None  # coordinate/reference datum when relevant

    def to_si(self) -> "Quantity":
        if self.unit not in LINEAR:
            raise UnitError(f"no conversion registered for unit {self.unit!r}")
        factor, si_unit = LINEAR[self.unit]
        return Quantity(float(self.value) * factor, si_unit, self.datum)

    def to(self, unit: str) -> "Quantity":
        if unit not in LINEAR:
            raise UnitError(f"unknown unit {unit!r}")
        f_src, dim_src = LINEAR[self.unit]
        f_dst, dim_dst = LINEAR[unit]
        if dim_src != dim_dst:
            raise UnitError(f"cannot convert {self.unit!r} ({dim_src}) to {unit!r} ({dim_dst})")
        return Quantity(float(self.value) * f_src / f_dst, unit, self.datum)

    def to_dict(self) -> dict:
        out = {"value": self.value, "unit": self.unit}
        if self.datum:
            out["datum"] = self.datum
        return dict(out)


def convert(value: float, unit: str, target: str) -> float:
    return Quantity(value, unit).to(target).value


# --- angles --------------------------------------------------------------
def normalize_degrees(angle: float) -> float:
    """Wrap any angle to [0, 360)."""
    return angle % 360.0


def deg_to_rad(angle_deg: float) -> float:
    return angle_deg * math.pi / 180.0


def rad_to_deg(angle_rad: float) -> float:
    return angle_rad * 180.0 / math.pi


def unwrap_headings_deg(values: list[float]) -> list[float]:
    """Unwrap a time-ordered heading sequence so 359 -> 1 becomes 361."""
    out: list[float] = []
    last: float | None = None
    for value in values:
        if last is None:
            out.append(float(value))
        else:
            step = (float(value) - last + 180.0) % 360.0 - 180.0
            out.append(last + step)
        last = out[-1]
    return out
