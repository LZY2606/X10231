"""Domain quantities carry a unit and a coordinate/reference frame.

Import keeps the original textual representation untouched.  Only the
derived layer normalizes units (e.g. mm/s -> m/s).  A value that cannot be
converted stays in place with a diagnostic: it is never dropped just to
keep a plot renderable.
"""
from __future__ import annotations

from typing import Optional, Tuple

# velocity factors to the SI derived unit m/s
VELOCITY_TO_MS = {"mm/s": 0.001, "m/s": 1.0}

KNOWN_UNITS = {
    "mm/s",        # beam velocity as logged by the ADCP
    "m/s",         # normalized velocity
    "deg",         # angles / heading (frame says magnetic vs true)
    "counts",      # amplitude / echo intensity
    "%",           # correlation
    "s",           # time
    "bool",        # bad-beam flags (true/false)
    "json",        # transducer configuration record
}


def parse_numeric(value_raw: object) -> Optional[float]:
    try:
        return float(value_raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def parse_bool(value_raw: object) -> Optional[float]:
    if isinstance(value_raw, bool):
        return 1.0 if value_raw else 0.0
    text = str(value_raw).strip().lower()
    if text in ("true", "1", "bad", "yes"):
        return 1.0
    if text in ("false", "0", "good", "no"):
        return 0.0
    return None


def normalize(value_raw, unit: str) -> Tuple[Optional[float], str, Optional[str]]:
    """Return (numeric, status, diagnostic) for a raw value/unit pair."""
    if unit == "bool":
        numeric = parse_bool(value_raw)
        if numeric is None:
            return None, "unconverted", f"value {value_raw!r} is not a boolean; kept raw"
        return numeric, "ok", None
    if unit == "json":
        return None, "ok", None
    numeric = parse_numeric(value_raw)
    if unit not in KNOWN_UNITS:
        return (
            None,
            "unconverted",
            f"unknown unit {unit!r}; raw value {value_raw!r} kept in place",
        )
    if numeric is None:
        return (
            None,
            "unconverted",
            f"value {value_raw!r} is not numeric; raw value kept in place",
        )
    return numeric, "ok", None


def velocity_to_ms(value: float, unit: str) -> float:
    factor = VELOCITY_TO_MS.get(unit)
    if factor is None or value is None:
        raise ValueError(f"cannot convert velocity unit {unit!r}")
    return float(value) * factor
