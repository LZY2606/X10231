"""Domain quantities: raw representation is preserved at import time;
normalization to canonical units happens only in the derived layer.
Values that cannot be converted stay in place with a diagnostic."""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional

# (from_unit, to_unit) -> multiplier
UNIT_TO_SI = {
    ("m/s", "m/s"): 1.0,
    ("cm/s", "m/s"): 0.01,
    ("mm/s", "m/s"): 0.001,
    ("deg", "deg"): 1.0,
    ("rad", "deg"): 180.0 / math.pi,
    ("count", "count"): 1.0,
    ("dB", "dB"): 1.0,
    ("s", "s"): 1.0,
}


@dataclass
class Quantity:
    raw: str                 # original representation, never rewritten
    value: Optional[float]   # normalized value in `unit`, None if unconvertible
    unit: str                # canonical unit after normalization
    frame: str               # coordinate frame / datum, e.g. beam/instrument/ship/earth/none
    diagnostic: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Quantity":
        return Quantity(d["raw"], d["value"], d["unit"], d["frame"], d.get("diagnostic"))


def parse_quantity(raw, target_unit: str, frame: str = "none") -> Quantity:
    raw_str = str(raw)
    parts = raw_str.strip().split()
    try:
        num = float(parts[0])
    except (ValueError, IndexError):
        return Quantity(raw_str, None, target_unit, frame, f"unparseable:{raw_str}")
    unit = parts[1] if len(parts) > 1 else target_unit
    key = (unit, target_unit)
    if key not in UNIT_TO_SI:
        return Quantity(raw_str, None, target_unit, frame,
                        f"unconvertible:{unit}->{target_unit}")
    return Quantity(raw_str, num * UNIT_TO_SI[key], target_unit, frame, None)
