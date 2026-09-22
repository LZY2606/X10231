"""Bottom track: ship-relative bottom velocity rotated into the same frames.

Bottom track is kept as a separate raw source. Earth-frame water velocity is
corrected by subtracting earth-frame ship velocity (from bottom track); the
two must go through the same attitude/mount/declination configuration.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def correct_water_velocity(water_earth: Sequence[float],
                           ship_earth: Sequence[float]) -> Dict[str, object]:
    w = np.asarray(water_earth, dtype=float)
    s = np.asarray(ship_earth, dtype=float)
    return {"velocity_earth_absolute": (w - s).tolist(),
            "ship_velocity_earth": s.tolist()}
