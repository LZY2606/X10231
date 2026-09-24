"""Transducer / beam geometry.

Instrument frame convention (right handed):
  +X -> instrument "starboard" axis, +Y -> instrument "forward" axis, +Z up.
Beams point downward for a down-looking ADCP (Janus convex), so a water
velocity along a beam direction produces a positive radial measurement
    r_i = d_i . v,  d_i = (sin(theta)*sin(a_i), sin(theta)*cos(a_i), -cos(theta))
with azimuth a measured from +Y (forward) toward +X (starboard).

Standard 4-beam convex Janus: a = (45, 315, 135, 225) degrees for beams
1..4, beam angle theta = 20 deg.  This geometry enters every derived
fingerprint.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class BeamGeometry:
    name: str
    beam_angle_deg: float                 # angle from instrument vertical axis
    azimuths_deg: Tuple[float, ...]       # azimuth of each beam from +Y -> +X
    convex: bool = True                   # False = concave / Janus concave

    @property
    def n_beams(self) -> int:
        return len(self.azimuths_deg)

    def directions(self) -> np.ndarray:
        """Unit direction vectors [n_beams, 3] of radial projection d_i."""
        theta = math.radians(self.beam_angle_deg)
        dirs = np.zeros((self.n_beams, 3))
        sign = 1.0 if self.convex else -1.0
        for i, az in enumerate(self.azimuths_deg):
            a = math.radians(az)
            # downward beam: -Z component
            dirs[i] = (sign * math.sin(theta) * math.sin(a),
                       sign * math.sin(theta) * math.cos(a),
                       -math.cos(theta))
        return dirs

    def geometry_fp(self) -> dict:
        return {"name": self.name, "beam_angle_deg": self.beam_angle_deg,
                "azimuths_deg": list(self.azimuths_deg), "convex": self.convex}


JANUS_CONVEX = BeamGeometry(
    name="janus_convex_rdi",
    beam_angle_deg=20.0,
    azimuths_deg=(45.0, 315.0, 135.0, 225.0),
    convex=True,
)
