"""Offline ADCP evidence case pack: "beam conversion bench".

Domain-specific fixtures (not generic tables): synthetic but physically
forward-modelled beam velocities from known earth-frame truth, complete with
correlation/amplitude counts, bad-beam flags, attitude time series (including a
360 crossing and a time gap), magnetic declination, mounting angles and bottom
track ranges.
"""
from .scenarios import (
    BASELINE_ID,
    SCENARIOS,
    build_baseline,
    apply_scenario,
    run_archive,
)

__all__ = [
    "BASELINE_ID",
    "SCENARIOS",
    "build_baseline",
    "apply_scenario",
    "run_archive",
]
