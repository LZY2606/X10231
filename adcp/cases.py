"""Offline domain case pack for the ADCP replay bench.

Deterministic, repository-only fixtures (no generic spreadsheets, no network).
Synthetic beam velocities are generated from a known true current through the
exact beam geometry and attitude, so the four-beam residual is ~0 while the
two legal three-beam candidates genuinely differ.

Every raw fact keeps its original representation/unit; normalization happens
only in the derived layer. Heading/pitch/roll, declination and mounting angles
are SEPARATE facts. An unconvertible depth stays in place with a diagnostic.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

from . import attitude as att
from .geometry import (DEFAULT_BEAM_ANGLE_DEG, DEFAULT_BEAM_AZIMUTH_DEG,
                       look_matrix)

PING_TIMES = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
PINGS = [f"P{i}" for i in range(len(PING_TIMES))]
CELLS = list(range(6))          # cells 0..5, nominal depths 1..6 m
ATT_TIMES = [5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0]
# A source time gap is explicitly flagged between t=25 and t=35.
GAP_AFTER = 25.0

INITIAL_BEAM_ANGLE_DEG = DEFAULT_BEAM_ANGLE_DEG
INITIAL_AZIMUTHS_DEG = DEFAULT_BEAM_AZIMUTH_DEG
INITIAL_DECLINATION_DEG = -7.0
REVISED_DECLINATION_DEG = -7.3
INITIAL_MOUNT = {"heading_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}
REVISED_MOUNT = {"heading_deg": 1.5, "pitch_deg": 0.0, "roll_deg": 0.0}
INITIAL_ORDER = "RDI_HPR"
ALTERNATE_ORDER = "RPH_ZYX"
CORR_THRESHOLD = 180
AMP_THRESHOLD_DB = 60

# True earth-frame current (m/s) by cell index: (east, north, up).
TRUE_CURRENT = {
    0: (0.30, -0.10, 0.00),
    1: (0.35, -0.12, 0.01),
    2: (0.40, -0.15, 0.02),
    3: (0.38, -0.14, 0.02),
    4: (0.30, -0.10, 0.01),
    5: (0.20, -0.05, 0.00),
}
# Ship velocity over ground (for bottom track), earth frame.
TRUE_BT_EARTH = np.array([0.50, 0.20, 0.02])

HEADINGS_DEG = [359.0, 1.0, 3.0, 200.0, 202.0, 204.0, 206.0]
PITCHES_DEG = [0.5, 0.6, 0.7, 0.4, 0.3, 0.2, 0.1]
ROLLS_DEG = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2]
BOTTOM_RANGE_M = {0: 4.2, 1: 5.5, 2: 6.5, 3: 4.8, 4: 6.0, 5: 5.2}


def sid_of(kind: str, key: str, payload: object) -> str:
    h = hashlib.sha256(
        f"{kind}|{key}|{payload!r}".encode("utf-8")).hexdigest()[:12]
    return f"sid_{kind}_{h}"


def _att_for_time(ti: int) -> dict:
    return {"heading_deg": HEADINGS_DEG[ti], "pitch_deg": PITCHES_DEG[ti],
            "roll_deg": ROLLS_DEG[ti]}


def _interp_attitude(ping_index: int) -> dict:
    # samples at ping-5s and ping+5s bracket each ping
    return att.interpolate_attitude(
        [att.AttitudeSample(
            ATT_TIMES[k], HEADINGS_DEG[k], PITCHES_DEG[k], ROLLS_DEG[k],
            gap_after=(ATT_TIMES[k] == GAP_AFTER))
         for k in range(len(ATT_TIMES))],
        PING_TIMES[ping_index])


def build_baseline() -> dict:
    """Return the raw fact store (kind -> key -> fact record)."""
    facts: dict[str, dict[str, dict]] = {}

    def put(kind, key, value, raw_unit=None, raw_repr=None, source_ref=None,
            version=1, replaces_sid=None):
        f = {"sid": sid_of(kind, key, (value, raw_unit, raw_repr, version)),
             "value": value, "raw_unit": raw_unit, "raw_repr": raw_repr,
             "source_ref": source_ref, "version": version,
             "replaces_sid": replaces_sid}
        facts.setdefault(kind, {})[key] = f
        return f

    g = look_matrix(INITIAL_BEAM_ANGLE_DEG, INITIAL_AZIMUTHS_DEG)
    put("beam_geometry", "janus4",
        {"beam_angle_deg": INITIAL_BEAM_ANGLE_DEG,
         "azimuths_deg": list(INITIAL_AZIMUTHS_DEG),
         "convention": "x=beam3->beam4,y=beam1->beam2,z=up; radial + outward"},
        source_ref="transducer_sheet_rev1.json")
    put("declination", "magnetic_declination",
        {"deg": INITIAL_DECLINATION_DEG, "model": "WMM", "model_year": 2025},
        raw_unit="deg", source_ref="WMM2025/Tokyo")
    for axis, val in INITIAL_MOUNT.items():
        put("mount", axis, {"deg": val}, raw_unit="deg",
            source_ref="installation_survey_rev1.md")
    put("transform_order", "ship_to_earth", {"order": INITIAL_ORDER},
        source_ref="processing_manual_rev1.md")
    put("thresholds", "quality",
        {"correlation_count": CORR_THRESHOLD,
         "amplitude_db": AMP_THRESHOLD_DB},
        source_ref="processing_manual_rev1.md")

    # Attitude channels as separate facts.
    for k, t in enumerate(ATT_TIMES):
        put("attitude_heading", f"att{t:g}", {"deg": HEADINGS_DEG[k],
                                              "gap_after": t == GAP_AFTER},
            raw_unit="deg", source_ref="gyro_log.csv")
        put("attitude_pitch", f"att{t:g}", {"deg": PITCHES_DEG[k],
                                            "gap_after": t == GAP_AFTER},
            raw_unit="deg", source_ref="gyro_log.csv")
        put("attitude_roll", f"att{t:g}", {"deg": ROLLS_DEG[k],
                                           "gap_after": t == GAP_AFTER},
            raw_unit="deg", source_ref="gyro_log.csv")

    for pi, p in enumerate(PINGS):
        t = PING_TIMES[pi]
        put("ping_time", p, {"time_s": t, "ensemble": pi},
            source_ref="adcp_raw.PD0")
        attitude = None
        try:
            attitude = _interp_attitude(pi)
        except att.AttitudeGap:
            attitude = None
        m_inst = att.mount_matrix(
            INITIAL_MOUNT["roll_deg"], INITIAL_MOUNT["pitch_deg"],
            INITIAL_MOUNT["heading_deg"])
        m_earth = (att.earth_matrix(
            attitude["heading_deg"], attitude["pitch_deg"],
            attitude["roll_deg"], INITIAL_DECLINATION_DEG, INITIAL_ORDER)
            if attitude else None)
        rng = BOTTOM_RANGE_M[pi]
        put("bottom_track", f"{p}:range",
            {"range_m": rng, "unit": "m", "ensemble": pi},
            raw_unit="m", source_ref="adcp_raw.PD0")

        # Bottom-track beam velocities from true ship velocity.
        if m_earth is not None:
            v_inst_bt = np.linalg.solve(
                m_earth @ m_inst, TRUE_BT_EARTH)
            bt_beams = g @ v_inst_bt
        else:
            bt_beams = np.zeros(4)
        for b in range(4):
            put("bt_velocity", f"{p}:b{b}",
                {"cm_s": round(float(bt_beams[b]) * 100.0, 2)},
                raw_unit="cm/s", source_ref="adcp_raw.PD0")

        for c in CELLS:
            key = f"{p}:c{c}"
            # Nominal cell depth (m). P3/c4 carries an unconvertible raw.
            if pi == 3 and c == 4:
                put("cell_depth", key,
                    {"m": None, "diagnostic": "raw value 'n/a' cannot be "
                                               "converted to m; kept raw"},
                    raw_unit="m", raw_repr="n/a", source_ref="adcp_raw.PD0")
            else:
                put("cell_depth", key, {"m": float(c + 1)}, raw_unit="m",
                    source_ref="adcp_raw.PD0")

            # Water-track beams from the true current (only where attitude and
            # depth are valid; otherwise emit nominal raw samples anyway so
            # every source channel exists independently).
            if m_earth is not None:
                v_inst = np.linalg.solve(
                    m_earth @ m_inst, np.array(TRUE_CURRENT[c], dtype=float))
                beams = g @ v_inst
            else:
                beams = np.zeros(4)
            for b in range(4):
                put("beam_velocity", f"{key}:b{b}",
                    {"cm_s": round(float(beams[b]) * 100.0, 2)},
                    raw_unit="cm/s", source_ref="adcp_raw.PD0")
                corr = 220 - (6 if (pi == 4 and c == 1 and b == 3) else 0)
                amp = 90 - (45 if (pi == 4 and c == 3 and b == 2) else 0)
                put("correlation", f"{key}:b{b}", {"count": corr},
                    raw_unit="count", source_ref="adcp_raw.PD0")
                put("amplitude", f"{key}:b{b}", {"db": amp},
                    raw_unit="dB", source_ref="adcp_raw.PD0")
                # Single-beam failure at P1/c2 (beam2); two-beam failure at
                # P1/c4 (beams 1 and 3).
                bad = False
                if pi == 1 and c == 2 and b == 2:
                    bad = True
                if pi == 1 and c == 4 and b in (1, 3):
                    bad = True
                put("bad_beam_flag", f"{key}:b{b}", {"bad": bad},
                    source_ref="adcp_raw.PD0")
    return facts


def case_metadata() -> dict:
    return {
        "pings": PINGS, "ping_times": PING_TIMES, "cells": CELLS,
        "attitude_times": ATT_TIMES, "gap_after": GAP_AFTER,
        "bottom_range_m": BOTTOM_RANGE_M,
        "three_beam_cell": ("P1", 2, 2),
        "two_bad_beam_cell": ("P1", 4, [1, 3]),
        "gap_ping": "P2",
        "invalid_depth_cell": ("P3", 4),
        "low_corr_cell": ("P4", 1, 3),
        "low_amp_cell": ("P4", 3, 2),
        "true_current": TRUE_CURRENT,
        "true_bt_earth": list(TRUE_BT_EARTH),
    }
