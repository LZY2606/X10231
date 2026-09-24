"""Offline domain case pack:
"流束换算台：ADCP 波束速度、姿态旋转与质量层级".

This is NOT a generic spreadsheet: every record follows the ADCP instrument
model.  Beam radial velocities are generated forward from known instrument-
frame currents via the Janus beam matrix, so a correct replay must recover the
truth within rounding.  All quantities keep their original unit/reference.

Timeline: 12 pings, 10 s apart; 6 depth cells (2 m bins).  Attitude samples
live on their own grid (offset 5 s) to exercise real interpolation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .geometry import janus_matrix

N_PINGS = 12
N_CELLS = 6
BIN_M = 2.0
BEAM_ANGLE_DEG = 20.0
PERIOD_S = 10.0
MAX_GAP_S = 25.0
CORR_BAD = 50          # counts below this invalidate a cell
RANGE_M_FLOOR = 13.0  # baseline acoustic bottom range
DECLINATION_DEG = 7.0
MOUNT_HEADING_DEG = 0.0


@dataclass
class SourceRecord:
    kind: str
    raw: object
    unit: str
    seq: int
    ping_idx: int | None = None
    beam: int | None = None
    cell: int | None = None
    ref: str | None = None

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "raw": self.raw,
            "unit": self.unit,
            "ref": self.ref,
            "seq": self.seq,
            "ping_idx": self.ping_idx,
            "beam": self.beam,
            "cell": self.cell,
        }


def ping_time(p: int) -> float:
    return p * PERIOD_S


def cell_depth(c: int) -> float:
    return (c + 1) * BIN_M


def true_instrument_velocity(p: int, c: int) -> np.ndarray:
    """Known truth in instrument frame: [forward, starboard, down] m/s."""
    fwd = 0.30 + 0.01 * c + 0.002 * p
    star = -0.10 + 0.004 * p - 0.003 * c
    down = 0.02 + 0.001 * c
    return np.array([fwd, star, down])


def _radial(v_inst: np.ndarray) -> list[float]:
    r = janus_matrix(BEAM_ANGLE_DEG) @ v_inst
    return [round(float(x), 4) for x in r]


def baseline_records() -> list[dict]:
    recs: list[SourceRecord] = []
    seq = 0

    def add(kind, raw, unit, p=None, b=None, c=None, ref=None):
        nonlocal seq
        recs.append(SourceRecord(kind, raw, unit, seq, p, b, c, ref))
        seq += 1

    # Transducer geometry (single configuration record, rad serialized).
    add("transducer", BEAM_ANGLE_DEG, "deg", ref="beam-geom@janus-v1")
    # Magnetic declination, east-positive, explicitly versioned.
    add("declination", DECLINATION_DEG, "deg", ref="decl@nautical-chart-2024-r1")
    # Mounting angles: heading / pitch / roll offsets of instrument vs ship.
    add("mount", MOUNT_HEADING_DEG, "deg", ref="mount@install-survey-r1")
    add("mount", 0.0, "deg", ref="mount-pitch")
    add("mount", 0.0, "deg", ref="mount-roll")

    # Independent attitude time grid: times -5, 5, ..., 115 s; heading ramps.
    n_samp = N_PINGS + 1
    for s in range(n_samp):
        t = -PERIOD_S / 2 + s * PERIOD_S
        add("sample_time", t, "s", p=s)
        add("heading", round(10.0 + 2.0 * s, 3), "deg", p=s, ref="magnetic")
        add("pitch", 1.0, "deg", p=s)
        add("roll", -0.5, "deg", p=s)

    for p in range(N_PINGS):
        for c in range(N_CELLS):
            r = _radial(true_instrument_velocity(p, c))
            for b in range(4):
                add("beam_velocity", r[b], "m/s", p=p, b=b, c=c)
                add("correlation", 180, "counts", p=p, b=b, c=c)
                add("amplitude", 110, "counts", p=p, b=b, c=c)
                add("bad_beam", 0, "flag", p=p, b=b, c=c)
        # Bottom track: 4 radial velocities (m/s) and 4 ranges (m), separate.
        for b in range(4):
            add("bottom_track", 0.0, "m/s", p=p, b=b, ref="radial")
            add("bottom_track", RANGE_M_FLOOR, "m", p=p, b=b, ref="range")
    return [r.as_dict() for r in recs]


@dataclass
class Scenario:
    key: str
    title: str
    changed: list[dict] = field(default_factory=list)
    rule: str | None = None
    accept: dict | None = None  # {node_uid: beam_combo}
    switch_pointer: bool = True
    actor: str = "hydrologist"


def _rec(kind, raw, unit, p=None, b=None, c=None, ref=None):
    return {
        "kind": kind, "raw": raw, "unit": unit, "ref": ref,
        "ping_idx": p, "beam": b, "cell": c,
    }


def scenarios() -> list[Scenario]:
    """Ordered replay scenarios; seq is assigned by the pack loader."""
    out: list[Scenario] = []

    s = Scenario("single_beam_fail", "ping 4 beam 2 hard failure")
    s.changed.append(_rec("bad_beam", 2, "flag", p=4, b=2, c=0))
    for c in range(1, N_CELLS):
        s.changed.append(_rec("bad_beam", 2, "flag", p=4, b=2, c=c))
    s.changed.append(_rec("correlation", 12, "counts", p=4, b=2, c=0))
    out.append(s)

    s = Scenario("heading_wrap_360", "heading samples cross 0/360 deg")
    for sidx, h in ((4, 357.0), (5, 3.0), (6, 6.0)):
        s.changed.append(_rec("heading", h, "deg", p=sidx, ref="magnetic"))
    out.append(s)

    s = Scenario("time_gap", "attitude samples moved, opening a 31 s gap")
    # Move the samples at ping-grid index 6 and 7 (t=55,65) to t=86,88.
    s.changed.append(_rec("sample_time", 86.0, "s", p=6))
    s.changed.append(_rec("sample_time", 88.0, "s", p=7))
    out.append(s)

    s = Scenario("three_beam_ambiguous", "ping 9 beam 3 flagged suspect -> 5 candidates")
    for c in range(N_CELLS):
        s.changed.append(_rec("bad_beam", 1, "flag", p=9, b=3, c=c))
        s.changed.append(_rec("correlation", 58, "counts", p=9, b=3, c=c))
    out.append(s)

    s = Scenario("mount_revision", "install-survey mounting heading revised 0 -> 2 deg")
    s.changed.append(
        _rec("mount", 2.0, "deg", ref="mount@install-survey-r2")
    )
    out.append(s)

    s = Scenario(
        "transform_order_legacy",
        "counter-example: heading-before-mount order (not adopted)",
        switch_pointer=False,
        actor="reviewer",
    )
    s.rule = "transform-order@heading-first-v0"
    out.append(s)

    s = Scenario("invalid_cells", "unconvertible raw, low correlation, shallow bottom")
    s.changed.append(_rec("beam_velocity", "LOST", "m/s", p=9, b=0, c=1))
    s.changed.append(_rec("correlation", 31, "counts", p=3, b=1, c=3))
    for b in range(4):
        s.changed.append(_rec("bottom_track", 9.0, "m", p=10, b=b, ref="range"))
    out.append(s)

    s = Scenario(
        "local_source_change",
        "one beam velocity nudged at ping 1 cell 2 beam 0",
        actor="hydrologist",
    )
    r = _radial(true_instrument_velocity(1, 2))
    s.changed.append(
        _rec("beam_velocity", round(r[0] + 0.05, 4), "m/s", p=1, b=0, c=2)
    )
    out.append(s)

    s = Scenario(
        "accept_three_beam_candidate",
        "reviewer adopts combo (0,1,3) three-beam solution at ping 9",
        actor="reviewer",
    )
    s.accept = {f"inst_p9_c{c}": (0, 1, 3) for c in range(N_CELLS)}
    out.append(s)

    return out


# Concurrent-edit branch scenarios (replayed last).
def concurrent_branch_a() -> Scenario:
    s = Scenario(
        "declination_revision",
        "editor A: chart declination revised 7 -> 8 deg",
        actor="editor-A",
    )
    s.changed.append(
        _rec("declination", 8.0, "deg", ref="decl@nautical-chart-2026-r2")
    )
    return s


def concurrent_branch_b() -> Scenario:
    s = Scenario(
        "independent_quality_edit",
        "editor B: ping 0 beam 1 correlation downgraded",
        actor="editor-B",
    )
    for c in range(N_CELLS):
        s.changed.append(_rec("correlation", 45, "counts", p=0, b=1, c=c))
    return s
