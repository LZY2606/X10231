"""Per ping/cell derivation: beam -> instrument -> ship -> earth.

The pipeline is deterministic and purely functional. Every candidate carries a
``fingerprint`` over its direct inputs, the beam geometry, the transform order
and the magnetic-declination / install-angle versions. Three-beam and
four-beam candidates are distinct values; ambiguous quality yields multiple
candidates rather than a single forced answer.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from .geometry import BeamGeometry, solve_four, solve_three
from .quality import BeamQuality, candidate_beam_sets, classify_beam, ERR_VELOCITY_LIMIT
from .transforms import (
    crosses_heading_wrap,
    instrument_to_ship,
    interpolate_attitude,
    ship_to_earth,
    unwrap_headings,
    wrap_360,
)


@dataclass(frozen=True)
class AttitudeSample:
    time_s: float
    heading_deg: Optional[float]
    pitch_deg: Optional[float]
    roll_deg: Optional[float]


@dataclass(frozen=True)
class AttitudeAt:
    time_s: float
    heading: float
    pitch: float
    roll: float
    status: str          # "direct" | "interpolated"
    gap: bool
    seam: bool
    diagnostic: Optional[str] = None


@dataclass
class PingCellInput:
    ping_id: str
    cell_index: int
    depth_m: Optional[float]
    valid_cell: bool
    beam_velocity: list[Optional[float]]   # normalized m/s per beam
    correlation: list[Optional[float]]
    amplitude: list[Optional[float]]
    bad_flags: list[bool]


@dataclass
class CandidateResult:
    kind: str                    # "four_beam" | "three_beam"
    beams: list[int]
    assumption: str
    instrument: list[float]
    ship: list[float]
    earth: list[float]
    error_velocity: Optional[float]
    diagnostics: list[str]
    fingerprint: str


@dataclass
class CellResult:
    ping_id: str
    cell_index: int
    depth_m: Optional[float]
    cell_status: str            # "ok" | "excluded_invalid_cell"
    qualities: list[dict]
    candidates: list[CandidateResult]
    adopted_kind: Optional[str]
    adopted_beams: Optional[list[int]]
    non_unique: bool
    diagnostics: list[str]


GAP_SECONDS = 10.0


def attitude_gap_segments(samples: list[AttitudeSample], *, gap_seconds: float = GAP_SECONDS):
    """Return (pairs_with_gap_flags, seam_count) over the ordered samples."""
    seams = 0
    gaps: list[tuple[int, int, float]] = []
    ordered = sorted(samples, key=lambda s: s.time_s)
    for i in range(1, len(ordered)):
        dt = ordered[i].time_s - ordered[i - 1].time_s
        if dt > gap_seconds:
            gaps.append((i - 1, i, dt))
        if ordered[i].heading_deg is not None and ordered[i - 1].heading_deg is not None:
            if crosses_heading_wrap(ordered[i - 1].heading_deg, ordered[i].heading_deg):
                seams += 1
    return gaps, seams


def attitude_at(
    time_s: float,
    samples: list[AttitudeSample],
    *,
    gap_seconds: float = GAP_SECONDS,
) -> AttitudeAt:
    """Resolve attitude at a ping time.

    Direct match -> ``direct``. Otherwise interpolate only inside a contiguous
    segment shorter than the gap threshold. Interpolation NEVER crosses a time
    gap; a ping stranded by a gap keeps a diagnostic and no attitude, so the
    earth step is excluded rather than invented.
    """
    ordered = sorted(samples, key=lambda s: s.time_s)
    exact = next((s for s in ordered if s.time_s == time_s), None)
    if exact is not None and None not in (exact.heading_deg, exact.pitch_deg, exact.roll_deg):
        return AttitudeAt(time_s, exact.heading_deg, exact.pitch_deg, exact.roll_deg,
                          "direct", False, False)
    # bracketing samples
    before = [s for s in ordered if s.time_s <= time_s]
    after = [s for s in ordered if s.time_s >= time_s]
    if not before or not after:
        return AttitudeAt(time_s, 0.0, 0.0, 0.0, "unresolved", True, False,
                          "attitude_outside_recorded_span")
    s0, s1 = before[-1], after[0]
    if s0 is s1:
        return AttitudeAt(time_s, s0.heading_deg, s0.pitch_deg, s0.roll_deg, "direct", False, False)
    dt = s1.time_s - s0.time_s
    seam = (s0.heading_deg is not None and s1.heading_deg is not None
            and crosses_heading_wrap(s0.heading_deg, s1.heading_deg))
    if dt > gap_seconds:
        return AttitudeAt(time_s, 0.0, 0.0, 0.0, "unresolved", True, seam,
                          f"attitude_gap_{dt:g}s_blocks_interpolation")
    if None in (s0.heading_deg, s0.pitch_deg, s0.roll_deg, s1.heading_deg, s1.pitch_deg, s1.roll_deg):
        return AttitudeAt(time_s, 0.0, 0.0, 0.0, "unresolved", True, seam,
                          "attitude_component_missing_for_interpolation")
    h = interpolate_attitude(time_s, s0.time_s, s1.time_s, s0.heading_deg, s1.heading_deg,
                             is_heading=True)
    p = interpolate_attitude(time_s, s0.time_s, s1.time_s, s0.pitch_deg, s1.pitch_deg)
    r = interpolate_attitude(time_s, s0.time_s, s1.time_s, s0.roll_deg, s1.roll_deg)
    return AttitudeAt(time_s, h, p, r, "interpolated", False, seam)


def _fp(pieces: dict) -> str:
    blob = json.dumps(pieces, sort_keys=True, separators=(",", ":"), default=str)
    return "dv_" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def derivation_fingerprint(
    *,
    beam_record_ids: list[str],
    corr_record_ids: list[str],
    amp_record_ids: list[str],
    badflag_record_ids: list[str],
    attitude_record_ids: list[str],
    geometry: BeamGeometry,
    declination_version: str,
    install_version: str,
    transform_order: str,
    kind: str,
    beams: list[int],
    assumption: str,
) -> str:
    """Stable identity of a derived value.

    Geometry, transform order and declination/install versions are all part of
    the fingerprint, so changing any of them yields a new identity rather than
    overwriting the old value.
    """
    return _fp({
        "beams_rec": beam_record_ids,
        "corr_rec": corr_record_ids,
        "amp_rec": amp_record_ids,
        "bad_rec": badflag_record_ids,
        "att_rec": attitude_record_ids,
        "geo": geometry.fingerprint(),
        "decl_v": declination_version,
        "install_v": install_version,
        "order": transform_order,
        "kind": kind,
        "beamset": beams,
        "assumption": assumption,
    })


def solve_cell(
    cell: PingCellInput,
    *,
    geometry: BeamGeometry,
    attitude: AttitudeAt,
    declination_deg: float,
    declination_version: str,
    install_angle_deg: float,
    install_version: str,
    transform_order: str = "hpr_standard",
    record_ids: Optional[dict] = None,
    valid_depth_range=(0.0, 120.0),
) -> CellResult:
    """Run the full quality -> candidates -> frame chain for one cell."""
    record_ids = record_ids or {}
    diagnostics: list[str] = []
    qualities = [
        classify_beam(
            b,
            cell.beam_velocity[b] if b < len(cell.beam_velocity) else None,
            cell.correlation[b] if b < len(cell.correlation) else None,
            cell.amplitude[b] if b < len(cell.amplitude) else None,
            cell.bad_flags[b] if b < len(cell.bad_flags) else True,
        )
        for b in range(geometry.n_beams)
    ]
    result = CellResult(
        ping_id=cell.ping_id, cell_index=cell.cell_index, depth_m=cell.depth_m,
        cell_status="ok", qualities=[asdict(q) for q in qualities],
        candidates=[], adopted_kind=None, adopted_beams=None,
        non_unique=False, diagnostics=diagnostics,
    )

    dmin, dmax = valid_depth_range
    if not cell.valid_cell:
        result.cell_status = "excluded_invalid_cell"
        diagnostics.append("cell_flagged_invalid_by_importer")
    if cell.depth_m is None or not (dmin <= cell.depth_m <= dmax):
        result.cell_status = "excluded_invalid_cell"
        diagnostics.append("cell_depth_outside_valid_range")
    if result.cell_status != "ok":
        return result

    beam_sets = candidate_beam_sets(qualities)
    if not beam_sets:
        diagnostics.append("fewer_than_three_good_beams")
        return result

    att_recs = record_ids.get("attitude", [])
    for beams in beam_sets:
        kind = "four_beam" if len(beams) == 4 else "three_beam"
        assumption = "none"
        diag: list[str] = []
        bvals = np.array([cell.beam_velocity[b] for b in beams], dtype=float)
        if kind == "four_beam":
            try:
                u, v, w, err = solve_four(geometry, cell.beam_velocity)
            except np.linalg.LinAlgError:
                diag.append("four_beam_matrix_singular")
                continue
            if abs(err) > ERR_VELOCITY_LIMIT:
                diag.append(f"error_velocity_{err:.4f}_exceeds_limit")
        else:
            missing = sorted(set(range(geometry.n_beams)) - set(beams))
            assumption = f"error_velocity_pinned_zero_missing_beam_{missing}"
            try:
                u, v, w = solve_three(geometry, beams, cell.beam_velocity)
            except np.linalg.LinAlgError:
                diag.append("three_beam_matrix_singular")
                continue
            err = None

        vel_inst = np.array([u, v, w], dtype=float)
        vel_ship = instrument_to_ship(vel_inst, install_angle_deg)
        if attitude.status == "unresolved":
            diag.append("earth_frame_unresolved:" + (attitude.diagnostic or "attitude"))
            vel_earth = [None, None, None]
        else:
            vel_earth = ship_to_earth(
                vel_ship, attitude.heading, attitude.pitch, attitude.roll,
                declination_deg, order=transform_order,
            ).tolist()

        fp = derivation_fingerprint(
            beam_record_ids=[record_ids.get("beam", [""] * geometry.n_beams)[b] for b in beams],
            corr_record_ids=[record_ids.get("correlation", [""] * geometry.n_beams)[b] for b in beams],
            amp_record_ids=[record_ids.get("amplitude", [""] * geometry.n_beams)[b] for b in beams],
            badflag_record_ids=[record_ids.get("badflag", [""] * geometry.n_beams)[b] for b in beams],
            attitude_record_ids=att_recs,
            geometry=geometry,
            declination_version=declination_version,
            install_version=install_version,
            transform_order=transform_order,
            kind=kind, beams=list(beams), assumption=assumption,
        )
        result.candidates.append(CandidateResult(
            kind=kind, beams=list(beams), assumption=assumption,
            instrument=vel_inst.tolist(), ship=vel_ship.tolist(),
            earth=vel_earth, error_velocity=err, diagnostics=diag, fingerprint=fp,
        ))

    fours = [c for c in result.candidates if c.kind == "four_beam"]
    threes = [c for c in result.candidates if c.kind == "three_beam"]
    if fours:
        result.adopted_kind = "four_beam"
        result.adopted_beams = fours[0].beams
        # three-beam alternatives stay recorded; never merged into the value
        result.non_unique = len(threes) > 0
    elif len(threes) == 1:
        result.adopted_kind = "three_beam"
        result.adopted_beams = threes[0].beams
        result.non_unique = False
    elif len(threes) > 1:
        # genuinely ambiguous: several legal three-beam values, adopt none
        result.adopted_kind = None
        result.adopted_beams = None
        result.non_unique = True
        diagnostics.append("multiple_distinct_three_beam_candidates_require_review")
    return result
