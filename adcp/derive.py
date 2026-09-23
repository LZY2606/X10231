"""Derivation orchestration: raw sources -> beam/instrument/ship/earth values.

Each output carries:

* a stable ``value_id`` (case/ping/cell/beam-independent ordinal key),
* a content ``fingerprint`` that binds the exact upstream source versions,
  beam geometry, transform order and magnetic-declination version,
* a quality tier and, when excluded, a reason,
* every legal 3-beam candidate (never collapsed into one value).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import attitude as att
from . import geometry as geom
from . import quality as qmod
from . import transforms as tf
from .fingerprint import fingerprint, rule_versions
from .units import normalize


@dataclass
class SourceRefs:
    """Exact version pins of the eight source categories."""

    transducer: tuple[str, str]
    install: tuple[str, str]
    declination: tuple[str, str]
    settings: tuple[str, str]
    beam: dict[tuple[int, int, int], tuple[str, str]] = field(default_factory=dict)
    correlation: dict[tuple[int, int, int], tuple[str, str]] = field(default_factory=dict)
    amplitude: dict[tuple[int, int, int], tuple[str, str]] = field(default_factory=dict)
    badflag: dict[tuple[int, int, int], tuple[str, str]] = field(default_factory=dict)
    attitude: dict[int, tuple[str, str]] = field(default_factory=dict)

    def pins(self) -> dict[str, Any]:
        return {
            "transducer": self.transducer,
            "install": self.install,
            "declination": self.declination,
            "settings": self.settings,
            "beam": sorted([[list(k), list(v)] for k, v in self.beam.items()]),
            "correlation": sorted([[list(k), list(v)] for k, v in self.correlation.items()]),
            "amplitude": sorted([[list(k), list(v)] for k, v in self.amplitude.items()]),
            "badflag": sorted([[list(k), list(v)] for k, v in self.badflag.items()]),
            "attitude": sorted([[k, list(v)] for k, v in self.attitude.items()]),
        }


@dataclass
class CandidateView:
    dropped_beam: int
    instrument: list[float]
    ship: list[float]
    earth: list[float]
    fingerprint: str


@dataclass
class DerivedValue:
    value_id: str
    ping_index: int
    cell_index: int
    stage: str
    tier: str
    exclusion: str | None
    beam_solution: dict[str, Any] | None
    instrument: list[float] | None
    ship: list[float] | None
    earth: list[float] | None
    candidates: list[CandidateView]
    attitude_used: dict[str, Any] | None
    fingerprint: str
    diagnostics: list[str]
    source_pins: dict[str, Any]
    order: str
    depth_m: float | None
    beam_quality: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "value_id": self.value_id,
            "ping_index": self.ping_index,
            "cell_index": self.cell_index,
            "stage": self.stage,
            "tier": self.tier,
            "exclusion": self.exclusion,
            "beam_solution": self.beam_solution,
            "instrument": self.instrument,
            "ship": self.ship,
            "earth": self.earth,
            "candidates": [vars(c) for c in self.candidates],
            "attitude_used": self.attitude_used,
            "fingerprint": self.fingerprint,
            "diagnostics": self.diagnostics,
            "source_pins": self.source_pins,
            "order": self.order,
            "depth_m": self.depth_m,
            "beam_quality": self.beam_quality,
        }


@dataclass
class DerivationResult:
    values: list[DerivedValue]
    events: list[dict[str, Any]]
    rules: dict[str, str]
    options: dict[str, Any]


def derive(case, refs: SourceRefs, options: dict[str, Any] | None = None) -> DerivationResult:
    options = dict(options or {})
    order = options.get("order", tf.ORDER_TILT_THEN_HEADING)
    corr_threshold = float(options.get("correlation_threshold_percent",
                                       case.settings.get("correlation_threshold_percent", 64.0)))
    max_attitude_gap = float(options.get("max_attitude_gap_s",
                                         case.settings.get("max_attitude_gap_s", 2.0)))
    three_beam_ambiguous = bool(options.get("keep_all_three_beam_candidates", True))

    geometry = geom.BeamGeometry.from_config(case.transducer["geometry"])
    radial_sign = float(case.transducer.get("radial_sign", -1.0))
    bin_g = case.transducer["bin_geometry"]
    install_angles = case.install["angles"]
    declination_deg = float(case.declination["declination_deg"]["value"])
    decl_version = str(case.declination.get("version", "?"))

    samples = []
    for rec in case.attitudes:
        h = normalize(rec.heading)
        p = normalize(rec.pitch)
        r = normalize(rec.roll)
        flag = normalize(rec.flag)
        qf = int(flag.value) if flag.ok else 1
        if h.ok and p.ok and r.ok:
            samples.append(att.AttitudeSample(
                rec.time_s, np.degrees(h.value), np.degrees(p.value),
                np.degrees(r.value), qf))

    values: list[DerivedValue] = []
    events: list[dict[str, Any]] = []

    def resolve_attitude(ping):
        direct = att.lookup(samples, ping.time_s)
        if direct.ok:
            return direct, "direct"
        interp = att.interpolate(samples, ping.time_s, max_attitude_gap)
        return interp, ("interpolated" if interp.ok else interp.diagnostic)

    for ping in case.pings:
        att_result, att_mode = resolve_attitude(ping)
        for cell in range(len(ping.cell_depths)):
            values.append(_derive_cell(
                case, refs, options, order, corr_threshold, geometry, radial_sign,
                bin_g, install_angles, declination_deg, decl_version,
                att_result, att_mode, ping, cell, events, three_beam_ambiguous))

    for bt in case.bottom_tracks:
        values.append(_derive_bottom(case, refs, order, geometry, radial_sign,
                                     install_angles, declination_deg, decl_version,
                                     samples, max_attitude_gap, bt, events))

    return DerivationResult(values, events, rule_versions(), {
        "order": order,
        "correlation_threshold_percent": corr_threshold,
        "max_attitude_gap_s": max_attitude_gap,
        "keep_all_three_beam_candidates": three_beam_ambiguous,
    })


def _emit(events, code, ping, cell=None, beam=None, status=None, detail=None,
          affected=None):
    ev = {"code": code, "status": status or "info"}
    if ping is not None:
        ev["ping"] = ping
    if cell is not None:
        ev["cell"] = cell
    if beam is not None:
        ev["beam"] = beam
    if detail is not None:
        ev["detail"] = detail
    if affected is not None:
        ev["affected"] = affected
    events.append(ev)


def _derive_cell(case, refs, options, order, corr_threshold, geometry, radial_sign,
                 bin_g, install_angles, declination_deg, decl_version,
                 att_result, att_mode, ping, cell, events, ambiguous):
    diagnostics: list[str] = []
    value_id = f"{case.case_id}:p{ping.ping_index}:c{cell}:profile"
    pins = refs.pins()

    depth_conv = normalize(ping.cell_depths[cell])
    depth_m = depth_conv.value if depth_conv.ok else None

    beam_quality = []
    radial = np.full(geometry.n_beams, np.nan)
    good = np.zeros(geometry.n_beams, dtype=bool)
    for b in range(geometry.n_beams):
        reasons = []
        v = normalize(ping.velocity[cell][b])
        c = normalize(ping.correlation[cell][b])
        a = normalize(ping.amplitude[cell][b])
        f = normalize(ping.bad_flag[cell][b])
        flagged = bool(f.value) if f.ok else True
        if flagged:
            reasons.append(qmod.EX_BAD_BEAM_FLAG)
        if not v.ok:
            reasons.append(f"{qmod.EX_UNCONVERTIBLE}:{v.diagnostic}")
        corr_pct = c.value if c.ok and c.canonical_unit in ("percent",) else (
            c.value if c.ok else None)
        if c.ok and not qmod.correlation_ok(float(c.value), corr_threshold):
            reasons.append(qmod.EX_LOW_CORRELATION)
        is_good = v.ok and not flagged and (
            c.ok and qmod.correlation_ok(float(c.value), corr_threshold))
        good[b] = is_good
        if v.ok:
            radial[b] = v.value
        beam_quality.append({
            "beam": b,
            "velocity": v.to_dict(),
            "correlation": c.to_dict(),
            "amplitude": a.to_dict(),
            "bad_flag": f.to_dict(),
            "good": bool(is_good),
            "reasons": reasons,
        })

    valid_depth = depth_conv.ok and qmod.valid_cell(
        depth_m, float(bin_g["blanking_m"]), float(bin_g["cell_size_m"]),
        int(bin_g["n_cells"]), float(bin_g["first_bin_distance_m"]))

    if not valid_depth:
        diag = depth_conv.diagnostic or qmod.EX_INVALID_CELL
        diagnostics.append(diag)
        _emit(events, "invalid_depth_cell", ping.ping_index, cell,
               status="excluded", detail=diag,
               affected=[value_id])
        return _pack(value_id, ping.ping_index, cell, "profile",
                     qmod.TIER_EXCLUDED, qmod.EX_INVALID_CELL, None,
                     None, None, None, [], None, diagnostics, refs, order,
                     depth_m, beam_quality, options)

    n_good = int(good.sum())
    if geometry.n_beams == 4 and n_good == 3:
        _emit(events, "single_beam_failure", ping.ping_index, cell,
              beam=int(np.nonzero(~good)[0][0]), status="degraded",
              detail="one_beam_flagged_or_low_quality_exact_three_beam_solution",
              affected=[value_id])

    if n_good < 3:
        diagnostics.append(qmod.EX_INSUFFICIENT_BEAMS)
        _emit(events, "insufficient_beams", ping.ping_index, cell,
              status="excluded", detail=f"{n_good}_good_beams",
              affected=[value_id])

    sol = geom.solve_beams(geometry, radial, good, radial_sign)
    candidates: list[CandidateView] = []

    if not att_result.ok:
        reason = (qmod.EX_ATTITUDE_GAP if (att_result.diagnostic or "").startswith("attitude_gap")
                  else qmod.EX_ATTITUDE_MISSING)
        diagnostics.append(att_result.diagnostic or reason)
        _emit(events, "attitude_unavailable", ping.ping_index, cell,
              status="excluded", detail=att_result.diagnostic,
              affected=[value_id])
        inst = list(sol.velocity) if sol.velocity else None
        return _pack(value_id, ping.ping_index, cell, "profile",
                     qmod.TIER_EXCLUDED, reason, sol.to_dict(),
                     inst, None, None, [], None, diagnostics, refs, order,
                     depth_m, beam_quality, options)

    pitch_d, roll_d, head_d = tf.apply_install_offsets(
        att_result.pitch_deg, att_result.roll_deg, att_result.heading_deg,
        install_angles)
    attitude_used = {
        "mode": att_mode,
        "heading_deg": head_d,
        "pitch_deg": pitch_d,
        "roll_deg": roll_d,
        "bracket": att_result.bracket,
        "declination_deg": declination_deg,
        "declination_version": decl_version,
        "install_version": refs.install[1],
    }

    chain = None
    if sol.velocity is not None:
        chain = tf.run_chain(sol.velocity, head_d, pitch_d, roll_d,
                             declination_deg, order)

    # Multi-solution handling -------------------------------------------------
    err_threshold = float(case.settings.get("error_velocity_threshold_m_s", 0.25))
    ambiguous_multi = False
    if n_good >= 4:
        if sol.error_velocity is not None and abs(sol.error_velocity) > err_threshold and ambiguous:
            ambiguous_multi = True
            cands = geom.three_beam_candidates(geometry, radial,
                                               radial_sign=radial_sign)
            for cand in cands:
                cchain = tf.run_chain(cand.velocity, head_d, pitch_d, roll_d,
                                      declination_deg, order)
                view = CandidateView(cand.dropped_beam, list(cand.velocity),
                                     list(cchain.ship), list(cchain.earth), "")
                view.fingerprint = fingerprint({
                    "kind": "candidate", "value_id": value_id,
                    "dropped_beam": cand.dropped_beam,
                    "instrument": view.instrument, "pins": pins,
                    "options": options})
                candidates.append(view)
            diagnostics.append("ambiguous_three_beam_candidates_retained")
            _emit(events, "three_beam_multisolution", ping.ping_index, cell,
                  status="ambiguous",
                  detail=f"|error_velocity|={abs(sol.error_velocity):.3f}>{err_threshold}",
                  affected=[value_id] + [f"{value_id}#drop{c.dropped_beam}"
                                         for c in cands])
    elif n_good == 3:
        cands = geom.three_beam_candidates(
            geometry, radial,
            allowed_drops=tuple(int(i) for i in np.nonzero(~good)[0]),
            radial_sign=radial_sign)
        for cand in cands:
            cchain = tf.run_chain(cand.velocity, head_d, pitch_d, roll_d,
                                  declination_deg, order)
            view = CandidateView(cand.dropped_beam, list(cand.velocity),
                                 list(cchain.ship), list(cchain.earth), "")
            view.fingerprint = fingerprint({
                "kind": "candidate", "value_id": value_id,
                "dropped_beam": cand.dropped_beam,
                "instrument": view.instrument, "pins": pins,
                "options": options})
            candidates.append(view)

    if sol.kind == "insufficient":
        tier, exclusion = qmod.TIER_EXCLUDED, qmod.EX_INSUFFICIENT_BEAMS
    elif sol.kind == "three":
        tier, exclusion = qmod.TIER_THREE, None
    else:
        tier, exclusion = qmod.TIER_FOUR, None
        if ambiguous_multi:
            tier = qmod.TIER_UNRESOLVED

    fp = fingerprint({
        "kind": "value", "value_id": value_id,
        "beam_solution": sol.to_dict(), "depth_m": depth_m,
        "attitude": attitude_used, "pins": pins, "options": options})
    dv = _pack(value_id, ping.ping_index, cell, "profile", tier, exclusion,
               sol.to_dict(),
               list(chain.instrument) if chain else None,
               list(chain.ship) if chain else None,
               list(chain.earth) if chain else None,
               candidates, attitude_used, diagnostics, refs, order,
               depth_m, beam_quality, options, fp=fp)
    return dv


def _pack(value_id, ping, cell, stage, tier, exclusion, beam_solution,
          instrument, ship, earth, candidates, attitude_used, diagnostics,
          refs, order, depth_m, beam_quality, options, fp=None):
    if fp is None:
        fp = fingerprint({
            "kind": "excluded", "value_id": value_id, "tier": tier,
            "exclusion": exclusion, "pins": refs.pins(), "options": options})
    return DerivedValue(
        value_id, ping, cell, stage, tier, exclusion, beam_solution,
        instrument, ship, earth, candidates, attitude_used, fp,
        diagnostics, refs.pins(), order, depth_m, beam_quality)


def _derive_bottom(case, refs, order, geometry, radial_sign, install_angles,
                   declination_deg, decl_version, samples, max_gap, bt, events):
    value_id = f"{case.case_id}:bt:{int(bt.time_s * 1000)}:bottom_track"
    diagnostics: list[str] = []
    radial = np.full(geometry.n_beams, np.nan)
    good = np.zeros(geometry.n_beams, dtype=bool)
    quality = []
    for b in range(geometry.n_beams):
        v = normalize(bt.velocity[b])
        f = normalize(bt.bad_flag[b])
        flagged = bool(f.value) if f.ok else True
        good[b] = v.ok and not flagged
        if v.ok:
            radial[b] = v.value
        quality.append({"beam": b, "velocity": v.to_dict(),
                        "bad_flag": f.to_dict(), "good": bool(good[b])})
    sol = geom.solve_beams(geometry, radial, good, radial_sign)
    att_result = att.lookup(samples, bt.time_s)
    if not att_result.ok:
        att_result = att.interpolate(samples, bt.time_s, max_gap)
    chain = None
    attitude_used = None
    exclusion = None
    if sol.kind == "insufficient":
        tier = qmod.TIER_EXCLUDED
        exclusion = qmod.EX_INSUFFICIENT_BEAMS
    elif not att_result.ok:
        tier = qmod.TIER_EXCLUDED
        exclusion = qmod.EX_ATTITUDE_GAP
    else:
        tier = qmod.TIER_FOUR if sol.kind == "four" else qmod.TIER_THREE
        pitch_d, roll_d, head_d = tf.apply_install_offsets(
            att_result.pitch_deg, att_result.roll_deg, att_result.heading_deg,
            install_angles)
        attitude_used = {"mode": "bt", "heading_deg": head_d,
                         "pitch_deg": pitch_d, "roll_deg": roll_d,
                         "declination_deg": declination_deg,
                         "declination_version": decl_version}
        chain = tf.run_chain(sol.velocity, head_d, pitch_d, roll_d,
                             declination_deg, order)
    fp = fingerprint({
        "kind": "bottom_track", "value_id": value_id,
        "beam_solution": sol.to_dict(), "pins": refs.pins(),
        "order": order})
    return DerivedValue(
        value_id, -1, -1, "bottom_track", tier, exclusion, sol.to_dict(),
        list(chain.instrument) if chain else None,
        list(chain.ship) if chain else None,
        list(chain.earth) if chain else None, [], attitude_used, fp,
        diagnostics, refs.pins(), order, None, quality)
