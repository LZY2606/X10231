"""核心规则层：beam velocity -> instrument -> ship -> earth 的逐级派生。

硬语义（任何上层都不得绕过）：
  * 三波束解与四波束解是不同候选，绝不合并为单一值；无法唯一决定时保留全部合法候选。
  * 姿态插值不跨缺口；缺口处 instrument/ship 仍可给，earth 标记 attitude_gap 而非硬造。
  * 坐标转换顺序、波束几何版本、磁偏角版本、安装角版本都进入派生指纹。
所有角度在派生层规范化（SI/度），原始表示留在存储层。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .attitude import AttitudeResult, instrument_to_ship, ship_to_earth
from .beam import BeamGeometry
from .units import parse_quantity


@dataclass
class BeamObservation:
    index: int
    velocity_si: float | None
    diagnostic: str | None
    correlation_pct: float | None
    amplitude_counts: float | None
    bad_flag: bool
    quality_ok: bool
    quality_reasons: list[str] = field(default_factory=list)
    amplitude_diagnostic: str | None = None
    correlation_diagnostic: str | None = None


def _q(raw: Any, unit: str) -> tuple[float | None, str | None]:
    q = parse_quantity(raw, unit)
    return (q.si, q.diagnostic)


def evaluate_beams(
    beam_vel: list[Any],
    beam_vel_unit: str,
    correlation: list[Any],
    correlation_unit: str,
    amplitude: list[Any],
    amplitude_unit: str,
    bad_flags: list[Any],
    *,
    corr_threshold_pct: float,
    n_beams: int = 4,
) -> list[BeamObservation]:
    obs: list[BeamObservation] = []
    for i in range(n_beams):
        vel, vdiag = _q(beam_vel[i] if i < len(beam_vel) else None, beam_vel_unit)
        corr, cdiag = _q(correlation[i] if i < len(correlation) else None, correlation_unit)
        amp, adia = _q(amplitude[i] if i < len(amplitude) else None, amplitude_unit)
        raw_flag = str(bad_flags[i]).strip().lower() if i < len(bad_flags) and bad_flags[i] is not None else "0"
        flagged = raw_flag in ("1", "true", "bad", "y", "yes")
        bad_unsupported = raw_flag not in ("0", "1", "true", "false", "bad", "good", "y", "n", "yes", "no")
        reasons: list[str] = []
        if vdiag:
            reasons.append(f"velocity:{vdiag}")
        if cdiag:
            reasons.append(f"correlation:{cdiag}")
        if adia:
            # amplitude 无法转换不直接判 beam 坏，但要在质量面板保留诊断
            pass
        if bad_unsupported:
            reasons.append(f"bad_flag_unrecognized:{raw_flag}")
        if flagged:
            reasons.append("bad_flag")
        if corr is not None and corr < corr_threshold_pct:
            reasons.append("low_correlation")
        if vel is None and not flagged:
            reasons.append("missing_velocity")
        obs.append(BeamObservation(i, vel, vdiag, corr, amp, flagged, not reasons, reasons,
                                   adia, cdiag))
    return obs


def _vec3(v: np.ndarray | None) -> list[float] | None:
    return None if v is None else [float(x) for x in v]


@dataclass
class Candidate:
    candidate_id: str
    kind: str                     # four | three
    missing_beams: list[int]
    instrument: list[float] | None
    ship: list[float] | None
    earth: list[float] | None
    earth_reason: str | None
    error_velocity_mps: float | None
    quality: str                  # nominal | high_error_velocity


def solve_instrument_candidates(
    obs: list[BeamObservation], geometry: BeamGeometry, errvel_threshold_mps: float
) -> tuple[list[Candidate], list[dict]]:
    events: list[dict] = []
    good = [o for o in obs if o.quality_ok and o.velocity_si is not None]
    beams = np.array([o.velocity_si if o.velocity_si is not None else np.nan for o in obs])
    cands: list[Candidate] = []
    if len(good) == 4:
        err = geometry.error_velocity(beams)
        v4 = geometry.solve_four(beams)
        quality = "nominal"
        if abs(err) > errvel_threshold_mps:
            quality = "high_error_velocity"
            events.append({"type": "error_velocity_exceeded",
                           "detail": {"error_velocity_mps": err, "threshold_mps": errvel_threshold_mps}})
        c4 = Candidate("cand4", "four", [], None, None, None, None, err, quality)
        c4.instrument = _vec3(v4)
        # 误差速度过大时，四波束解不再唯一可信：展开全部 4 个合法三波束候选
        if quality == "high_error_velocity":
            for present in geometry.triple_indices():
                missing = [i for i in range(4) if i not in present]
                if geometry.triple_is_independent(present):
                    v3 = geometry.solve_three(beams, present)
                    cands.append(Candidate(f"cand3-missing{missing[0]}", "three", missing,
                                           _vec3(v3), None, None, None, None, "nominal"))
            events.append({"type": "three_beam_multiplicity",
                           "detail": {"n_candidates": len(cands),
                                      "candidate_ids": [c.candidate_id for c in cands]}})
        cands.insert(0, c4)
    elif len(good) == 3:
        present = tuple(o.index for o in good)
        missing = [i for i in range(4) if i not in present]
        if geometry.triple_is_independent(present):
            v3 = geometry.solve_three(beams, present)
            cands.append(Candidate(f"cand3-missing{missing[0]}", "three", missing,
                                   _vec3(v3), None, None, None, None, "nominal"))
            events.append({"type": "single_beam_failure",
                           "detail": {"missing_beam": missing[0], "candidate_id": cands[-1].candidate_id}})
        else:
            events.append({"type": "no_solution", "detail": {"reason": "dependent triple", "present": list(present)}})
    else:
        events.append({"type": "no_solution",
                       "detail": {"reason": "fewer than three good beams", "n_good": len(good)}})
    return cands, events


def _apply_frame_rotations(
    cands: list[Candidate],
    attitude: AttitudeResult,
    declination_deg: float,
    mount: tuple[float, float, float],
    order: str,
) -> None:
    r_is = instrument_to_ship(*mount)
    r_se: np.ndarray | None = None
    earth_reason = None
    if attitude.status != "ok":
        earth_reason = "attitude_gap" if attitude.status == "gap" else "attitude_missing"
    else:
        true_heading = (attitude.heading_deg + declination_deg) % 360.0
        r_se = ship_to_earth(true_heading, attitude.pitch_deg, attitude.roll_deg, order)
    for c in cands:
        vi = np.array(c.instrument)
        c.ship = _vec3(r_is @ vi)
        if r_se is not None:
            c.earth = _vec3(r_se @ np.array(c.ship))
        else:
            c.earth_reason = earth_reason


def cell_exclusion_reasons(
    depth_m: float, *, blanking_m: float, min_depth_m: float, bottom_depth_m: float | None
) -> list[str]:
    reasons: list[str] = []
    if depth_m < blanking_m or depth_m < min_depth_m:
        reasons.append("blanking_zone")
    if bottom_depth_m is not None and depth_m > bottom_depth_m:
        reasons.append("below_bottom")
    return reasons


def compute_ping_cells(ctx: dict[str, Any]) -> dict[str, Any]:
    """计算一个 ping 下全部 depth cell 的派生结果。返回 profile 结构 + 事件。"""
    geom: BeamGeometry = ctx["geometry"]
    cells_out: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = list(ctx.get("attitude_events", []))
    for cell in ctx["cells"]:
        depth_m = cell["depth_m"]
        exclusions = cell_exclusion_reasons(
            depth_m, blanking_m=ctx["blanking_m"], min_depth_m=ctx["min_depth_m"],
            bottom_depth_m=ctx.get("bottom_depth_m"))
        obs = evaluate_beams(
            cell["beam_vel"], cell.get("beam_vel_unit", "m/s"),
            cell["correlation"], cell.get("correlation_unit", "pct"),
            cell["amplitude"], cell.get("amplitude_unit", "count"),
            cell["bad_flags"], corr_threshold_pct=ctx["corr_threshold_pct"])
        cands, cevents = solve_instrument_candidates(obs, geom, ctx["errvel_threshold_mps"])
        for ev in cevents:
            ev.setdefault("ping_id", ctx["ping_id"])
            ev.setdefault("cell_id", cell["cell_id"])
        events.extend(cevents)
        _apply_frame_rotations(cands, ctx["attitude"], ctx["declination_deg"],
                               ctx["mount"], ctx["transform_order"])
        if not cands and "no_velocity_candidate" not in exclusions:
            exclusions.append("no_velocity_candidate")
        candidates = [vars(c) for c in cands]
        default_id = candidates[0]["candidate_id"] if candidates else None
        # 多解时不自动选定四波束：默认保持未裁定
        ambiguous = any(c["kind"] == "three" for c in candidates) and any(c["kind"] == "four" for c in candidates)
        if ambiguous:
            default_id = None
        cells_out.append({
            "cell_id": cell["cell_id"],
            "ping_id": ctx["ping_id"],
            "depth_m": depth_m,
            "depth_frame": ctx.get("depth_frame", "transducer_down"),
            "observations": [vars(o) for o in obs],
            "candidates": candidates,
            "default_candidate_id": default_id,
            "ambiguous": ambiguous,
            "excluded": bool(exclusions),
            "exclude_reasons": exclusions,
            "events": cevents,
            "units": {"velocity": "m/s", "correlation": "pct", "amplitude": "count", "depth": "m"},
        })
    return {
        "ping_id": ctx["ping_id"],
        "time_s": ctx["time_s"],
        "attitude_status": ctx["attitude"].status,
        "attitude": vars(ctx["attitude"]),
        "cells": cells_out,
        "events": events,
        "basis": {
            "geometry_version": ctx["geom_version"],
            "declination_version": ctx["decl_version"],
            "mount_version": ctx["mount_version"],
            "transform_order": ctx["transform_order"],
            "errvel_threshold_mps": ctx["errvel_threshold_mps"],
            "corr_threshold_pct": ctx["corr_threshold_pct"],
        },
    }


def compute_bottom_track(ctx: dict[str, Any]) -> dict[str, Any]:
    geom: BeamGeometry = ctx["geometry"]
    bt = ctx["bottom_track"]
    obs = evaluate_beams(
        bt["beam_vel"], bt.get("beam_vel_unit", "m/s"),
        bt.get("correlation", [None] * 4), bt.get("correlation_unit", "pct"),
        bt.get("amplitude", [None] * 4), bt.get("amplitude_unit", "count"),
        bt["bad_flags"], corr_threshold_pct=ctx["corr_threshold_pct"])
    cands, events = solve_instrument_candidates(obs, geom, ctx["errvel_threshold_mps"])
    _apply_frame_rotations(cands, ctx["attitude"], ctx["declination_deg"],
                           ctx["mount"], ctx["transform_order"])
    return {
        "bt_id": bt["bt_id"], "ping_id": ctx["ping_id"],
        "range_m": bt.get("range_m"),
        "observations": [vars(o) for o in obs],
        "candidates": [vars(c) for c in cands],
        "events": events,
        "basis": {
            "geometry_version": ctx["geom_version"],
            "declination_version": ctx["decl_version"],
            "mount_version": ctx["mount_version"],
            "transform_order": ctx["transform_order"],
        },
    }
