"""波束几何与 beam velocity -> instrument frame 求解。

坐标约定（左-handed，z 向下为正）：
  x 指向右舷(starboard)，y 指向船首(forward)，z 向下(down)。
  beam velocity 正值表示水体“沿波束方向远离换能器”（典型 RDI/凸阵约定）。

凸阵 Janus 四波束在 x/y 平面方位（从 +y 顺时针到 +x 为正）：
  beam1 右前(+x,+y)  beam2 左前(-x,+y)  beam3 左后(-x,-y)  beam4 右后(+x,-y)
束与垂直轴夹角为 beam_angle。
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

BEAM_AZIMUTH_DEG = (45.0, -45.0, 180.0 - 45.0, -(180.0 - 45.0))
# 单位几何向量行（m: beam->instrument 映射）
# v_beam = - G @ u_instrument，G[i] 为波束 i 的单位指向


def geometry_matrix(beam_angle_rad: float) -> np.ndarray:
    """返回 4x3 矩阵 G，满足 v_beam = -G @ u_inst。

    G[i] = (sinθ·sin az, sinθ·cos az, cosθ)
    """
    rows = []
    s, c = np.sin(beam_angle_rad), np.cos(beam_anglerad := beam_angle_rad)
    c = np.cos(beam_angle_rad)
    for az_deg in BEAM_AZIMUTH_DEG:
        az = np.deg2rad(az_deg)
        rows.append([s * np.sin(az), s * np.cos(az), c])
    return np.array(rows, dtype=float)


def solve_beams(
    velocities: dict[int, float],
    beam_angle_rad: float,
    eligible: Optional[set[int]] = None,
) -> dict[str, Any]:
    """由可用 beam velocity 求 instrument-frame 速度。

    velocities: {beam_index(1..4): velocity in m/s}
    eligible: 允许参与的 beam（质量阈值后）；缺省用 velocities 的键。

    四波束：最小二乘解（本几何下即精确解）+ error velocity（零空间残差）。
    三波束：该三元组唯一确定 (u,v,w)，error velocity 无法校验 -> None。
    返回 dict: u,v,w,error_velocity,residual_rms,beams,method
    """
    G_full = geometry_matrix(beam_angle_rad)
    beams = sorted(velocities)
    if eligible is not None:
        beams = sorted(b for b in beams if b in eligible)
    if len(beams) < 3:
        return {
            "solvable": False, "reason": "少于三个可用波束",
            "beams": beams, "u": None, "v": None, "w": None,
            "error_velocity": None, "residual_rms": None, "method": "none",
        }
    idx = [b - 1 for b in beams]
    G = G_full[idx]
    v = np.array([velocities[b] for b in beams], dtype=float)
    # v_beam = -G u  =>  u = lstsq(-G, v)
    uvec, residuals, rank, _ = np.linalg.lstsq(-G, v, rcond=None)
    recon = -G @ uvec
    residual_rms = float(np.sqrt(np.mean((recon - v) ** 2)))
    u, vv, w = (float(x) for x in uvec)
    out: dict[str, Any] = {
        "solvable": True, "beams": beams,
        "u": u, "v": vv, "w": w, "residual_rms": residual_rms,
    }
    if len(beams) == 4:
        # 零空间 n=(1,-1,1,-1)/2，error velocity = n·v_beam
        err = 0.5 * (v[0] - v[1] + v[2] - v[3])
        out["error_velocity"] = float(err)
        out["method"] = "4beam_lstsq"
        out["multiplicity"] = 1
    else:
        out["error_velocity"] = None
        out["method"] = "3beam_exact"
        out["multiplicity"] = 1
    return out


def three_beam_candidates(
    velocities: dict[int, float],
    beam_angle_rad: float,
    marginal_beams: Optional[set[int]] = None,
) -> list[dict[str, Any]]:
    """枚举所有合法三波束候选；不把多解合并为一个值。

    - 恰好 3 个可用波束：1 个候选。
    - 4 个波束且存在 marginal beam：每个“去掉一个 marginal beam”的三元组都是合法候选，
      同时另行给出四波束解（四波束与三波束分别成项，不混为一值）。
    """
    candidates: list[dict[str, Any]] = []
    beams = sorted(velocities)
    marginal_beams = marginal_beams or set()
    if len(beams) == 3:
        sol = solve_beams(velocities, beam_angle_rad)
        sol["candidate_id"] = "3b-" + "".join(str(b) for b in sol["beams"])
        candidates.append(sol)
        return candidates
    if len(beams) == 4:
        # 四波束解单独成项
        four = solve_beams(velocities, beam_angle_rad)
        four["candidate_id"] = "4b-1234"
        candidates.append(four)
        for missing in beams:
            if missing in marginal_beams:
                triple = {b: velocities[b] for b in beams if b != missing}
                sol = solve_beams(triple, beam_angle_rad)
                sol["candidate_id"] = f"3b-drop{missing}"
                sol["dropped_beam"] = missing
                candidates.append(sol)
    return candidates


def instrument_candidates(
    beam_velocity: dict[int, float],
    beam_angle_rad: float,
    beam_quality: dict[int, str],
) -> list[dict[str, Any]]:
    """按质量层级生成 instrument 解候选。

    beam_quality: {beam: 'ok'|'marginal'|'bad'}
    bad beam 不得参与；marginal beam 触发“保留全部合法候选”的多解。
    """
    good = {b for b, q in beam_quality.items() if q == "ok"}
    marginal = {b for b, q in beam_quality.items() if q == "marginal"}
    bad = {b for b, q in beam_quality.items() if q == "bad"}
    usable = {b: beam_velocity[b] for b in beam_velocity if b not in bad}
    candidates: list[dict[str, Any]] = []
    ngood = len(good)
    if len(usable) == 4 and not marginal:
        sol = solve_beams(usable, beam_angle_rad)
        sol["candidate_id"] = "4b-1234"
        return [sol]
    if len(usable) == 3 and not (marginal & set(usable)):
        sol = solve_beams(usable, beam_angle_rad)
        sol["candidate_id"] = "3b-" + "".join(str(b) for b in sorted(usable))
        return [sol]
    # 存在 marginal：枚举全部合法三元组 + 四波束
    return three_beam_candidates(usable, beam_angle_rad, marginal_beams=marginal)
