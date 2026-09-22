"""逐级向量变换的纯函数核心（无 IO、无数据库）。

链路：
    原始四类证据(波束速度/correlation/amplitude/坏标志)
      -> 波束三态判定
      -> 候选波束集合（三波束/四波束，多解全部保留）
      -> instrument 坐标速度（三波束无 error velocity；四波束有）
      -> ship 坐标速度（安装角，顺序进指纹）
      -> earth(ENU) 坐标速度（姿态 h/p/r + 磁偏角版本，顺序进指纹）

深度 cell 有效性（盲区/超过 bottom track）独立判定；无效 cell 保留原始材料，
只把派生标记为 EXCLUDED。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .config import (DeclinationConfig, GeometryConfig, InstallConfig,
                     TransformConfig)
from .geometry import BeamGeometry
from .quality import (QCThresholds, BeamVerdict, candidate_beam_sets,
                      judge_beam)
from .rotations import install_matrix
from .units import Quantity, to_quantity


def _vec_dict(v: np.ndarray, unit: str, datum: str) -> Dict[str, Any]:
    return {
        "components": [
            Quantity(float(v[0]), unit, f"{datum}:1").as_dict(),
            Quantity(float(v[1]), unit, f"{datum}:2").as_dict(),
            Quantity(float(v[2]), unit, f"{datum}:3").as_dict(),
        ],
        "unit": unit,
        "datum": datum,
    }


def qc_cell(n_beams: int, cell_evidence: Dict[str, dict],
            thresholds: QCThresholds) -> List[BeamVerdict]:
    """cell_evidence[beam] = {velocity, velocity_unit, correlation,
    correlation_unit, amplitude, amplitude_unit, bad_flag}（原始表示）。"""
    verdicts = []
    for b in range(n_beams):
        ev = cell_evidence.get(b) or {}
        verdicts.append(judge_beam(
            b,
            velocity=ev.get("velocity"),
            velocity_raw_unit=ev.get("velocity_unit"),
            correlation=ev.get("correlation"),
            correlation_raw_unit=ev.get("correlation_unit"),
            amplitude=ev.get("amplitude"),
            amplitude_raw_unit=ev.get("amplitude_unit"),
            bad_flag=ev.get("bad_flag"),
            thresholds=thresholds,
        ))
    return verdicts


def solve_instrument(geometry: BeamGeometry, beams: Sequence[int],
                     cell_evidence: Dict[str, dict]) -> Dict[str, Any]:
    """对选定波束集合解 instrument 速度。"""
    bvals = []
    used = []
    for b in beams:
        ev = cell_evidence[b]
        q, diag = to_quantity(ev.get("velocity"), ev.get("velocity_unit"), "m/s")
        if diag is not None or q is None:
            raise ValueError(f"beam {b} 速度不可用，不应进入求解集合")
        bvals.append(q.value)
        used.append(b)
    v, err = geometry.solve(np.asarray(bvals), used)
    rank = len(used)
    out = {
        "beams": list(used),
        "rank": rank,
        "velocity": _vec_dict(v, "m/s", "instrument"),
        "radial_mps": bvals,
    }
    if rank >= 4:
        out["error_velocity"] = Quantity(err, "m/s", "error-velocity").as_dict()
    else:
        # 三波束解没有四波束冗余定义的 error velocity，绝不拿四波束值填充
        out["error_velocity"] = None
    return out


def _heading_matrix(h: float) -> np.ndarray:
    c, s = math.cos(h), math.sin(h)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def _pitch_matrix(p: float) -> np.ndarray:
    c, s = math.cos(p), math.sin(p)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _roll_matrix(r: float) -> np.ndarray:
    c, s = math.cos(r), math.sin(r)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


_HEAD = {"h": _heading_matrix, "p": _pitch_matrix, "r": _roll_matrix}


def ship_earth_matrix(heading_rad: float, pitch_rad: float, roll_rad: float,
                      order: str) -> np.ndarray:
    """按 order（h/p/r 的 intrinsic 次序）合成 ship->earth 矩阵。

    order='hpr' => R_h · R_p · R_r。
    """
    amap = {"h": heading_rad, "p": pitch_rad, "r": roll_rad}
    R = np.eye(3)
    for ax in order:
        R = _HEAD[ax](amap[ax]) @ R
    return R


def to_ship(v_inst: np.ndarray, install: InstallConfig) -> Dict[str, Any]:
    R = install_matrix(install.angles_rad(), order=install.order)
    v = R @ v_inst
    return {"velocity": _vec_dict(v, "m/s", "ship"),
            "matrix": R.tolist(),
            "install_version": install.version}


def to_earth(v_ship: np.ndarray, heading_deg_mag: float, pitch_deg: float,
             roll_deg: float, decl: DeclinationConfig, tform: TransformConfig
             ) -> Dict[str, Any]:
    heading_true = heading_deg_mag + decl.declination_deg
    R = ship_earth_matrix(math.radians(heading_true), math.radians(pitch_deg),
                          math.radians(roll_deg), tform.ship_earth_order)
    v = R @ v_ship
    return {
        "velocity": _vec_dict(v, "m/s", "earth-ENU"),
        "matrix": R.tolist(),
        "heading_magnetic_deg": Quantity(heading_deg_mag, "deg", "MAGNETIC").as_dict(),
        "heading_true_deg": Quantity(heading_true, "deg", "TRUE").as_dict(),
        "pitch_deg": pitch_deg,
        "roll_deg": roll_deg,
        "declination": decl.fingerprint_input(),
        "transform_version": tform.version,
        "transform_order": tform.ship_earth_order,
    }


def depth_cell_validity(cell_depth_m: float, blank_m: float,
                        bottom_m: Optional[float]) -> Dict[str, Any]:
    """无效深度 cell：盲区以下才有效，且不能超过 bottom track 深度。"""
    reasons = []
    valid = True
    if cell_depth_m < blank_m - 1e-9:
        valid = False
        reasons.append("IN_BLANK_ZONE")
    if bottom_m is not None and cell_depth_m > bottom_m + 1e-9:
        valid = False
        reasons.append("BELOW_BOTTOM_TRACK")
    if not math.isfinite(cell_depth_m):
        valid = False
        reasons.append("NON_FINITE_DEPTH")
    return {
        "depth": Quantity(cell_depth_m, "m", "transducer-down").as_dict(),
        "blank_m": blank_m,
        "bottom_m": bottom_m,
        "valid": valid,
        "reasons": reasons,
    }


def derive_cell(*, n_beams: int, cell_evidence: Dict[str, dict],
                thresholds: QCThresholds, geometry_cfg: GeometryConfig,
                install: InstallConfig, depth: Dict[str, Any],
                attitude: Optional[Dict[str, Any]],
                decl: DeclinationConfig, tform: TransformConfig
                ) -> Dict[str, Any]:
    """单个深度 cell 的完整逐级派生（含全部候选）。"""
    geometry = geometry_cfg.geometry
    verdicts = qc_cell(n_beams, cell_evidence, thresholds)
    sets = candidate_beam_sets(verdicts, n_beams)

    depth_status = depth["valid"]
    candidates = []
    if depth_status:
        for cand in sets:
            beams = cand["beams"]
            inst = solve_instrument(geometry, beams, cell_evidence)
            v_inst = np.asarray([c["value"] for c in inst["velocity"]["components"]])
            ship = to_ship(v_inst, install)
            v_ship = np.asarray([c["value"] for c in ship["velocity"]["components"]])
            earth = None
            if attitude is not None and attitude.get("status") == "OK":
                earth = to_earth(v_ship, attitude["heading_deg"],
                                 attitude["pitch_deg"], attitude["roll_deg"],
                                 decl, tform)
            elif attitude is not None:
                earth = {"status": attitude["status"],
                         "reasons": attitude.get("reasons", [])}
            candidates.append({
                "beams": list(beams),
                "rank": cand["rank"],
                "indeterminate_included": list(cand["indeterminate_included"]),
                "instrument": inst,
                "ship": ship,
                "earth": earth,
            })

    return {
        "qc": [v.as_dict() for v in verdicts],
        "candidate_sets": [
            {"beams": list(c["beams"]), "rank": c["rank"],
             "indeterminate_included": list(c["indeterminate_included"])}
            for c in sets
        ],
        "depth": depth,
        "candidates": candidates,
        "status": "OK" if depth_status else "EXCLUDED",
    }


def solve_bottom_track(*, n_beams: int, bt_evidence: Dict[str, dict],
                       thresholds: QCThresholds, geometry_cfg: GeometryConfig,
                       install: InstallConfig, attitude: Optional[Dict[str, Any]],
                       decl: DeclinationConfig, tform: TransformConfig
                       ) -> Dict[str, Any]:
    """bottom track 与水层走同一套质量/候选/逐级变换。"""
    verdicts = qc_cell(n_beams, bt_evidence, thresholds)
    sets = candidate_beam_sets(verdicts, n_beams)
    candidates = []
    for cand in sets:
        beams = cand["beams"]
        inst = solve_instrument(geometry_cfg.geometry, beams, bt_evidence)
        v_inst = np.asarray([c["value"] for c in inst["velocity"]["components"]])
        ship = to_ship(v_inst, install)
        v_ship = np.asarray([c["value"] for c in ship["velocity"]["components"]])
        earth = None
        if attitude is not None and attitude.get("status") == "OK":
            earth = to_earth(v_ship, attitude["heading_deg"], attitude["pitch_deg"],
                             attitude["roll_deg"], decl, tform)
        elif attitude is not None:
            earth = {"status": attitude["status"], "reasons": attitude.get("reasons", [])}
        candidates.append({
            "beams": list(beams), "rank": cand["rank"],
            "indeterminate_included": list(cand["indeterminate_included"]),
            "instrument": inst, "ship": ship, "earth": earth,
        })
    # bottom track 深度（沿波束测距反演的垂直深度，用 beam4/3 平均近似）
    ranges = {b: (bt_evidence.get(b) or {}).get("range_m") for b in range(n_beams)}
    zen = math.radians(geometry_cfg.geometry.zenith_deg[0])
    depths = [r / math.cos(zen) for r in ranges.values()
              if isinstance(r, (int, float)) and math.isfinite(r)]
    bottom_m = sum(depths) / len(depths) if depths else None
    return {
        "qc": [v.as_dict() for v in verdicts],
        "candidate_sets": [
            {"beams": list(c["beams"]), "rank": c["rank"],
             "indeterminate_included": list(c["indeterminate_included"])}
            for c in sets
        ],
        "candidates": candidates,
        "bottom_depth_m": bottom_m,
    }
