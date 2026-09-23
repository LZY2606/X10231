"""姿态旋转矩阵与姿态时间序列插值。

三个坐标基准：
  instrument（仪器系，随换能器）
  ship     （船体系，经安装角校正）
  earth    （地理系，x 真北、y 东、z 向下）

旋转链顺序是派生指纹的一部分，两种合法顺序分别实现、不互相冒充：
  rdi_xyz : R = R_h @ R_p @ R_r（RDI XYZ 惯例，先 roll 后 heading）
  hpr     : R = R_h @ R_r @ P（heading-roll-pitch 另一种常见顺序）
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

ORDERS = ("rdi_xyz", "hpr")


def _rz(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rx(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def ship_to_earth(heading_deg: float, pitch_deg: float, roll_deg: float, order: str = "rdi_xyz") -> np.ndarray:
    if order not in ORDERS:
        raise ValueError(f"unknown rotation order {order!r}; valid={ORDERS}")
    h, p, r = math.radians(heading_deg), math.radians(pitch_deg), math.radians(roll_deg)
    if order == "rdi_xyz":
        return _rz(h) @ _ry(p) @ _rx(r)
    return _rz(h) @ _rx(r) @ _ry(p)


def instrument_to_ship(mount_heading_deg: float, mount_pitch_deg: float, mount_roll_deg: float) -> np.ndarray:
    # 安装角是仪器系到船体系的固定相对旋转，走同一物理链。
    return ship_to_earth(mount_heading_deg, mount_pitch_deg, mount_roll_deg, "rdi_xyz")


@dataclass
class AttitudeResult:
    ping_id: str
    time_s: float
    heading_deg: float | None
    pitch_deg: float | None
    roll_deg: float | None
    status: str                       # ok | gap | missing
    reason: str | None = None
    segment_id: str | None = None
    bracketed_by: tuple[str, str] | None = None
    events: list[dict] = field(default_factory=list)
    heading_wrapped: bool = False


def _segments(samples: list[dict], gap_threshold_s: float) -> list[list[dict]]:
    segs: list[list[dict]] = []
    for smp in samples:
        if not segs or smp["time_s"] - segs[-1][-1]["time_s"] > gap_threshold_s:
            segs.append([smp])
        else:
            segs[-1].append(smp)
    return segs


def interpolate_attitude(
    ping_id: str,
    time_s: float,
    samples: list[dict],
    *,
    gap_threshold_s: float = 10.0,
) -> AttitudeResult:
    """在连续段内做线性插值；绝不跨时间缺口插值。

    samples: [{'id','time_s','heading_deg','pitch_deg','roll_deg'}]，按 time 升序。
    heading 先解卷绕再插值，跨越 360 度边界被记录为事件，不产生跳变。
    """
    events: list[dict] = []
    segs = _segments(samples, gap_threshold_s)
    for seg_index, seg in enumerate(segs):
        seg_id = f"seg{seg_index}:{seg[0]['id']}..{seg[-1]['id']}"
        # 找包围样本（必须在同一段内；缺口两侧的样本不会同时出现在这里）
        left = right = None
        for i, smp in enumerate(seg):
            if smp["time_s"] <= time_s:
                left = i
            if smp["time_s"] >= time_s and right is None:
                right = i
        if left is None or right is None:
            # 可能落在两段之间的空档；继续看下一段，最后统一判 gap
            if not (seg[0]["time_s"] - 1e-9 <= time_s <= seg[-1]["time_s"] + 1e-9):
                continue
            return AttitudeResult(ping_id, time_s, None, None, None, "missing",
                                  "ping outside attitude segment", seg_id, events=events)
        headings = [smp["heading_deg"] for smp in seg]
        unwrapped = np.unwrap([math.radians(h) for h in headings])
        if seg[left]["time_s"] == seg[right]["time_s"]:
            frac = 0.0
        else:
            frac = (time_s - seg[left]["time_s"]) / (seg[right]["time_s"] - seg[left]["time_s"])
        h_u = unwrapped[left] + frac * (unwrapped[right] - unwrapped[left])
        heading = math.degrees(h_u) % 360.0
        pitch = seg[left]["pitch_deg"] + frac * (seg[right]["pitch_deg"] - seg[left]["pitch_deg"])
        roll = seg[left]["roll_deg"] + frac * (seg[right]["roll_deg"] - seg[left]["roll_deg"])
        wrapped = (math.floor(math.degrees(unwrapped[left]) / 360.0)
                   != math.floor(math.degrees(unwrapped[right]) / 360.0)) or (
                   abs(headings[right] - headings[left]) > 180.0)
        if wrapped:
            events.append({
                "type": "heading_wrap_360", "ping_id": ping_id,
                "detail": {"left_heading_deg": headings[left], "right_heading_deg": headings[right],
                           "interpolated_deg": heading},
            })
        return AttitudeResult(ping_id, time_s, heading, pitch, roll, "ok",
                              None, seg_id, (seg[left]["id"], seg[right]["id"]), events, wrapped)
    return AttitudeResult(ping_id, time_s, None, None, None, "gap",
                          f"no attitude sample within contiguous segment (gap threshold {gap_threshold_s}s)",
                          events=events)
