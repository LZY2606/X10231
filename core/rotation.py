"""instrument -> ship -> earth 的逐级旋转，以及坐标变换顺序的版本化。

海洋惯例（右手系）：
- ship x 船首，y 右舷，z 向下；earth 取 ENU：x 东，y 北，z 向上。
  注意 ship z 向下而 ENU z 向上，因此零姿态、零安装角时 w_earth = -w_ship。
- heading H 为自北顺时针 0..360；pitch P 船首向上为正；roll R 向右舷倾为正。
- 安装角（heading_alignment_a, pitch_alignment_p, roll_alignment_r）是换能器
  相对船体系的固定偏置，与换能器波束几何分开记录。heading alignment 在进入
  航向旋转时与磁航向、磁偏角相加：H_true = H_mag + declination + a。
- 磁偏角 declination（东向为正）是带版本的独立来源。

变换顺序本身是规则版本的一部分（transform_order 进入派生指纹）。
主顺序 "HPR"：先 R_x(R)，再 R_y(-P)，最后航向+基准翻转到 ENU；
"HRP" 交换 pitch/roll 的施加顺序，作为坐标变换顺序反例，数值上可证不同。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

VALID_ORDERS = ("HPR", "HRP")
EARTH_DATUM = "ENU"


@dataclass(frozen=True)
class MountingAngles:
    heading_alignment_deg: float = 0.0
    pitch_alignment_deg: float = 0.0
    roll_alignment_deg: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "heading_alignment_deg": self.heading_alignment_deg,
            "pitch_alignment_deg": self.pitch_alignment_deg,
            "roll_alignment_deg": self.roll_alignment_deg,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MountingAngles":
        return cls(
            heading_alignment_deg=float(data.get("heading_alignment_deg", 0.0)),
            pitch_alignment_deg=float(data.get("pitch_alignment_deg", 0.0)),
            roll_alignment_deg=float(data.get("roll_alignment_deg", 0.0)),
        )


def _rx(deg: float) -> np.ndarray:
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _ry_neg(deg: float) -> np.ndarray:
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _heading_shipzdown_to_enu(h_deg: float) -> np.ndarray:
    """水平船体系（x 船首/y 右舷/z 向下）到 ENU 的基变换。

    x_ship（船首）水平 ENU 分量为 (sin H, cos H, 0)；
    y_ship（右舷）为 (cos H, -sin H, 0)；z_ship（向下）映射到 (0,0,-1)。
    """
    r = np.deg2rad(h_deg)
    return np.array([
        [np.sin(r), np.cos(r), 0.0],
        [np.cos(r), -np.sin(r), 0.0],
        [0.0, 0.0, -1.0],
    ], dtype=float)


def _level(roll_deg: float, pitch_deg: float, order: str) -> np.ndarray:
    if order == "HPR":
        return _ry_neg(pitch_deg) @ _rx(roll_deg)
    if order == "HRP":
        return _rx(roll_deg) @ _ry_neg(pitch_deg)
    raise ValueError(f"unknown transform order {order!r}")


def instrument_to_ship(v_instrument: np.ndarray, mounting: MountingAngles, *,
                       order: str = "HPR") -> np.ndarray:
    """安装角把 instrument 系向量转到 ship 系（只用 pitch/roll 固定偏置）。"""
    return _level(mounting.roll_alignment_deg, mounting.pitch_alignment_deg, order) \
        @ np.asarray(v_instrument, dtype=float)


def ship_to_earth(v_ship: np.ndarray, heading_mag_deg: float, pitch_deg: float,
                  roll_deg: float, declination_deg: float, mounting: MountingAngles,
                  *, order: str = "HPR") -> np.ndarray:
    """ship 系向量经活姿态、磁偏角、heading alignment 转到 ENU。"""
    h_true = heading_mag_deg + declination_deg + mounting.heading_alignment_deg
    leveled = _level(roll_deg, pitch_deg, order) @ np.asarray(v_ship, dtype=float)
    return _heading_shipzdown_to_enu(h_true) @ leveled


def instrument_to_earth(v_instrument: np.ndarray, heading_mag_deg: float,
                        pitch_deg: float, roll_deg: float, declination_deg: float,
                        mounting: MountingAngles, *, order: str = "HPR") -> dict[str, Any]:
    """返回逐级向量，供复核界面展示 beam->instrument->ship->earth 全链路。"""
    v_instrument = np.asarray(v_instrument, dtype=float)
    v_ship = instrument_to_ship(v_instrument, mounting, order=order)
    v_earth = ship_to_earth(v_ship, heading_mag_deg, pitch_deg, roll_deg,
                            declination_deg, mounting, order=order)
    return {
        "instrument": [float(x) for x in v_instrument],
        "ship": [float(x) for x in v_ship],
        "earth": [float(x) for x in v_earth],
        "heading_true_deg": (heading_mag_deg + declination_deg
                             + mounting.heading_alignment_deg) % 360.0,
        "earth_datum": EARTH_DATUM,
    }
