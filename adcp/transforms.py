"""逐级坐标旋转链: instrument -> ship -> earth。

红线语义: 坐标转换顺序是规则的一部分, 不同顺序产生不同指纹;
安装角修订与磁偏角修订分别作为独立来源版本参与, 不就地覆盖。

约定 (TRANSFORM_RULE = "rdi-HPR/v1"):
  instrument->ship:  R = R_x(roll') @ R_y(pitch')   (撇号表示含安装角)
                     可选顺序 "XY": R = R_y(pitch') @ R_x(roll')
  ship->earth:       按 RDI 惯例先绕 z 用真航向, 再 pitch, 再 roll:
                     R = R_x(roll) @ R_y(pitch) @ R_z(heading_true)
                     可选顺序 "ZYX_test" 用于案例包中的顺序反例。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .attitude import apply_mount_angles, magnetic_to_true

TRANSFORM_RULE = "rdi-HPR/v1"


def _rx(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    # x: 右舷为正; x-y 平面旋转, z-down 系下 roll 使 y->z
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]], dtype=float)


def _ry(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]], dtype=float)


def _rz(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=float)


@dataclass(frozen=True)
class TransformStep:
    """逐级变换证据: 界面逐级向量展示直接读这些记录。"""

    name: str
    matrix: Tuple[Tuple[float, ...], ...]
    input_frame: str
    output_frame: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "matrix": [list(row) for row in self.matrix],
            "input_frame": self.input_frame,
            "output_frame": self.output_frame,
        }


@dataclass(frozen=True)
class FullTransform:
    """一次完整的 instrument->earth 链与中间向量。"""

    instrument: Tuple[float, float, float]
    ship: Tuple[float, float, float]
    earth: Tuple[float, float, float]
    heading_true: float
    steps: Tuple[TransformStep, ...]
    order: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "instrument": list(self.instrument),
            "ship": list(self.ship),
            "earth": list(self.earth),
            "heading_true": self.heading_true,
            "order": self.order,
            "steps": [s.as_dict() for s in self.steps],
        }


def instrument_to_ship_matrix(pitch_deg: float, roll_deg: float,
                              order: str = "YX") -> np.ndarray:
    """instrument(x 右舷, y 船首, z 向下) -> ship, 可选旋转顺序。"""
    if order == "YX":
        return _rx(roll_deg) @ _ry(pitch_deg)
    if order == "XY":
        return _ry(pitch_deg) @ _rx(roll_deg)
    raise ValueError("unknown instrument->ship order %r" % order)


def ship_to_earth_matrix(heading_true_deg: float, pitch_deg: float,
                         roll_deg: float, order: str = "HPR") -> np.ndarray:
    """ship -> earth (东, 北, up 约定; 本工具 z 仍记向下, 界面明示基准)。"""
    if order == "HPR":
        return _rx(roll_deg) @ _ry(pitch_deg) @ _rz(heading_true_deg)
    if order == "PRH":
        # 反例顺序: 先姿态后航向, 与 RDI 标准不同, 案例包专门构造差异
        return _rz(heading_true_deg) @ _rx(roll_deg) @ _ry(pitch_deg)
    raise ValueError("unknown ship->earth order %r" % order)


def build_transform(instrument_uvw: Sequence[float],
                    heading_mag_deg: float, pitch_deg: float, roll_deg: float,
                    declination_deg: float,
                    mount: Optional[Dict[str, float]] = None,
                    is_order: str = "YX",
                    earth_order: str = "HPR") -> FullTransform:
    """组装完整链。mount 安装角先作用于姿态, 磁偏角只作用于航向。"""
    mount = mount or {}
    hp, pp, rp = apply_mount_angles(
        heading_mag_deg, pitch_deg, roll_deg,
        mount.get("heading", 0.0),
        mount.get("pitch", 0.0),
        mount.get("roll", 0.0),
    )
    heading_true = magnetic_to_true(hp, declination_deg)

    r_is = instrument_to_ship_matrix(pp, rp, order=is_order)
    r_se = ship_to_earth_matrix(heading_true, pp, rp, order=earth_order)
    vinst = np.array(instrument_uvw, dtype=float)
    vship = r_is @ vinst
    vearth = r_se @ vship

    steps = (
        TransformStep(
            name="instrument->ship (R=Rx(roll)Ry(pitch), order=%s)" % is_order,
            matrix=tuple(tuple(float(x) for x in row) for row in r_is),
            input_frame="instrument", output_frame="ship",
        ),
        TransformStep(
            name="magnetic->true (+declination %.3f deg)" % declination_deg,
            matrix=tuple(tuple(float(x) for x in row)
                         for row in np.eye(3)),
            input_frame="ship-heading:magnetic",
            output_frame="ship-heading:true",
        ),
        TransformStep(
            name="ship->earth (order=%s)" % earth_order,
            matrix=tuple(tuple(float(x) for x in row) for row in r_se),
            input_frame="ship", output_frame="earth",
        ),
    )
    return FullTransform(
        instrument=tuple(float(x) for x in vinst),
        ship=tuple(float(x) for x in vship),
        earth=tuple(float(x) for x in vearth),
        heading_true=float(heading_true),
        steps=steps,
        order="is:%s;earth:%s" % (is_order, earth_order),
    )
