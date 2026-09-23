"""ADCP 坐标变换核心规则。

坐标链(固定顺序, 进入派生指纹): beam > instrument > ship > earth
旋转顺序: Rz(heading) @ Ry(pitch) @ Rx(roll), 随后 Rz(declination) 磁北->真北。
派生层速度单位 m/s, 角度单位度。原始记录保留原始单位(如 mm/s), 仅在派生层规范化。
"""
from __future__ import annotations

import math

TRANSFORM_ORDER = "beam>instrument>ship>earth"
ROTATION_ORDER = "Rz(heading)*Ry(pitch)*Rx(roll)"
GAP_THRESHOLD_S = 2.5
CORRELATION_MIN = 80.0
AMPLITUDE_WARN = 20.0

RULE_VERSIONS = {
    "normalize": "norm-v1",
    "quality": "qual-v1",
    "attitude": "att-v1",
    "beam2instrument": "b2i-v1",
    "instrument2ship": "i2s-v1",
    "ship2earth": "s2e-v1",
    "water": "wtr-v1",
    "profile": "prof-v1",
}


def wrap360(deg: float) -> float:
    return deg % 360.0


def interpolate_attitude(samples, t, gap_threshold=GAP_THRESHOLD_S):
    """在姿态样本间线性插值。

    samples: [(t, heading_deg, pitch_deg, roll_deg)] 按 t 排序。
    返回 (result, (i0, i1)), result["status"] 为 "ok" 或 "no_attitude"。
    规则: heading 先解卷绕再插值(跨越 360/0 不取均值);
          相邻样本间隔超过 gap_threshold 时不在缺口上插值。
    """
    if not samples:
        return {"status": "no_attitude", "reason": "no_samples"}, (None, None)
    ts = [s[0] for s in samples]
    if t <= ts[0]:
        i0 = i1 = 0
    elif t >= ts[-1]:
        i0 = i1 = len(ts) - 1
    else:
        i0 = 0
        while ts[i0 + 1] < t:
            i0 += 1
        i1 = i0 + 1
    t0, h0, p0, r0 = samples[i0]
    t1, h1, p1, r1 = samples[i1]
    if i0 != i1 and t1 - t0 > gap_threshold and t not in (t0, t1):
        return {"status": "no_attitude", "reason": "gap", "gap": [t0, t1]}, (i0, i1)
    w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
    dh = ((h1 - h0 + 180.0) % 360.0) - 180.0
    return {
        "status": "ok",
        "heading_deg": wrap360(h0 + w * dh),
        "pitch_deg": p0 + w * (p1 - p0),
        "roll_deg": r0 + w * (r1 - r0),
    }, (i0, i1)


def beam_to_instrument4(b1, b2, b3, b4, theta_deg):
    """四波束 Janus -> instrument 坐标。波束方位角 (j-1)*90 度, 倾角 theta。

    返回 (u, v, w, e): e 为误差速度(四波束冗余检验量)。
    """
    s = math.sin(math.radians(theta_deg))
    c = math.cos(math.radians(theta_deg))
    u = (b1 - b3) / (2.0 * s)
    v = (b2 - b4) / (2.0 * s)
    w = (b1 + b2 + b3 + b4) / (4.0 * c)
    e = (b1 - b2 + b3 - b4) / (4.0 * c)
    return {"u": u, "v": v, "w": w, "e": e}


def _beam_row(beam_index, theta_deg):
    az = math.radians((beam_index - 1) * 90.0)
    s = math.sin(math.radians(theta_deg))
    c = math.cos(math.radians(theta_deg))
    return (s * math.cos(az), s * math.sin(az), c)


def beam_to_instrument3_candidates(vels, beams, theta_deg):
    """三波束多解: 一个波束失效时误差速度不可知, 保留全部合法候选。

    candA: 最小二乘直解(等价于假设误差速度 e=0)。
    candB: 假设垂向流速 w=0, 由三个波束最小二乘解 u, v。
    两个候选永不合并为单一值。
    """
    rows = [_beam_row(j, theta_deg) for j in beams]
    y = list(vels)
    sol = _solve3([list(r) for r in rows], y)
    out = {
        "candA": {
            "u": sol[0], "v": sol[1], "w": sol[2], "e": None,
            "assumption": "error_velocity=0 (3-beam least squares)",
        }
    }
    ata = [[0.0, 0.0], [0.0, 0.0]]
    aty = [0.0, 0.0]
    for (sx, sy, _cz), b in zip(rows, y):
        ata[0][0] += sx * sx
        ata[0][1] += sx * sy
        ata[1][0] += sx * sy
        ata[1][1] += sy * sy
        aty[0] += sx * b
        aty[1] += sy * b
    det = ata[0][0] * ata[1][1] - ata[0][1] * ata[1][0]
    u = (aty[0] * ata[1][1] - aty[1] * ata[0][1]) / det
    v = (ata[0][0] * aty[1] - ata[1][0] * aty[0]) / det
    out["candB"] = {
        "u": u, "v": v, "w": 0.0, "e": None,
        "assumption": "w=0 (no vertical motion)",
    }
    return out


def _solve3(a, y):
    (a11, a12, a13), (a21, a22, a23), (a31, a32, a33) = a
    b1, b2, b3 = y
    det = (a11 * (a22 * a33 - a23 * a32)
           - a12 * (a21 * a33 - a23 * a31)
           + a13 * (a21 * a32 - a22 * a31))
    dx = (b1 * (a22 * a33 - a23 * a32)
          - a12 * (b2 * a33 - a23 * b3)
          + a13 * (b2 * a32 - a22 * b3))
    dy = (a11 * (b2 * a33 - a23 * b3)
          - b1 * (a21 * a33 - a23 * a31)
          + a13 * (a21 * b3 - b2 * a31))
    dz = (a11 * (a22 * b3 - b2 * a32)
          - a12 * (a21 * b3 - b2 * a31)
          + b1 * (a21 * a32 - a22 * a31))
    return (dx / det, dy / det, dz / det)


def rotate_x(v, deg):
    x, y, z = v
    a = math.radians(deg)
    ca, sa = math.cos(a), math.sin(a)
    return (x, ca * y - sa * z, sa * y + ca * z)


def rotate_y(v, deg):
    x, y, z = v
    a = math.radians(deg)
    ca, sa = math.cos(a), math.sin(a)
    return (ca * x + sa * z, y, -sa * x + ca * z)


def rotate_z(v, deg):
    x, y, z = v
    a = math.radians(deg)
    ca, sa = math.cos(a), math.sin(a)
    return (ca * x - sa * y, sa * x + ca * y, z)


def instrument_to_ship(v, mounting_deg, extra_yaw_deg=0.0):
    """instrument -> ship: 绕垂轴旋转安装角(及规则变体的附加偏航)。"""
    return rotate_z(v, mounting_deg + extra_yaw_deg)


def ship_to_earth(v, heading_deg, pitch_deg, roll_deg):
    """ship -> earth(磁北基准): Rz(heading) @ Ry(pitch) @ Rx(roll)。"""
    v = rotate_x(v, roll_deg)
    v = rotate_y(v, pitch_deg)
    v = rotate_z(v, heading_deg)
    return v


def apply_declination(v, declination_deg):
    """磁北 -> 真北: 绕垂轴旋转磁偏角(东正)。"""
    return rotate_z(v, declination_deg)
