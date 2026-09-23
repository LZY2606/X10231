"""姿态序列处理：heading 展开与受限插值。

语义红线：姿态插值不跨缺口。相邻姿态样本间隔超过 max_gap_s 时，
缺口内的 ping 不做插值，返回诊断而不是编造姿态。
"""
from __future__ import annotations


def unwrap_degrees(headings: list[float]) -> list[float]:
    """把 [0,360) 的 heading 序列展开为连续序列（跨越 360° 不回跳）。"""
    out: list[float] = []
    offset = 0.0
    prev = None
    for heading in headings:
        if prev is not None:
            diff = heading + offset - prev
            if diff > 180.0:
                offset -= 360.0
            elif diff < -180.0:
                offset += 360.0
        value = heading + offset
        out.append(value)
        prev = value
    return out


def wrap_360(heading: float) -> float:
    return heading % 360.0


def interpolate_attitude(samples: list[dict], t: float, max_gap_s: float):
    """在时刻 t 插值姿态。

    samples: 按时间升序的 [{"t":..,"heading_deg":..,"pitch_deg":..,"roll_deg":..}]
    返回 (attitude_or_None, diagnostic_or_None, meta)。
    meta 记录是否发生 360° 展开、使用了哪两个样本，供事件与指纹使用。
    """
    meta = {"unwrapped": False, "bracket": None}
    if not samples:
        return None, "attitude_missing", meta
    if t < samples[0]["t"] or t > samples[-1]["t"]:
        return None, "attitude_out_of_range", meta

    lower = None
    upper = None
    for index, sample in enumerate(samples):
        if sample["t"] <= t:
            lower = index
        if sample["t"] >= t:
            upper = index
            break
    if lower is None:
        lower = upper
    if upper is None:
        upper = lower
    s0 = samples[lower]
    s1 = samples[upper]
    meta["bracket"] = [s0["t"], s1["t"]]

    if s1["t"] - s0["t"] > max_gap_s:
        return None, f"attitude_gap:{s1['t'] - s0['t']:.3f}s>{max_gap_s}s", meta

    headings = unwrap_degrees([s0["heading_deg"], s1["heading_deg"]])
    if headings[0] != s0["heading_deg"] or headings[1] != s1["heading_deg"]:
        meta["unwrapped"] = True

    span = s1["t"] - s0["t"]
    ratio = 0.0 if span == 0 else (t - s0["t"]) / span
    heading = headings[0] + (headings[1] - headings[0]) * ratio
    pitch = s0["pitch_deg"] + (s1["pitch_deg"] - s0["pitch_deg"]) * ratio
    roll = s0["roll_deg"] + (s1["roll_deg"] - s0["roll_deg"]) * ratio
    return {
        "heading_deg": wrap_360(heading),
        "pitch_deg": pitch,
        "roll_deg": roll,
    }, None, meta
