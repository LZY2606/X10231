"""质量层级：beam 测量、correlation、amplitude、坏 beam 标志分开记录、分开判定。

判定不“为了画图而丢掉”任何值：被排除的 cell 保留在原位置并附带结构化原因，
垂向剖面只绘制 accepted cell，但被排除 cell 的证据可在复核界面读回。
阈值属于规则集（Rules），规则版本进入派生指纹。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .units import Quantity


@dataclass(frozen=True)
class QualityRules:
    min_correlation: int = 64        # correlation count 阈值
    min_amplitude_counts: int = 40   # amplitude counts 阈值
    max_error_velocity_mps: float = 0.50
    min_depth_m: float = 0.30        # 小于该深度的 cell 视为无效（近场）
    max_depth_m: float = 30.0        # 超过配置量程视为无效

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_correlation": self.min_correlation,
            "min_amplitude_counts": self.min_amplitude_counts,
            "max_error_velocity_mps": self.max_error_velocity_mps,
            "min_depth_m": self.min_depth_m,
            "max_depth_m": self.max_depth_m,
        }


@dataclass(frozen=True)
class BeamQuality:
    beam_index: int
    bad_flag: bool
    correlation_ok: bool
    amplitude_ok: bool
    velocity_ok: bool
    reasons: tuple[str, ...]

    @property
    def good(self) -> bool:
        return not self.reasons

    def as_dict(self) -> dict[str, Any]:
        return {
            "beam_index": self.beam_index,
            "bad_flag": self.bad_flag,
            "correlation_ok": self.correlation_ok,
            "amplitude_ok": self.amplitude_ok,
            "velocity_ok": self.velocity_ok,
            "reasons": list(self.reasons),
            "good": self.good,
        }


def judge_beam(beam_index: int, velocity: Quantity, correlation_counts: int | None,
               amplitude_counts: int | None, bad_flag: bool,
               rules: QualityRules) -> BeamQuality:
    reasons: list[str] = []
    if bad_flag:
        reasons.append("bad-beam-flag")
    corr_ok = correlation_counts is not None and correlation_counts >= rules.min_correlation
    if not corr_ok:
        reasons.append("low-correlation")
    amp_ok = amplitude_counts is not None and amplitude_counts >= rules.min_amplitude_counts
    if not amp_ok:
        reasons.append("low-amplitude")
    vel_ok = velocity.convertible
    if not vel_ok:
        reasons.append(f"unconvertible-velocity:{velocity.diagnostic or 'unknown'}")
    return BeamQuality(beam_index, bool(bad_flag), corr_ok, amp_ok, vel_ok, tuple(reasons))


@dataclass(frozen=True)
class CellQuality:
    ping: int
    cell: int
    depth_m: float | None
    accepted: bool
    beam_quality: tuple[BeamQuality, ...]
    reason_codes: tuple[str, ...]
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def good_beam_count(self) -> int:
        return sum(1 for b in self.beam_quality if b.good)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ping": self.ping,
            "cell": self.cell,
            "depth_m": self.depth_m,
            "accepted": self.accepted,
            "good_beam_count": self.good_beam_count,
            "reason_codes": list(self.reason_codes),
            "beams": [b.as_dict() for b in self.beam_quality],
            "detail": self.detail,
        }


def judge_cell(ping: int, cell: int, depth: Quantity | None,
               beam_quality: list[BeamQuality], rules: QualityRules,
               error_velocity_mps: float | None = None,
               explicit_invalid: bool = False) -> CellQuality:
    """综合 beam 层质量、深度有效性与 error velocity，决定 cell 是否进入剖面。"""
    reasons: list[str] = []
    detail: dict[str, Any] = {}
    if explicit_invalid:
        reasons.append("explicit-invalid-cell")
    depth_m = None
    if depth is None:
        reasons.append("missing-depth")
    elif not depth.convertible:
        reasons.append(f"unconvertible-depth:{depth.diagnostic or 'unknown'}")
    else:
        depth_m = depth.value
        if depth_m < rules.min_depth_m:
            reasons.append("depth-below-min")
            detail["depth_m"] = depth_m
        elif depth_m > rules.max_depth_m:
            reasons.append("depth-above-max")
            detail["depth_m"] = depth_m
    good = sum(1 for b in beam_quality if b.good)
    if good < 3:
        reasons.append(f"fewer-than-3-good-beams:{good}")
    elif good == 3:
        reasons.append("three-beam-candidate")  # 不是硬排除，但必须显式裁定
    if error_velocity_mps is not None and abs(error_velocity_mps) > rules.max_error_velocity_mps:
        reasons.append("error-velocity-exceeded")
        detail["error_velocity_mps"] = error_velocity_mps
    # accepted：深度有效、四波束可用且无其它原因；三波束 cell 不自动进入剖面。
    accepted = depth_m is not None and good == 4 and not reasons
    return CellQuality(ping, cell, depth_m, accepted, tuple(beam_quality),
                       tuple(reasons), detail)
