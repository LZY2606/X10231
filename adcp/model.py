"""ADCP 领域数据模型。

关键设计：八类“直接来源”在模型层就是互相独立的记录类型，
波束测量、correlation、amplitude、坏 beam 标志分别有独立身份与版本，
任何一类的改动都不会覆盖其他来源。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# 直接来源种类（八类 + bottom track）。稳定字符串，进入身份与指纹。
SRC_TRANSDUCER = "transducer_config"
SRC_DECLINATION = "declination"
SRC_MOUNT = "mount_angle"
SRC_BEAM = "beam_velocity"
SRC_CORRELATION = "correlation"
SRC_AMPLITUDE = "amplitude"
SRC_BADBEAM = "bad_beam_flag"
SRC_ATTITUDE = "attitude"
SRC_BOTTOMTRACK = "bottom_track"

SOURCE_KINDS = (
    SRC_TRANSDUCER, SRC_DECLINATION, SRC_MOUNT,
    SRC_BEAM, SRC_CORRELATION, SRC_AMPLITUDE, SRC_BADBEAM,
    SRC_ATTITUDE, SRC_BOTTOMTRACK,
)

BEAMS = (1, 2, 3, 4)

# 候选层级
TIER_STRICT = "strict"   # 仅使用高质量 beam（corr/ampl 阈值全通过）
TIER_WEAK = "weak"       # 允许退化质量 beam（仅确认可用即纳入）
# 三波束/四波束
SOL_FOUR = "four_beam"
SOL_THREE = "three_beam"

# 变换阶段与基准
STAGE_QUALITY = "beam_quality"
STAGE_INSTRUMENT = "instrument"
STAGE_SHIP = "ship"
STAGE_EARTH = "earth"
STAGE_EXCLUDED = "excluded_cell"
STAGE_PROFILE = "vertical_profile"

FRAME_INSTRUMENT = "instrument_FSD"
FRAME_SHIP = "ship_FSD"
FRAME_EARTH = "earth_ENU"


@dataclass(frozen=True)
class TransducerConfig:
    config_id: str
    beam_angle_deg: float          # 波束偏离垂直方向夹角 θ
    geometry: str                  # "convex_janus"（标准 RDI 四波束）
    convex: bool                   # True=convex（波束向外斜）
    frequency_hz: int
    cell_depth_m: tuple            # 每个 cell 的深度（m，正值）
    blanking_cells: tuple          # 初始盲区 cell 序号（0-based）
    rule_version: str = "adcp-rules-1.0.0"


@dataclass(frozen=True)
class Declination:
    declination_id: str
    value_deg: float               # 东偏为正；磁北→真北
    version: str                   # 磁偏角版本，例如 "WMM-2020"


@dataclass(frozen=True)
class MountAngle:
    mount_id: str
    angle_deg: float               # 安装角：仪器 x 轴相对船首
    version: str                   # 安装角修订版本


@dataclass(frozen=True)
class Ping:
    ping_id: str
    timestamp: float               # Unix epoch 秒
    velocities: tuple              # 4 个 Quantity（mm/s，方向沿各 beam）
    correlations: tuple            # 4 个 Quantity（%）
    amplitudes: tuple              # 4 个 Quantity（count）
    bad_flags: tuple               # 4 个 bool
    cell_depths: tuple             # 本 ping 每个 cell 的深度 Quantity（m）
    valid_cells: tuple             # bool：该 cell 是否有测量


@dataclass(frozen=True)
class AttitudeSample:
    attitude_id: str
    ping_id: str
    timestamp: float
    heading: Any                   # Quantity（magnetic_deg）
    pitch: Any                     # Quantity（deg）
    roll: Any                      # Quantity（deg）


@dataclass(frozen=True)
class BottomTrack:
    bt_id: str
    ping_id: str
    timestamp: float
    range_m: tuple                 # 4 个 Quantity：各 beam 到底距离
    velocities: tuple              # 4 个 Quantity：bottom track beam velocity
    correlations: tuple
    bad_flags: tuple


@dataclass(frozen=True)
class BeamAssessment:
    beam: int
    flagged: bool
    corr_ok_strict: bool
    ampl_ok_strict: bool
    corr_ok_weak: bool
    ampl_ok_weak: bool
    usable_strict: bool
    usable_weak: bool
    reasons: tuple


@dataclass(frozen=True)
class Candidate:
    """一次合法的波束组合解。三波束/四波束严格分开，绝不混合。"""

    kind: str                      # SOL_FOUR / SOL_THREE
    tier: str                      # TIER_STRICT / TIER_WEAK
    beams: tuple                   # 使用的 beam 序号
    dropped: tuple                 # 被舍弃的 beam 序号
    assumption: str                # 三波束：error velocity = 0 等假设
    instrument: tuple              # (F, S, D, e) SI（m/s）；四波束 e 为误差速度
    quality_ok: bool               # 是否满足该 tier 的质量要求


@dataclass(frozen=True)
class TransformResult:
    """单个候选从 instrument 逐级变换的结果（逐级保留，不覆盖）。"""

    instrument_vec: tuple          # (F,S,D) m/s
    ship_vec: tuple                # (F,S,D) m/s
    earth_vec: tuple               # (E,N,U) m/s
    true_heading_rad: float
    pitch_rad: float
    roll_rad: float
    order_code: str                # 进入指纹的变换顺序码
    mount_deg: float
    decl_version: str


@dataclass(frozen=True)
class CellResult:
    ping_id: str
    cell: int
    depth_m: Optional[float]
    candidates: tuple              # Candidate
    transforms: tuple              # 与 candidates 对齐的 TransformResult
    assessments: tuple             # 4 个 BeamAssessment
    excluded: bool
    exclude_reasons: tuple
    diagnostics: tuple             # 无法转换值的诊断（留在原位）


@dataclass(frozen=True)
class Fingerprint:
    """派生指纹：坐标转换顺序、波束几何与磁偏角版本都必须进入。"""

    rule_version: str
    geometry: str
    beam_angle_deg: float
    convex: bool
    transform_order: str
    declination_version: str
    mount_version: str
    inputs: tuple
    value: str = ""


@dataclass(frozen=True)
class Decision:
    decision_id: str
    event_id: str
    base_version: str
    kind: str                      # accept / reject / parameter_adjust / source_correction / conflict
    target_ref: str
    payload: dict
    parent_event_id: Optional[str] = None
