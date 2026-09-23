"""派生指纹。

红线: 坐标转换顺序、波束几何、磁偏角版本都必须进入派生指纹;
此外姿态规则、gap 容差、输入来源版本与 (三束场景下) 候选假设也进入指纹。
指纹不同 => 结果必然重算; 指纹相同 => 结果仍有效 (reused)。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping, Sequence

from .attitude import ATTITUDE_RULE
from .geometry import BeamGeometry
from .transforms import TRANSFORM_RULE

FINGERPRINT_RULE = "derived-fingerprint/v1"


def canonical_json(obj: Any) -> str:
    """确定性 JSON: 键排序, 不允许 ASCII 转义差异。"""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=_default)


def _default(o: Any) -> Any:
    if hasattr(o, "as_dict"):
        return o.as_dict()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError("not fingerprint-serializable: %r" % type(o))


def content_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def derive_fingerprint(*,
                       node_type: str,
                       inputs: Mapping[str, Any],
                       geom: BeamGeometry,
                       declination_version: str,
                       mount_version: str,
                       attitude_version: str = ATTITUDE_RULE,
                       transform_rule: str = TRANSFORM_RULE,
                       is_order: str = "YX",
                       earth_order: str = "HPR",
                       gap_tolerance: float = 2.0,
                       candidate_assumption: str = "",
                       extra: Mapping[str, Any] = None) -> str:
    payload: Dict[str, Any] = {
        "fingerprint_rule": FINGERPRINT_RULE,
        "node_type": node_type,
        "inputs": _hashable(inputs),
        "geometry": geom.as_dict(),
        "declination_version": declination_version,
        "mount_version": mount_version,
        "attitude_rule": attitude_version,
        "transform_rule": transform_rule,
        "is_order": is_order,
        "earth_order": earth_order,
        "gap_tolerance": gap_tolerance,
        "candidate_assumption": candidate_assumption,
    }
    if extra:
        payload["extra"] = _hashable(extra)
    return content_hash(payload)[:16]


def _hashable(obj: Any) -> Any:
    if hasattr(obj, "as_dict"):
        return obj.as_dict()
    if isinstance(obj, dict):
        return {str(k): _hashable(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_hashable(v) for v in obj]
    return obj


def fingerprint_basis(*,
                      geom: BeamGeometry,
                      declination_version: str,
                      mount_version: str,
                      is_order: str = "YX",
                      earth_order: str = "HPR",
                      gap_tolerance: float = 2.0) -> Dict[str, Any]:
    """供证据界面展示"这个值由哪些版本决定"。"""
    return {
        "geometry": geom.as_dict(),
        "declination_version": declination_version,
        "mount_version": mount_version,
        "attitude_rule": ATTITUDE_RULE,
        "transform_rule": TRANSFORM_RULE,
        "is_order": is_order,
        "earth_order": earth_order,
        "gap_tolerance": gap_tolerance,
    }
