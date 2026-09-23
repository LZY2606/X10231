"""派生指纹:坐标转换顺序、波束几何、磁偏角版本等都按阶段进入指纹。

指纹按阶段划定作用域,使安装角修订只让 ship/earth 失效,
而 beam->instrument 结果可以证明"仍然有效"。
"""
from __future__ import annotations

import hashlib
import json

STAGE_SCOPE = {
    "quality": ["corr_threshold", "amp_threshold"],
    "attitude": ["gap_threshold_s"],
    "instrument": ["beam_geometry", "three_beam_policy", "error_velocity_candidates"],
    "ship": ["beam_geometry", "three_beam_policy", "error_velocity_candidates",
             "mounting_angle_deg", "mounting_version"],
    "earth": ["beam_geometry", "three_beam_policy", "error_velocity_candidates",
              "mounting_angle_deg", "mounting_version",
              "declination_deg", "declination_version",
              "transform_order", "gap_threshold_s"],
}


def stage_fingerprint(stage, params):
    if stage not in STAGE_SCOPE:
        raise ValueError("未知阶段: %r" % (stage,))
    scoped = {k: params.get(k) for k in STAGE_SCOPE[stage]}
    blob = json.dumps({"stage": stage, "params": scoped}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
