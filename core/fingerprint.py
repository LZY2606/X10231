"""派生指纹：对“输入内容 + 规则版本”做规范化哈希。

任何派生结果的指纹都显式包含：
- 直接输入的值与单位/基准；
- 上游节点指纹（传递依赖）；
- 波束几何版本（beam_angle、config_code）；
- 磁偏角来源版本；
- 安装角版本；
- 坐标变换顺序（transform_order）；
- 质量阈值规则版本。

指纹是派生值身份的一部分：同值但依据不同 -> 不同身份；依赖图据此判断
某结果为何仍有效（指纹未变）或为何需要重算（某个输入指纹变化）。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      default=_default)


def _default(obj: Any) -> Any:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, (tuple, set)):
        return sorted(obj, key=lambda x: canonical(x))
    raise TypeError(f"not fingerprintable: {type(obj)!r}")


def fingerprint(parts: dict[str, Any]) -> str:
    """parts 的键命名即“依据类型”（geometry/declination/mounting/order/...）。"""
    return hashlib.sha256(canonical(parts).encode("utf-8")).hexdigest()


def short(fp: str, n: int = 12) -> str:
    return fp[:n]
