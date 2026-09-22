"""派生指纹。

对依赖内容做确定性（canonical）序列化后计算 SHA-256。任何输入、规则、
顺序或版本变化都会改变指纹：

* 波束几何版本（BeamGeometry.fingerprint_input）
* 坐标变换顺序（轴序字符串）
* 磁偏角配置版本
* 上游节点指纹与槽位键

因此“坐标转换顺序、波束几何和磁偏角版本都进入派生指纹”。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=_default)


def _default(o: Any) -> Any:
    if hasattr(o, "fingerprint_input"):
        return o.fingerprint_input()
    if hasattr(o, "as_dict"):
        return o.as_dict()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"无法规范化类型 {type(o)!r}")


def fingerprint(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def node_id(kind: str, payload_fingerprint_input: Any) -> str:
    """节点稳定身份：kind + 规范化内容。"""
    material = {"kind": kind, "in": payload_fingerprint_input}
    return f"{kind}_{fingerprint(material)[:16]}"
