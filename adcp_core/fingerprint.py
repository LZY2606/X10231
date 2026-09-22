"""派生指纹：规范化 JSON + sha256。

坐标转换顺序、波束几何、磁偏角版本等所有影响派生结果的输入都参与指纹，
因此任一上游变化都会得到不同指纹，依赖图据此判断结果是否仍有效。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def canonical(obj: Any) -> str:
    return json.dumps(_normalize(obj), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def fingerprint(obj: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def _normalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _normalize(obj[k]) for k in sorted(obj.keys(), key=str)}
    if isinstance(obj, (list, tuple)):
        return [_normalize(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return _normalize(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj
