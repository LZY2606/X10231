"""派生指纹:坐标转换顺序、波束几何、磁偏角版本等全部进入指纹。"""
from __future__ import annotations

import hashlib
import json


def fp(payload) -> str:
    """对派生输入做稳定指纹(排序键、紧凑分隔、UTF-8)。"""
    text = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
