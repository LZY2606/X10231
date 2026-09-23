"""派生指纹：把坐标转换顺序、波束几何版本、磁偏角版本、安装角版本、
源记录版本与质量阈值组合成稳定指纹。任何输入变化都改变指纹。"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def derive_fingerprint(kind: str, payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, default=str)
    digest = hashlib.sha256(f"{kind}|{canonical}".encode("utf-8")).hexdigest()
    return f"fp-{digest[:20]}"
