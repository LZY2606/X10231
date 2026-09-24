"""派生指纹：坐标转换顺序、波束几何、磁偏角版本等都进入指纹。"""
from __future__ import annotations

import hashlib
import json


def fingerprint(payload) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def short(fp: str) -> str:
    return fp[:8]
