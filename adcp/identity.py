"""稳定身份：内容寻址 ID。

相同内容（相同规则版本、输入、参数、坐标基准）在任何数据库、
导出/删除/再导入之后都得到相同 ID 与相同来源关系。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False, default=str)


def content_id(prefix: str, payload: Any) -> str:
    digest = hashlib.blake2b(canonical_json(payload).encode("utf-8"),
                             digest_size=16).hexdigest()
    return f"{prefix}_{digest}"
