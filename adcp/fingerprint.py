"""稳定身份与内容寻址派生指纹。

身份规则：
  - 源记录 stable_id 由命名空间 + 稳定坐标键确定性生成；同一逻辑记录
    在导出/删除/导入后身份不变。
  - 派生指纹 = 算子版本 + 全部输入内容（源/依赖派生/规则）的规范化哈希。
    坐标转换顺序、波束几何版本、磁偏角版本都进入指纹，因此规则修订
    会使依赖它的结果失效并沿依赖图传播。
"""
from __future__ import annotations

import hashlib
import json


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=str)


def digest(obj) -> str:
    return hashlib.sha1(_canonical(obj).encode("utf-8")).hexdigest()


def stable_id(namespace, *key_parts) -> str:
    """确定性稳定身份：与插入顺序、自增主键无关。"""
    raw = [namespace] + [str(k) for k in key_parts]
    return f"{namespace}:{hashlib.sha1('|'.join(raw).encode()).hexdigest()[:16]}"


def derivation_fingerprint(op, op_version, inputs, rules):
    """派生指纹。

    inputs: {role: [输入身份...]} 或 {role: 输入身份}
    rules:  进入指纹的规则内容（旋转顺序、几何、磁偏角版本等）
    """
    payload = {
        "op": op,
        "op_version": op_version,
        "inputs": inputs,
        "rules": rules,
    }
    return "fp:" + digest(payload)
