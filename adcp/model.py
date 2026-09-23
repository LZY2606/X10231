"""原始事实模型：测量、姿态、磁偏角、安装角各自独立记录。"""
from __future__ import annotations

import hashlib
import json
from uuid import uuid4

FACT_TYPES = (
    "beam",          # 波束径向速度 (ping,cell,beam)
    "bt_beam",       # bottom track 径向速度 (ping,beam)
    "correlation",   # (ping,cell,beam)
    "bt_correlation",
    "amplitude",     # (ping,cell,beam)
    "bt_amplitude",
    "badflag",       # (ping,cell,beam) 0/1
    "bt_badflag",
    "attitude",      # (time_s) heading/pitch/roll 在同一行采样但三量分开字段
    "heading", "pitch", "roll",  # 也可三量完全分行记录
    "declination",   # 独立磁偏角来源，含 model/epoch
    "mount",         # 独立安装角来源
    "geoconfig",     # 换能器配置：beam angle/convex/blanking/cell depths...
    "celldepth",     # (ping,cell) 深度
    "transform_order",
    "qconfig",
    "decision",      # 人工 accept/reject 某候选
)


def new_id(prefix: str = "f") -> str:
    return f"{prefix}_{uuid4().hex[:16]}"


def fact_hash(fact: dict) -> str:
    """事实内容哈希：版本变化即变（指纹的输入）。"""
    payload = {k: fact.get(k) for k in (
        "ftype", "ping", "cell", "beam", "time_s",
        "raw", "raw_unit", "datum", "payload",
    )}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def make_fact(ftype: str, raw=None, raw_unit=None, datum=None,
              ping=None, cell=None, beam=None, time_s=None,
              payload=None, fact_id=None, supersedes=None, **extra) -> dict:
    fact = {
        "fact_id": fact_id or new_id(ftype[:2]),
        "ftype": ftype,
        "ping": ping, "cell": cell, "beam": beam, "time_s": time_s,
        "raw": raw, "raw_unit": raw_unit, "datum": datum,
        "payload": payload or {},
        "supersedes": supersedes,
    }
    fact.update(extra)
    fact["fhash"] = fact_hash(fact)
    return fact
