"""Stable identity + content fingerprints.

Identities are deterministic so they survive export -> delete -> import
unchanged. They never depend on an autoincrement row id.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

PREFIX = {
    "src": "src", "rule": "rule", "deriv": "drv", "node": "node",
    "dval": "val", "cand": "cand", "event": "evt", "branch": "br",
    "decision": "dec", "case": "case",
}


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def fingerprint(parts) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(_canonical(part).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:12]


def make_id(kind: str, *parts: Any) -> str:
    return f"{PREFIX[kind]}_{fingerprint(parts)}"


def source_id(table: str, *key: Any) -> str:
    return make_id("src", table, *key)


def event_id(case_id: str, code: str, seq: int | str) -> str:
    return make_id("event", case_id, code, seq)


def node_id(deriv_id: str, ntype: str, ping_id: str | None,
            channel: str | None = None, cell_idx: int | None = None) -> str:
    return make_id("node", deriv_id, ntype, ping_id, channel, cell_idx)


def dval_id(*parts: Any) -> str:
    return make_id("dval", *parts)


def cand_id(deriv_id, ping_id, channel, cell_idx, method, beam_subset) -> str:
    return make_id("cand", deriv_id, ping_id, channel, cell_idx, method,
                   sorted(beam_subset))
