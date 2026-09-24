"""Stable, content-derived fingerprints.

A derivation fingerprint covers: transform order rule, beam geometry rule,
declination version, mount version, and the exact source identities that fed
the ping/cell.  Two different rule or version combinations therefore never
collapse to one value.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical(payload: Any) -> str:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def fingerprint(payload: Any) -> str:
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()[:16]


def derivation_fp(
    *,
    rule: str,
    geom_rule: str,
    declination_uid: str,
    mount_uid: str,
    sources: dict[str, str],
    extra: dict[str, Any] | None = None,
) -> str:
    body: dict[str, Any] = {
        "rule": rule,
        "geom": geom_rule,
        "decl": declination_uid,
        "mount": mount_uid,
        "sources": dict(sorted(sources.items())),
    }
    if extra:
        body["extra"] = dict(sorted(extra.items()))
    return fingerprint(body)
