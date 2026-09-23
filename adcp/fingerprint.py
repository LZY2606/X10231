"""Stable derived fingerprints.

A node fingerprint covers: rule identity/version, the fingerprints of its
dependency nodes, and the *identifying* parameters of the rule — including
the beam geometry id, the magnetic-declination record version and the
coordinate transform order.  Thus changing geometry, declination or order
changes the fingerprint even when raw numbers are identical.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

RULE_SCHEMA_VERSION = "adcp-rule-v3"


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=_default)


def _default(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (set, frozenset, tuple)):
        return sorted(obj, key=str)
    raise TypeError(f"not canonicalisable: {type(obj)!r}")


def fingerprint(rule: str, rule_version: str,
                deps: Mapping[str, str] | Sequence[str],
                params: Mapping[str, Any] | None = None,
                basis: Sequence[str] | None = None) -> str:
    if isinstance(deps, Mapping):
        dep_part = canonical(dict(sorted(deps.items())))
    else:
        dep_part = canonical(sorted(deps))
    payload = {
        "schema": RULE_SCHEMA_VERSION,
        "rule": rule,
        "rule_version": rule_version,
        "deps": dep_part,
        "params": canonical(dict(sorted((params or {}).items()))),
    }
    digest = hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()
    return digest[:16]


def source_fingerprint(kind: str, canonical_record: Mapping[str, Any]) -> str:
    payload = {"kind": kind, "record": canonical_record}
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()[:16]
