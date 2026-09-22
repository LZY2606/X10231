"""Stable content fingerprints.

Canonical JSON (sorted keys, compact separators, deterministic float
rendering) hashed with SHA-256 gives every record and derivation an identity
that survives export/delete/re-import.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict


def _canonical(obj: Any) -> str:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_default,
    )


def _default(obj: Any) -> Any:
    if hasattr(obj, "to_fingerprint"):
        return obj.to_fingerprint()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    raise TypeError("object of type %r is not fingerprintable" % type(obj))


def canonical_json(obj: Any) -> str:
    return _canonical(obj)


def digest(obj: Any, prefix: str = "sha256") -> str:
    blob = _canonical(obj).encode("utf-8")
    return "%s:%s" % (prefix, hashlib.sha256(blob).hexdigest())


def derive_fingerprint(
    kind: str,
    inputs: Dict[str, Any],
    rule_version: str,
    geometry_fp: Any = None,
    mounting_fp: Any = None,
    declination_fp: Any = None,
    transform_order: str = None,
) -> str:
    payload = {
        "kind": kind,
        "inputs": inputs,
        "rule_version": rule_version,
        "geometry": geometry_fp,
        "mounting": mounting_fp,
        "declination": declination_fp,
        "transform_order": transform_order,
    }
    return digest(payload)
