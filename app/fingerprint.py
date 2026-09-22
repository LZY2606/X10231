"""Content fingerprints for derived nodes and derivation sets.

Everything that can change a derived value enters the fingerprint:
  * rule code version
  * beam geometry (beam angle, azimuths, orientation, thresholds)
  * coordinate transform order
  * the specific source *versions* the node depends on (mount, declination,
    attitude, beams/bottom track for the ping/bin in question)

Fingerprints are stable across export/delete/import because they are content
hashes, not row ids.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, Mapping, Tuple

RULE_VERSION = "adcp-rules-2026-09-23"


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def content_id(parts: Iterable[Any]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(_canon(p).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:16]


def geometry_fingerprint(cfg_dict: Mapping[str, Any]) -> str:
    return content_id(["geometry", RULE_VERSION, dict(cfg_dict)])


def node_fingerprint(
    kind: str,
    location: Tuple[str, str],
    geometry_fp: str,
    transform_order: str,
    deps: Mapping[str, str],
    extra: Mapping[str, Any] = None,
) -> str:
    payload = {
        "rule_version": RULE_VERSION,
        "kind": kind,
        "location": list(location),
        "geometry_fp": geometry_fp,
        "transform_order": transform_order,
        "deps": dict(sorted(deps.items())),
    }
    if extra:
        payload["extra"] = dict(sorted(extra.items()))
    return content_id([payload])


def set_fingerprint(nodes: Iterable[Mapping[str, Any]], tag: str = "") -> str:
    blob = [(n["node_key"], n["fingerprint"]) for n in nodes]
    blob.sort()
    return content_id(["derivation-set", RULE_VERSION, blob, tag])


def fingerprint(parts: Iterable[Any]) -> str:
    return content_id(parts)
