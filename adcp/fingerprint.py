"""Derived-value fingerprints.

Every derived node carries a fingerprint that commits to the *rule context*
that produced it: transform order, beam geometry version, magnetic
declination version, mounting-angle version, QC version and the digests of
its raw inputs.  A change to any of these invalidates exactly the dependent
nodes -- and lets us explain why a node is still valid or must be recomputed.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, Iterable, Tuple


def canonical(parts: Dict[str, object]) -> str:
    return json.dumps(parts, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def fingerprint(parts: Dict[str, object]) -> str:
    return hashlib.sha256(canonical(parts).encode("utf-8")).hexdigest()[:16]


def rule_context(
    transform_order: str,
    beam_geometry_version: str,
    declination_version: str,
    mounting_angle_version: str,
    qc_version: str,
    rule_version: str,
) -> Dict[str, str]:
    """The rule half of every fingerprint (config values + versions)."""
    return {
        "transform_order": transform_order,
        "beam_geometry_version": beam_geometry_version,
        "declination_version": declination_version,
        "mounting_angle_version": mounting_angle_version,
        "qc_version": qc_version,
        "rule_version": rule_version,
    }


def combine(rule: Dict[str, str], inputs: Iterable[Tuple[str, str]]) -> str:
    """Fingerprint from rule context plus (node_id, digest) input pairs."""
    return fingerprint({"rule": rule, "inputs": sorted(inputs)})
