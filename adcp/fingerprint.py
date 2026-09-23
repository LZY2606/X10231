"""Stable fingerprints for source data, rule versions and derived results."""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical(obj: Any) -> str:
    """Deterministic JSON encoding (sorted keys, no whitespace surprises)."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      default=_default)


def _default(o: Any) -> Any:
    if hasattr(o, "as_dict"):
        return o.as_dict()
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)


def content_hash(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def short_hash(obj: Any, n: int = 12) -> str:
    return content_hash(obj)[:n]


def fingerprint(parts: list[tuple[str, Any]]) -> str:
    """Hash an ordered, labelled set of inputs.

    Coordinate transform order, beam geometry and magnetic-declination versions
    are all represented as explicit labelled parts, so changing any of them
    changes the fingerprint even if the numeric result happens to coincide.
    """
    material = [{"role": role, "data": data} for role, data in parts]
    return content_hash(material)
