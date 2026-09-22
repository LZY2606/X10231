"""Content fingerprints for derived nodes.

The fingerprint mixes: ruleset version, transform order, beam geometry version,
magnetic-declination version, mount-angle version, source record revisions and
the canonical normalised inputs. Geometry/order/declination therefore always
move the fingerprint, as required.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Sequence

from adcp import RULESET_VERSION


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=_default)


def _default(o: Any) -> Any:
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if isinstance(o, (set, tuple)):
        return sorted(o, key=str)
    return str(o)


def fingerprint(parts: Sequence[Any]) -> str:
    h = hashlib.sha256()
    h.update(("ruleset=" + RULESET_VERSION + "\n").encode("utf-8"))
    for part in parts:
        h.update(canonical(part).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def config_block(dataset: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "geometry": dataset["geometry"],
        "mount": dataset["mount"],
        "declination": dataset["declination"],
        "thresholds": dataset.get("thresholds", {}),
        "transform_order": dataset.get("transform_order", "mount_first"),
    }
