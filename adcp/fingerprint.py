"""Content fingerprints.

Every derived artifact binds the exact source representations, beam geometry,
magnetic declination version, mount angles, transform order and rule version
that produced it. Changing any of those changes the fingerprint, so history
can never be silently overwritten by a newer recomputation.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Sequence

RULE_VERSION = "rules-1.0"


def canonical(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def fingerprint(kind: str, inputs: Dict[str, Any], rule_tags: Sequence[str] = ()) -> str:
    body = canonical({
        "kind": kind,
        "rule_version": RULE_VERSION,
        "rule_tags": list(rule_tags),
        "inputs": inputs,
    })
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return "%s-%s" % (kind, digest[:16])
