"""Read-model helpers for the review UI: ping view, chains, evidence."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def ping_view(nodes: Dict[str, dict], pno: int) -> Dict[str, Any]:
    out = {"ping": pno, "cells": []}
    for uid, n in sorted(nodes.items()):
        v = n["value"]
        if n["kind"] == "beam_quality" and v.get("ping") == pno:
            cell = {"cell": v["cell"], "quality_node": n["uid"],
                    "status": v["status"], "valid_beams": v["valid_beams"],
                    "beam_quality": v["beam_quality"],
                    "bin_depth": v.get("bin_depth"),
                    "depth_valid": v.get("depth_valid")}
            sol = nodes.get(f"b-p{pno}-c{v['cell']}")
            if sol:
                cell["solution_node"] = sol["uid"]
                cell["resolution"] = sol["value"]["resolution"]
                cell["adopted"] = sol["value"].get("adopted")
                cell["candidate_ids"] = [c["id"] for c in
                                         sol["value"].get("candidates", [])]
            for kind, prefix in (("instrument", "i"), ("ship", "s"),
                                 ("earth", "e"), ("absolute", "a")):
                nd = nodes.get(f"{prefix}-p{pno}-c{v['cell']}")
                if nd:
                    cell[kind + "_node"] = nd["uid"]
                    cell[kind + "_value"] = nd["value"]
            out["cells"].append(cell)
    out["attitude"] = nodes.get(f"att-p{pno}", {}).get("value")
    out["bottom_track"] = nodes.get(f"bt-p{pno}", {}).get("value")
    return out


def evidence_chain(nodes: Dict[str, dict], uid: str,
                   depth: int = 0, seen: Optional[set] = None) -> Dict[str, Any]:
    """Walk deps + source refs back to the raw materials."""
    if seen is None:
        seen = set()
    if uid not in nodes or uid in seen:
        return {}
    seen.add(uid)
    n = nodes[uid]
    return {"uid": n["uid"], "kind": n["kind"], "fp": n["fp"],
            "value": n["value"], "refs": n["refs"],
            "deps": [evidence_chain(nodes, d["uid"], depth + 1, seen)
                     for d in n["deps"] if d["uid"] in nodes and "*"
                     not in d["uid"]]}


def excluded_view(nodes: Dict[str, dict]) -> List[dict]:
    ex = nodes.get("excluded")
    return ex["value"]["excluded"] if ex else []
