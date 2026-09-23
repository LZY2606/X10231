"""Loader and replayer for the offline domain case pack (流束换算台)."""
from __future__ import annotations

import json
import os

from . import transforms as tf
from .pipeline import DEFAULT_PARAMS, Pipeline

CASE_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "cases", "liushu_bench")


def _load(name, case_dir=CASE_DIR):
    with open(os.path.join(case_dir, name), encoding="utf-8") as f:
        return json.load(f)


def import_sources(store, doc, label=None):
    batch = store.add_batch(label or doc.get("label", "import"))
    for s in doc["sources"]:
        store.add_source(
            batch, s["kind"], ping=s.get("ping"), beam=s.get("beam"),
            cell=s.get("cell"), t=s.get("t"), frame=s.get("frame"),
            unit=s.get("unit"), raw=s.get("raw"), value=s.get("value"),
            meta=s.get("meta"))
    store.commit()
    store.add_event(store.current_version(), "sources_imported", {
        "batch": batch, "label": doc.get("label"),
        "count": len(doc["sources"]),
        "rule": "raw representations preserved at import; normalisation "
                "only happens in the derived layer"})
    return batch


def ensure_params(store):
    existing = store.params()
    for name, value, unit, meta in DEFAULT_PARAMS:
        if name not in existing:
            store.set_param(name, value, unit, meta)


def replay(store, case_dir=CASE_DIR):
    """Replay the full case-pack narrative.  Returns version ids."""
    pipe = Pipeline(store)
    out = {}
    ensure_params(store)

    # 1. baseline import + first derivation
    import_sources(store, _load("baseline.json", case_dir))
    out["v1_baseline"] = pipe.derive("v1 baseline derivation")

    # 2. local source change (only local dependencies recomputed)
    import_sources(store, _load("change_local.json", case_dir))
    out["v2_local_change"] = pipe.derive("v2 local source correction")

    # 3. mounting-angle parameter revision
    rev = _load("revision_mounting_angle.json", case_dir)
    out["v3_mounting_revision"] = pipe.derive(
        "v3 mounting-angle revision",
        param_updates=[(rev["param"], rev["value"], rev["unit"],
                        rev["meta"])])

    # 4. transform-order variant as a retained branch (pointer stays)
    store.set_param("transform_order", tf.ORDER_ATTITUDE_FIRST, "text",
                    {"variant": "attitude applied before mounting"})
    out["v4_order_variant"] = pipe.derive(
        "v4 transform-order variant (branch)", move_pointer=False)
    store.set_param("transform_order", tf.ORDER_MOUNTING_FIRST, "text",
                    {"restore": "mainline order"})
    store.add_event(out["v4_order_variant"], "transform_order_variant", {
        "branch_version": out["v4_order_variant"],
        "mainline_order": tf.ORDER_MOUNTING_FIRST,
        "variant_order": tf.ORDER_ATTITUDE_FIRST,
        "rule": "order is part of the fingerprint; the variant branch is "
                "retained for comparison, mainline pointer unchanged"})

    # 5. conflicting human decisions from the same base version
    conflict = _load("conflict_decisions.json", case_dir)
    base = out["v3_mounting_revision"]
    a = conflict["decision_a"]
    vid_a, _ = pipe.decide(base, conflict["node"], a["candidate"],
                           note=a["note"])
    out["v5_decision_a"] = vid_a
    b = conflict["decision_b"]
    vid_b, conflict_id = pipe.decide(base, conflict["node"],
                                     b["candidate"], note=b["note"])
    out["v6_decision_b_branch"] = vid_b
    out["conflict_id"] = conflict_id
    if conflict_id:
        keep = conflict.get("resolution", "a")
        store.resolve_conflict(conflict_id, keep)
        out["resolved_keep"] = keep
    return out
