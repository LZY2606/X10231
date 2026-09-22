"""Replay the “流束换算台” counterexample archive end to end.

The runner seeds the baseline (every source type separate), then exercises:
single-beam failure, heading crossing 360, attitude time gap, three-beam
multi-solution, mount-angle revision, transform-order choice, invalid depth
cell, a local-only source change and a conflicting human decision. Accepted
and unaccepted results are both retained; everything is an event with stable
identity and machine-readable payload.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from adcp import fixture
from adcp.fixture import build_view
from adcp.pipeline import run


def _derive(store, decisions: Optional[List[dict]] = None) -> Dict[str, Any]:
    pack = store.active_pack()
    return run(build_view(pack, decisions))


def _commit(store, branch: str, label: str, kind: str, note: str,
            base_cid: Optional[str], actor: str, etype: str,
            payload: Dict[str, Any], decisions: Optional[List[dict]] = None,
            ) -> Dict[str, str]:
    result = _derive(store, decisions)
    nodes = store.propagate(store.branch_head(branch)
                            if branch == "main" else base_cid, result["nodes"])
    return store.commit(branch, nodes, label, kind, note, base_cid, actor,
                        etype, {"note": note, **payload},
                        parent_commit=store.branch_head(branch)
                        if branch != "main" else None)


def seed(store) -> str:
    pack = fixture.build_case_pack()
    store.seed_pack(pack)
    result = _derive(store)
    nodes = store.propagate(None, result["nodes"])
    out = store.commit(
        "main", nodes, "baseline", "baseline",
        "基线：波束速度/correlation/amplitude/坏 beam 标志/姿态/磁偏角/安装角分开记录",
        None, "importer", "baseline_imported",
        {"excluded": result["excluded"], "summary": result["summary"]})
    store.log_decision(
        {"kind": "review_note",
         "text": "审阅者核对七类源数据分开记录，接受基线",
         "accepted": ["four_beam", "unique_three_beam"],
         "not_adopted": ["ambiguous triplets pending evidence"]},
        "reviewer", out["cid"])
    return out["cid"]


def play(store, baseline_cid: str) -> Dict[str, Any]:
    refs: Dict[str, Any] = {}

    # 1. single beam failure is already in the baseline (ping 2, beam 1)
    ev = store.list_events()
    refs["single_beam_failure_event"] = next(
        e["eid"] for e in ev if e["type"] == "baseline_imported")

    # 2. local-only source change: correct one beam velocity (cm/s note cell)
    r = next(s for s in store.active_sources()
             if s["type"] == "beam_velocity" and s["ping"] == 11
             and s["cell"] == 0 and s["beam"] == 0)
    rev = store.add_source_revision(
        r["sid"], 0.0, "m/s", note="现场复核：该 ping 的 beam0 读数置零复测",
        actor="field-team")
    before_nodes = len(store.get_nodes(baseline_cid))
    out = _commit(
        store, "main", "local-source-fix", "source_change",
        "只改动一个直接来源：ping11 cell0 beam0 局部修订", baseline_cid,
        "field-team", "local_source_change",
        {"sid": r["sid"], "rev": rev["rev"],
         "affected_scope": "ping11/cell0 subtree",
         "propagation": store.propagation_report()})
    refs["local_source_change"] = out["cid"]
    refs["local_source_change_event"] = out["event_id"]

    # 3. mount-angle revision (new version tag) -> geometry downstream moves
    pack = store.active_pack()
    new_mount = copy.deepcopy(pack["parameters"]["mount"])
    new_mount.update({"version": "mount-B", "heading_deg": 1.5,
                      "pitch_deg": 0.2})
    base = out["cid"]
    store.update_parameter(
        "mount", new_mount, actor="surveyor",
        note="安装角修订：横向偏角 1.5°、纵倾 0.2°")
    out = _commit(
        store, "main", "mount-revision", "parameter_revision",
        "安装角修订进入派生指纹，ship/earth 全量重算，instrument 不变",
        base, "surveyor", "mount_revised",
        {"old_version": "mount-A", "new_version": "mount-B"})
    refs["mount_revision"] = out["cid"]
    refs["mount_revision_event"] = out["event_id"]

    # 4. transform-order branch (counterexample: order changes results)
    store.create_branch("transform-order-alt", out["cid"])
    pack2 = store.active_pack()
    alt_params = copy.deepcopy(pack2["parameters"])
    alt_params["transform_order"] = "attitude_first"
    alt_pack = {**pack2, "parameters": alt_params}
    result = run(build_view(alt_pack))
    nodes = store.propagate(out["cid"], result["nodes"])
    alt = store.commit(
        "transform-order-alt", nodes, "attitude-first-order", "parameter",
        "坐标变换顺序反例：attitude_first 与 mount_first 结果不同", out["cid"],
        "reviewer", "transform_order_alternative",
        {"orders": ["mount_first", "attitude_first"],
         "note": "顺序进入派生指纹，两套结果并存不合并"})
    refs["transform_order_alternative"] = alt["cid"]
    refs["transform_order_event"] = alt["event_id"]

    # 5. declination version branch (magnetic declination revision)
    store.create_branch("declination-alt", refs["mount_revision"])
    alt_params2 = copy.deepcopy(store.active_pack()["parameters"])
    new_decl = copy.deepcopy(alt_params2["declination"])
    new_decl.update({"version": "decl-B", "deg": 7.8})
    alt_params2["declination"] = new_decl
    alt_pack2 = {**store.active_pack(), "parameters": alt_params2}
    result = run(build_view(alt_pack2))
    nodes = store.propagate(refs["mount_revision"], result["nodes"])
    dalt = store.commit(
        "declination-alt", nodes, "declination-B", "parameter",
        "磁偏角版本反例：decl-A 7.0° -> decl-B 7.8°",
        refs["mount_revision"], "navigator", "declination_revised",
        {"old_version": "decl-A", "new_version": "decl-B"})
    refs["declination_alternative"] = dalt["cid"]

    # 6. human decision on the ambiguous multi-candidate cell (ping 9)
    ambiguous_ping, ambiguous_cell = 9, 0
    decision = {"ping": ambiguous_ping, "cell": ambiguous_cell,
                "candidate_id": "t-012", "rationale":
                "beam2/3/0 与邻近 cell 一致，保留 beam1 未采纳候选备查",
                "kind": "candidate_adoption"}
    store.log_decision(decision, "reviewer", None)
    out = _commit(
        store, "main", "decide-three-beam", "decision",
        "三波束多解：人工选定一个合法候选，其余候选保留不覆盖",
        refs["mount_revision"], "reviewer", "candidate_decided",
        {"ping": ambiguous_ping, "cell": ambiguous_cell,
         "candidate_id": "t-012",
         "unadopted_kept": True}, decisions=[decision])
    refs["decision"] = out["cid"]
    refs["decision_event"] = out["event_id"]

    # 7. conflicting human decision from the stale baseline (two reviewers)
    stale_decision = {"ping": ambiguous_ping, "cell": ambiguous_cell,
                      "candidate_id": "t-013", "kind": "candidate_adoption",
                      "rationale": "另一审阅者基于旧基线选择不同候选"}
    try:
        _commit(store, "main", "stale-decision", "decision",
                "冲突反例：两名使用者从同一版本编辑", baseline_cid,
                "reviewer-2", "candidate_decided",
                {"candidate_id": "t-013"}, decisions=[stale_decision])
    except Exception as exc:  # EditConflict
        refs["conflict"] = getattr(exc, "conflict_id", str(exc))
    else:  # pragma: no cover - conflict must occur
        refs["conflict"] = None

    return refs
