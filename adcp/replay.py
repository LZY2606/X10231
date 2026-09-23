"""反例档案重放剧本：依次经过七类边界情形，只改动一个直接来源。

每步写入 machine-readable verdict 事件：
  verdict:{scenario} -> {status: contained|propagated|blocked, impacted:[...], evidence:{...}}
所有结论都可从最终状态经事件、流版本、basis 反查回具体材料/规则/决定。
"""
from __future__ import annotations

import copy
from typing import Any

from . import pipeline
from .fixture import build_baseline
from .repo import ConflictError, Repository
from .service import Service

SCENARIOS = [
    "single_beam_failure", "heading_wrap_360", "time_gap", "three_beam_multiplicity",
    "mount_revision", "transform_order", "invalid_depth_cell",
]


def _stream_payload(svc: Service, stream_id: str) -> tuple[dict[str, Any], int]:
    rec = next(r for r in svc.repo.current_state()["records"] if r["stream_id"] == stream_id)
    return copy.deepcopy(rec["payload"]), rec["version"]


def _verdict(seq: int, scenario: str, status: str, impacted: list[str], evidence: dict[str, Any]) -> dict[str, Any]:
    return {"event_type": f"verdict:{scenario}", "scope": "scenario", "subject": scenario,
            "detail": {"status": status, "impacted": impacted, "evidence": evidence, "seq": seq}}


def run_case(db_path: str, *, reset: bool = True, progress=None) -> dict[str, Any]:
    import os
    if reset and os.path.exists(db_path):
        os.remove(db_path)
    repo = Repository(db_path)
    svc = Service(repo)
    steps: list[dict[str, Any]] = []

    def log(step):
        steps.append(step)
        if progress:
            progress(step)

    baseline = build_baseline()
    init = svc.initialize(baseline, message="baseline: separate beam/corr/amp/flag/attitude/decl/mount streams")
    seq0 = init["ingest"]["seq"]
    log({"step": 0, "name": "baseline", "seq": seq0,
         "updated": len(init["derivations"]["updated"]), "unchanged": 0})

    # 1) 单 beam 失效（ping-20 / 12m，beam0）——仅核对基线证据：唯一三波束解
    cell1 = "ping-20-cell120"
    d1 = next(d for d in repo.current_derivations() if d["derivation_id"] == cell1)
    kinds1 = sorted(c["kind"] for c in d1["payload"]["candidates"])
    ev1 = [e["type"] for e in d1["payload"]["events"]]
    v = _verdict(seq0, "single_beam_failure",
                 "contained" if kinds1 == ["three"] and "single_beam_failure" in ev1 else "propagated",
                 [cell1], {"candidate_kinds": kinds1, "events": ev1,
                           "candidate": d1["payload"]["candidates"][0]["candidate_id"],
                           "missing": d1["payload"]["candidates"][0]["missing_beams"]})
    repo.add_events([v], seq=seq0)
    log({"step": 1, "name": "single_beam_failure", "seq": seq0, "verdict": v["detail"]["status"]})

    # 2) 姿态跨越 360：ping-45 已有跨卷绕事件（证据）；再编辑 50s 姿态 3->5（只改一个来源）
    prof45 = next(d for d in repo.current_derivations()
                  if d["derivation_id"] == "ping-45-profile")
    wrap_before = any(e["type"] == "heading_wrap_360" for e in prof45["payload"]["events"])
    pay, ver = _stream_payload(svc, "stm-att-50")
    pay["heading_raw"] = "5"
    res = svc.edit("stm-att-50", pay, base_version=ver, note="heading revision 3->5 deg across 360")
    impacted = res["derivations"]["updated"] and [u["derivation_id"] for u in res["derivations"]["updated"]]
    prof45b = next(d for d in repo.current_derivations() if d["derivation_id"] == "ping-45-profile")
    wrap_after = any(e["type"] == "heading_wrap_360" for e in prof45b["payload"]["events"])
    # 姿态是全局插值流：30s 及更早的 ping 不使用 50s 样本，应保持指纹不变
    pre_30 = [x for x in (
        u["derivation_id"] for u in res["derivations"]["updated"])]
    no_early_change = not any(x.startswith(("ping-00", "ping-05", "ping-10",
                                            "ping-15", "ping-20", "ping-30")) for x in impacted)
    local = no_early_change
    v = _verdict(res["seq"], "heading_wrap_360",
                 "contained" if wrap_before and wrap_after and local else "propagated",
                 impacted, {"wrap_before": wrap_before, "wrap_after": wrap_after,
                            "interpolated_heading_at_45": prof45b["payload"]["attitude"]["heading_deg"],
                            "unchanged_derivations": res["derivations"]["unchanged"]})
    repo.add_events([v], seq=res["seq"])
    log({"step": 2, "name": "heading_wrap_360", "seq": res["seq"], "verdict": v["detail"]["status"]})

    # 3) 时间缺口：ping-40 earth 必须被 attitude_gap 阻断，instrument/ship 保留
    cells40 = [d for d in repo.current_derivations() if d["kind"] == "cell_flow"
               and d["derivation_id"].startswith("ping-40-")]
    blocked = all(any(c["earth"] is None and c["earth_reason"] == "attitude_gap"
                      for c in d["payload"]["candidates"]) for d in cells40)
    kept = all(any(c["ship"] is not None for c in d["payload"]["candidates"]) for d in cells40)
    v = _verdict(repo.head_seq(), "time_gap", "blocked" if blocked and kept else "propagated",
                 [d["derivation_id"] for d in cells40],
                 {"earth_blocked": blocked, "instrument_ship_kept": kept,
                  "attitude_status": cells40[0]["payload"] and "gap"})
    repo.add_events([v], seq=repo.head_seq())
    log({"step": 3, "name": "time_gap", "seq": repo.head_seq(), "verdict": v["detail"]["status"]})

    # 4) 三波束多解：ping-30 / 6m，四波束误差速度超阈 -> 1 个四波束 + 4 个三波束候选共存
    cell4 = "ping-30-cell060"
    d4 = next(d for d in repo.current_derivations() if d["derivation_id"] == cell4)
    cand = d4["payload"]["candidates"]
    n3 = sum(1 for c in cand if c["kind"] == "three")
    n4 = sum(1 for c in cand if c["kind"] == "four")
    distinct = len({tuple(round(x, 9) for x in c["instrument"]) for c in cand})
    ambiguous = d4["payload"]["ambiguous"] and d4["payload"]["default_candidate_id"] is None
    v = _verdict(repo.head_seq(), "three_beam_multiplicity",
                 "contained" if n3 == 4 and n4 == 1 and distinct == 5 and ambiguous else "propagated",
                 [cell4], {"n_three": n3, "n_four": n4, "distinct_instruments": distinct,
                           "ambiguous_unresolved": ambiguous,
                           "candidate_ids": [c["candidate_id"] for c in cand],
                           "error_velocity_mps": cand[0]["error_velocity_mps"]})
    repo.add_events([v], seq=repo.head_seq())
    # 冲突人工决定：审阅者 A 选四波束，审阅者 B 从同一 seq 选三波束，必须可解决
    dec_a = svc.decide("cell_flow", cell4, "accept", chosen_candidate_id="cand4",
                       actor="reviewer-A", rationale="prefer least-squares for discharge")
    dec_b = svc.decide("cell_flow", cell4, "accept", chosen_candidate_id="cand3-missing1",
                       actor="reviewer-B", rationale="suspect beam1 from amplitude trace")
    resolve = svc.decide("cell_flow", cell4, "reconcile", chosen_candidate_id="cand3-missing1",
                         actor="lead", rationale="keep both visible; lead continues review-B branch; cand4 retained")
    log({"step": 4, "name": "three_beam_multiplicity", "seq": repo.head_seq(),
         "verdict": v["detail"]["status"], "conflict_decisions": [dec_a["seq"], dec_b["seq"], resolve["seq"]]})

    # 5) 安装角修订：mount heading 0.5 -> 1.2；instrument 不变，ship/earth 指纹变
    pay, ver = _stream_payload(svc, "mount-alpha")
    before = next(d for d in repo.current_derivations() if d["derivation_id"] == "ping-00-cell020")
    fp_before = before["fingerprint"]
    inst_before = before["payload"]["candidates"][0]["instrument"]
    pay["heading_deg"] = 1.2
    res = svc.edit("mount-alpha", pay, base_version=ver, note="mounting angle revision 0.5->1.2 deg")
    impacted = [u["derivation_id"] for u in res["derivations"]["updated"]]
    after = next(d for d in repo.current_derivations() if d["derivation_id"] == "ping-00-cell020")
    inst_same = after["payload"]["candidates"][0]["instrument"] == inst_before
    fp_changed = after["fingerprint"] != fp_before
    v = _verdict(res["seq"], "mount_revision",
                 "contained" if inst_same and fp_changed and len(impacted) > 1 else "propagated",
                 impacted, {"instrument_unchanged": inst_same, "fingerprint_changed": fp_changed})
    repo.add_events([v], seq=res["seq"])
    log({"step": 5, "name": "mount_revision", "seq": res["seq"], "verdict": v["detail"]["status"],
         "impacted_n": len(impacted)})

    # 6) 坐标转换顺序：rdi_xyz -> hpr；earth 指纹必须变
    pay, ver = _stream_payload(svc, "grid-main")
    earth_before = after_earth = None
    d6 = next(d for d in repo.current_derivations() if d["derivation_id"] == "ping-00-cell020")
    earth_before = d6["payload"]["candidates"][0]["earth"]
    fp6 = d6["fingerprint"]
    pay["transform_order"] = "hpr"
    res = svc.edit("grid-main", pay, base_version=ver, note="switch coordinate transform order to HPR")
    impacted = [u["derivation_id"] for u in res["derivations"]["updated"]]
    d6b = next(d for d in repo.current_derivations() if d["derivation_id"] == "ping-00-cell020")
    earth_after = d6b["payload"]["candidates"][0]["earth"]
    order_in_fp = d6b["basis"]["transform_order"] == "hpr"
    moved = earth_before != earth_after and d6b["fingerprint"] != fp6
    v = _verdict(res["seq"], "transform_order",
                 "contained" if moved and order_in_fp else "propagated", impacted,
                 {"earth_changed": earth_before != earth_after,
                  "fingerprint_changed": d6b["fingerprint"] != fp6,
                  "order_in_basis": order_in_fp})
    repo.add_events([v], seq=res["seq"])
    log({"step": 6, "name": "transform_order", "seq": res["seq"], "verdict": v["detail"]["status"]})

    # 7) 无效深度 cell：ping-15 / 12m 在底床以下，保留原始材料但 cell 被排除
    cell7 = "ping-15-cell120"
    d7 = next(d for d in repo.current_derivations() if d["derivation_id"] == cell7)
    excl = d7["payload"]["excluded"] and "below_bottom" in d7["payload"]["exclude_reasons"]
    raw_present = next(r for r in repo.current_state()["records"]
                       if r["stream_id"] == f"bv:{cell7}")["payload"]["beams"]
    v = _verdict(repo.head_seq(), "invalid_depth_cell",
                 "contained" if excl and all(x is not None for x in raw_present) else "propagated",
                 [cell7], {"excluded": d7["payload"]["excluded"],
                           "reasons": d7["payload"]["exclude_reasons"],
                           "raw_beams_retained": raw_present})
    repo.add_events([v], seq=repo.head_seq())
    log({"step": 7, "name": "invalid_depth_cell", "seq": repo.head_seq(), "verdict": v["detail"]["status"]})

    return {"db_path": db_path, "steps": steps, "head_seq": repo.head_seq(),
            "landmarks": baseline["landmarks"]}
