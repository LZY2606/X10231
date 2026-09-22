"""依赖图传播引擎。

一次重算（对应一个事件/版本）：

1. 读取指定的各配置版本（几何/安装/磁偏角/变换/QC/姿态插值/深度）；
2. 对每个 ping：bottom track -> 深度有效性 -> 姿态插值（不跨缺口）->
   每个 cell 的逐级派生（三波束/四波束候选全部建为独立节点）；
3. 节点内容寻址：未受影响的 cell 其输入指纹不变，直接复用旧节点，
   因此只更新受影响部分；
4. 汇总槽位清单，完整成功后原子提交新版本。

三波束与四波束候选是不同节点（beams 集合不同 -> 指纹不同），绝不混值。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from .attitude import interpolate_at
from .derive import depth_cell_validity, derive_cell, solve_bottom_track
from .fingerprint import fingerprint
from .repository import Repository


def _raw_scope(cell_id: str) -> str:
    return cell_id  # cell_id 即该 cell 原始证据的稳定身份


class Engine:
    def __init__(self, repo: Repository):
        self.repo = repo

    def affected_scopes(self, changed_cell_ids: List[str],
                        changed_config_types: Optional[List[str]] = None,
                        changed_attitude: bool = False) -> Dict[str, Any]:
        """说明哪些结果仍有效、哪些需要重算。"""
        cur = self.repo.current_version()
        slots = self.repo.version_slots(cur["version_id"]) if cur else {}
        affected_cells = set(changed_cell_ids)
        # 几何/QC 影响所有 cell；安装角影响 ship/earth；磁偏角/变换/姿态只影响 earth
        cfg = set(changed_config_types or [])
        all_scope = bool(cfg & {"geometry", "qc", "depth"})
        ship_scope = bool(cfg & {"install"})
        earth_scope = bool(cfg & {"declination", "transform", "attitude"}) or \
            changed_attitude
        return {
            "all_cells": all_scope,
            "recompute_ship": ship_scope or all_scope,
            "recompute_earth": earth_scope or ship_scope or all_scope,
            "changed_cells": sorted(affected_cells),
            "note": ("内容寻址：未列出的 cell 输入指纹不变，节点复用，"
                     "仍标记为有效。"),
        }

    def recompute(self, *, config_versions: Dict[str, str],
                  event_type: str, basis: dict, actor: str = "reviewer",
                  parent_seq: Optional[int] = None,
                  version_label: Optional[str] = None,
                  adopted: Optional[Dict[str, Tuple[str, bool]]] = None
                  ) -> Dict[str, Any]:
        """完整重算并原子提交一个新版本。返回事件与统计。"""
        repo = self.repo
        cfgs = repo.get_configs(config_versions)
        seq = repo.append_event(event_type, basis, actor=actor,
                                parent_seq=parent_seq)
        version_id = version_label or f"v{seq}"

        # 先沿用上一版本槽位（未受影响项保持身份）
        cur = repo.current_version()
        slots: Dict[str, str] = dict(
            repo.version_slots(cur["version_id"])) if cur else {}

        stats = {"nodes_reused": 0, "nodes_created": 0,
                 "cells": 0, "excluded_cells": 0, "attitude_gap_pings": 0,
                 "candidate_scopes": []}
        new_candidates: List[Tuple[int, str, List[Tuple[str, str]]]] = []

        pings = repo.list_pings()
        for ping in pings:
            pid = ping["ping_id"]
            batch_id = ping["batch_id"]
            # bottom track（提供深度）
            bt_ev = repo.bottom_track_evidence(pid)
            bt = solve_bottom_track(
                n_beams=cfgs["geometry"].geometry.n_beams,
                bt_evidence=bt_ev, thresholds=cfgs["qc"],
                geometry_cfg=cfgs["geometry"], install=cfgs["install"],
                attitude=None, decl=cfgs["decl"], tform=cfgs["tform"])
            bottom_m = bt.get("bottom_depth_m")

            # 姿态插值（独立流，不跨缺口）
            att_series = repo.attitude_series(batch_id)
            att = interpolate_at(att_series, ping["t_s"], cfgs["attitude"])
            att_dict = att.as_dict()
            if att.status != "OK":
                stats["attitude_gap_pings"] += 1
            # 含姿态再解一次 bottom track earth
            bt_full = solve_bottom_track(
                n_beams=cfgs["geometry"].geometry.n_beams,
                bt_evidence=bt_ev, thresholds=cfgs["qc"],
                geometry_cfg=cfgs["geometry"], install=cfgs["install"],
                attitude=att_dict, decl=cfgs["decl"], tform=cfgs["tform"])
            bt_fp = {
                "kind": "bottom_track", "ping_id": pid,
                "bt_evidence": _stable_evidence(bt_ev),
                "configs": _config_fp(cfgs), "attitude": att_dict,
            }
            bt_id, reused, _ = repo.get_or_create_node(
                "bottom_track", bt_fp, {}, seq)
            repo.set_node_result(bt_id, bt_full)
            slots[f"ping:{pid}/bottom_track"] = bt_id
            stats["nodes_reused" if reused else "nodes_created"] += 1

            for cell in repo.list_cells(pid):
                cid = cell["cell_id"]
                stats["cells"] += 1
                raw_depth = cell["depth_raw"]
                cell_depth = raw_depth if isinstance(raw_depth, (int, float)) \
                    else float("nan")
                depth = depth_cell_validity(
                    cell_depth, cfgs["depth"]["blank_m"], bottom_m)
                evidence = repo.cell_evidence(cid)
                result = derive_cell(
                    n_beams=cfgs["geometry"].geometry.n_beams,
                    cell_evidence=evidence, thresholds=cfgs["qc"],
                    geometry_cfg=cfgs["geometry"], install=cfgs["install"],
                    depth=depth, attitude=att_dict,
                    decl=cfgs["decl"], tform=cfgs["tform"])
                if not depth["valid"]:
                    stats["excluded_cells"] += 1
                fp_in = {
                    "kind": "cell_derivation", "cell_id": cid,
                    "evidence": _stable_evidence(evidence),
                    "depth": depth, "attitude": att_dict,
                    "configs": _config_fp(cfgs),
                }
                nid, reused, _ = repo.get_or_create_node(
                    "cell_derivation", fp_in,
                    {"bottom_track": bt_id}, seq,
                    explain={"why": "内容寻址：证据/配置/姿态/顺序指纹"})
                repo.set_node_result(nid, result)
                slots[f"cell:{cid}"] = nid
                stats["nodes_reused" if reused else "nodes_created"] += 1

                # 候选（三波束/四波束）分别登记，保留采纳与未采纳
                cand_nodes: List[Tuple[str, str]] = []
                for cand in result["candidates"]:
                    beams = cand["beams"]
                    label = f"{len(beams)}-beam:{','.join(map(str, beams))}"
                    cfp = {"scope": cid, "beams": beams,
                           "parent_cell_node": nid,
                           "instrument": cand["instrument"],
                           "ship": cand["ship"], "earth": cand["earth"]}
                    cnid, creused, _ = repo.get_or_create_node(
                        "velocity_candidate", cfp, {"cell": nid}, seq)
                    repo.set_node_result(cnid, cand)
                    cand_nodes.append((cnid, label))
                if cand_nodes:
                    repo.record_candidates(seq, cid, cand_nodes)
                    stats["candidate_scopes"].append(cid)
                    new_candidates.append((seq, cid, cand_nodes))

        repo.commit_version(version_id, seq, slots)
        # 人工采纳标记（可选）
        for _seq, scope, cands in new_candidates:
            if adopted and scope in adopted:
                node_id, ok = adopted[scope]
                repo.mark_candidate(scope, node_id, ok)
        return {"event_seq": seq, "version_id": version_id, "stats": stats,
                "slot_count": len(slots)}


def _config_fp(cfgs: dict) -> dict:
    v = cfgs["versions"]
    return {
        "geometry": cfgs["geometry"].fingerprint_input(),
        "install": cfgs["install"].fingerprint_input(),
        "declination": cfgs["decl"].fingerprint_input(),
        "transform": cfgs["tform"].fingerprint_input(),
        "qc": cfgs["qc"].fingerprint_input(),
        "attitude": cfgs["attitude"].fingerprint_input(),
        "depth": cfgs["depth"],
        "resolved_versions": v,
    }


def _stable_evidence(ev: Dict[int, dict]) -> dict:
    return {str(k): ev[k] for k in sorted(ev)}
