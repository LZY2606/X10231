"""可重复的反例/边界场景编排（只使用仓库内 fixture）。

每个步骤要么改动一个直接来源（patch_beam_evidence / 姿态缺口 / 配置版本），
要么做人工裁定；每步都产生独立事件与新版本。基线四类证据分开记录。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .db import connect, init_db
from .engine import Engine
from .repository import Repository

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, "fixtures")

BASELINE_VERSIONS = {
    "geometry": "v1", "install": "v1", "declination": "v1",
    "transform": "v1", "qc": "v1", "attitude": "v1", "depth": "v1",
}


def load_fixtures() -> Dict[str, dict]:
    with open(os.path.join(FIX, "adcp_case.json"), encoding="utf-8") as fh:
        case = json.load(fh)
    with open(os.path.join(FIX, "configs.json"), encoding="utf-8") as fh:
        configs = json.load(fh)
    return {"case": case, "configs": configs}


def build_workspace(db_path: str) -> Repository:
    """全新工作区：建库 + 导入基线案例与全部配置版本（不做任何重算）。"""
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = connect(db_path)
    init_db(conn)
    repo = Repository(conn)
    fx = load_fixtures()
    repo.import_case(fx["case"])
    for ctype, versions in fx["configs"].items():
        for ver, payload in versions.items():
            repo.put_config(ctype, ver, payload)
    return repo


def put_all_configs(repo: Repository) -> None:
    fx = load_fixtures()
    for ctype, versions in fx["configs"].items():
        for ver, payload in versions.items():
            repo.put_config(ctype, ver, payload)


def cell_id(ping_1based: int, bin_1based: int) -> str:
    return f"case-A/ping/{ping_1based}/bin/{bin_1based}"


def run_baseline(repo: Repository, actor: str = "reviewer") -> dict:
    eng = Engine(repo)
    return eng.recompute(
        config_versions=dict(BASELINE_VERSIONS),
        event_type="BASELINE_DERIVED",
        basis={"note": "初始导入后首次重算；四类证据分开记录",
               "config_versions": dict(BASELINE_VERSIONS)},
        actor=actor, version_label="baseline")


# ---------------- 七个反例步骤 ----------------

def step_single_beam_failure(repo: Repository, actor="reviewer") -> dict:
    """单 beam 失效：ping1/bin2/beam1 打坏标志（确定性 INVALID）。"""
    cid = cell_id(1, 2)
    repo.patch_beam_evidence(cid, beam=0, field="bad_flag", value=1)
    return Engine(repo).recompute(
        config_versions=dict(BASELINE_VERSIONS),
        event_type="SINGLE_BEAM_FAILURE",
        basis={"mutation": "bad_flag=1", "cell_id": cid, "beam": 0,
               "expect": "仅 3 波束候选；四波束解消失，不与三波束混值"},
        actor=actor, version_label="single-beam-fail")


def step_heading_cross_360(repo: Repository, actor="reviewer") -> dict:
    """姿态跨 360：基线姿态序列已含 359.5 -> 3.5，重算并保留基线配置。"""
    return Engine(repo).recompute(
        config_versions=dict(BASELINE_VERSIONS),
        event_type="HEADING_CROSS_360",
        basis={"note": "heading 在 ping4(t=13s) 附近跨越 0/360；先解绕再插值",
               "ping": 4},
        actor=actor, version_label="heading-360")


def step_time_gap(repo: Repository, actor="reviewer") -> dict:
    """时间缺口：ping5(t=16.5s) 两侧姿态样本 13.5/20.0 间隔 6.5s > 2s。"""
    return Engine(repo).recompute(
        config_versions=dict(BASELINE_VERSIONS),
        event_type="ATTITUDE_TIME_GAP",
        basis={"ping": 5, "t_s": 16.5, "gap_s": 6.5, "max_gap_s": 2.0,
               "expect": "earth=ATTITUDE_GAP；instrument/ship 仍有效，不跨缺口插值"},
        actor=actor, version_label="attitude-gap")


def step_three_beam_ambiguity(repo: Repository, actor="reviewer") -> dict:
    """三波束多解：ping3/bin1/beam2 抹掉 correlation+amplitude 证据 ->
    INDETERMINATE，含/不含该波束的 3、4 波束候选全部保留。"""
    cid = cell_id(3, 1)
    repo.patch_beam_evidence(cid, beam=1, field="correlation", value=None)
    repo.patch_beam_evidence(cid, beam=1, field="amplitude", value=None)
    return Engine(repo).recompute(
        config_versions=dict(BASELINE_VERSIONS),
        event_type="THREE_BEAM_MULTI_SOLUTION",
        basis={"cell_id": cid, "beam": 1,
               "mutation": "correlation 与 amplitude 证据移除",
               "expect": "保留 4-beam(含不确定波束) 与 3-beam(排除) 全部合法候选"},
        actor=actor, version_label="three-beam-ambiguity")


def step_mount_revision(repo: Repository, actor="reviewer") -> dict:
    """安装角修订：切换 install v2；只影响 ship/earth，instrument 指纹不变。"""
    versions = dict(BASELINE_VERSIONS)
    versions["install"] = "v2-mount-revision"
    return Engine(repo).recompute(
        config_versions=versions,
        event_type="MOUNT_ANGLE_REVISION",
        basis={"config": "install", "from": "v1", "to": "v2-mount-revision",
               "expect": "install 版本进入指纹；instrument 节点复用"},
        actor=actor, version_label="mount-revision")


def step_transform_order(repo: Repository, actor="reviewer") -> dict:
    """坐标变换顺序：hpr -> hrp，earth 结果与指纹改变，前级不变。"""
    versions = dict(BASELINE_VERSIONS)
    versions["transform"] = "v2-order"
    return Engine(repo).recompute(
        config_versions=versions,
        event_type="TRANSFORM_ORDER_CHANGED",
        basis={"config": "transform", "from_order": "hpr", "to_order": "hrp",
               "expect": "变换顺序进入指纹；仅 earth 层重算"},
        actor=actor, version_label="transform-order")


def step_invalid_depth_cell(repo: Repository, actor="reviewer") -> dict:
    """无效深度 cell：ping6 底深 2.5m，bin3=2.9m 超过 bottom track。"""
    return Engine(repo).recompute(
        config_versions=dict(BASELINE_VERSIONS),
        event_type="INVALID_DEPTH_CELL",
        basis={"ping": 6, "bin": 3, "cell_depth_m": 2.9, "bottom_m": 2.5,
               "expect": "cell=EXCLUDED(BELOW_BOTTOM_TRACK)，原始证据保留"},
        actor=actor, version_label="invalid-depth")


def step_declination_revision(repo: Repository, actor="reviewer") -> dict:
    """磁偏角版本修订：-7.0 -> -7.5，版本进入指纹，仅 earth 变化。"""
    versions = dict(BASELINE_VERSIONS)
    versions["declination"] = "v2-survey"
    return Engine(repo).recompute(
        config_versions=versions,
        event_type="DECLINATION_REVISION",
        basis={"config": "declination", "from": "v1", "to": "v2-survey",
               "expect": "磁偏角版本进入派生指纹"},
        actor=actor, version_label="declination-revision")


# ---------------- 裁定、冲突、历史 ----------------

def decide_candidate(repo: Repository, cell: str, node_id: str, adopt: bool,
                     actor="reviewer") -> dict:
    """接受/否定某个候选（未选候选保留，确认不是不可逆覆盖）。"""
    return repo.append_event(
        "CANDIDATE_DECISION",
        {"scope_key": cell, "node_id": node_id, "adopted": adopt,
         "note": "未选候选继续保留"}, actor=actor)


def concurrent_edit_conflict(repo: Repository) -> dict:
    """两位使用者从同一版本编辑：登记可解决冲突，两个版本都保留。

    工作区是共享的：A 先提交版本（只含 A 的局部改动），B 在同一 base 意图上
    分叉；系统不静默覆盖 A，而是记录 open 冲突并把 B 的意图并入下一版本，
    等待人工选择 winner。两版本均可回看、可恢复。
    """
    from .engine import Engine
    cur = repo.current_version()
    base = cur["version_id"]
    cid_a = cell_id(2, 2)
    cid_b = cell_id(4, 2)

    # A 的编辑（基于 base）
    repo.patch_beam_evidence(cid_a, beam=2, field="bad_flag", value=1)
    r_a = Engine(repo).recompute(
        config_versions=dict(BASELINE_VERSIONS),
        event_type="CONCURRENT_EDIT_A",
        basis={"actor": "hydrologist-A", "base_version": base,
               "change": {"cell_id": cid_a, "beam": 2, "bad_flag": 1}},
        actor="hydrologist-A", version_label="edit-A")

    # B 同样基于 base，但 A 已提交 -> 检测到分叉，登记冲突
    repo.patch_beam_evidence(cid_b, beam=3, field="bad_flag", value=1)
    r_b = Engine(repo).recompute(
        config_versions=dict(BASELINE_VERSIONS),
        event_type="CONCURRENT_EDIT_B",
        basis={"actor": "hydrologist-B", "base_version": base,
               "change": {"cell_id": cid_b, "beam": 3, "bad_flag": 1},
               "note": "与 edit-A 从同一 base 分叉，需可解决冲突"},
        actor="hydrologist-B", version_label="edit-B")
    conflict = repo.register_conflict(
        scope_key="workspace", base_version_id=base,
        winner=r_a["version_id"], loser=r_b["version_id"],
        detail={"reason": "两个编辑从同一 base_version 分叉",
                "A_change": {"cell": cid_a, "beam": 2},
                "B_change": {"cell": cid_b, "beam": 3},
                "resolution": "resolve_conflict(conflict_id, winner_version)"})
    repo.append_event("EDIT_CONFLICT_DETECTED",
                      {"conflict_id": conflict, "base_version": base,
                       "winner_pending": r_a["version_id"],
                       "loser_pending": r_b["version_id"]},
                      actor="system")
    return {"A": r_a, "B": r_b, "base_version": base,
            "conflict_id": conflict}


STEPS = [
    ("single_beam_failure", step_single_beam_failure),
    ("heading_cross_360", step_heading_cross_360),
    ("time_gap", step_time_gap),
    ("three_beam_ambiguity", step_three_beam_ambiguity),
    ("mount_revision", step_mount_revision),
    ("transform_order", step_transform_order),
    ("invalid_depth_cell", step_invalid_depth_cell),
    ("declination_revision", step_declination_revision),
]


def run_full_replay(db_path: str, with_conflict: bool = True) -> dict:
    """九十秒演示的完整短样例路径：基线 -> 七个边界 -> 冲突。"""
    repo = build_workspace(db_path)
    trace = [{"step": "baseline", "result": run_baseline(repo)}]
    for name, fn in STEPS:
        trace.append({"step": name, "result": fn(repo)})
    if with_conflict:
        trace.append({"step": "concurrent_conflict",
                      "result": concurrent_edit_conflict(repo)})
    return {"db_path": db_path, "trace": trace,
            "events": repo.list_events()}
