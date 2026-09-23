"""Offline case package: import and deterministic replay.

All fixtures live in ``fixtures/casepack/`` (repo-local, no network).

Initial import keeps eight source kinds in *separate* raw records:
beam velocity, correlation, amplitude, bad-beam flag, heading, pitch,
roll, magnetic declination and mounting angle (plus bottom track and
transducer configuration).  Raw representation, unit and frame are
preserved; only the derived layer normalizes.

Replay then performs, in order:
  1. revise-mounting     - a parameter revision (mounting angle);
  2. beam-fail           - one direct source change (ping4 beam2 flag);
  3. conflict-decision   - two reviewers edit from the same base version,
                           collide, and the conflict is resolved.
Every boundary situation in the replay has machine-readable evidence.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from . import engine
from .store import ConflictError, Store
from .units import normalize

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "casepack"

JUDGMENTS = [
    {"rule_id": "no-mixing-three-four-beam",
     "statement": "三波束解与四波束解不混为一值",
     "enforcement": "instr 节点对每个 cell 标记 solution_kind；"
     "三波束保留全部合法候选，下游不做跨 kind 平均"},
    {"rule_id": "no-attitude-interp-across-gap",
     "statement": "姿态插值不跨缺口",
     "enforcement": "att 节点对超过 max_att_gap_s 的括号输出 status=gap，"
     "ship/earth 节点标记 blocked 而不是硬算"},
    {"rule_id": "fingerprint-covers-order-geometry-declination",
     "statement": "坐标转换顺序、波束几何和磁偏角版本都进入派生指纹",
     "enforcement": "earth 指纹含 transform_order 与 declination 版本；"
     "instr 指纹含 beam_geometry_id"},
]


def load(name: str) -> Dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _record(rec_id: str, kind: str, value_raw, unit: str,
            frame: Optional[str] = None, source: Optional[str] = None,
            ping=None, beam=None, cell=None, t=None) -> Dict:
    numeric, status, diagnostic = normalize(value_raw, unit)
    return {"id": rec_id, "kind": kind, "ping": ping, "beam": beam,
            "cell": cell, "t": t, "value_raw": str(value_raw),
            "unit": unit, "frame": frame, "numeric": numeric,
            "status": status, "diagnostic": diagnostic, "source": source}


def records_from_baseline(data: Dict, vid: int = 1) -> List[Dict]:
    records: List[Dict] = []
    add = lambda **kw: records.append({**kw, "valid_from": vid})

    add(**_record("cfg:geometry", "transducer_config",
                  json.dumps(data["config"], sort_keys=True), "json",
                  source=data["config"]["geometry_id"]))
    dec = data["magnetic_declination"]
    add(**_record("dec:0", "magnetic_declination",
                  dec["value_deg"], "deg", frame="magnetic->true",
                  source=dec["version"]))
    mnt = data["mounting_angle"]
    add(**_record("mnt:0", "mounting_angle",
                  mnt["value_deg"], "deg", frame="ship",
                  source=mnt["version"]))

    for sample in data["attitude_samples"]:
        t = sample["t"]
        tag = f"t{t:g}".replace(".", "p")
        add(**_record(f"hdg:{tag}", "heading", sample["heading_mag_deg"],
                      "deg", frame="magnetic", t=t))
        add(**_record(f"pit:{tag}", "pitch", sample["pitch_deg"],
                      "deg", frame="ship", t=t))
        add(**_record(f"rol:{tag}", "roll", sample["roll_deg"],
                      "deg", frame="ship", t=t))

    for ping_info in data["pings"]:
        ping, t = ping_info["ping"], ping_info["t"]
        bv = data["beam_velocity"]["values"][str(ping)]
        cor = data["correlation"]["values"][str(ping)]
        amp = data["amplitude"]["values"][str(ping)]
        flags = data["bad_beam_flag"]["values"][str(ping)]
        bt = data["bottom_track"]["values"][str(ping)]
        for beam in range(1, 5):
            for cell in range(data["config"]["cells"]):
                add(**_record(f"rv:p{ping}:b{beam}:c{cell}",
                              "beam_velocity", bv[beam - 1][cell],
                              data["beam_velocity"]["unit"],
                              frame="beam", source=data["source"],
                              ping=ping, beam=beam, cell=cell, t=t))
                add(**_record(f"cor:p{ping}:b{beam}:c{cell}",
                              "correlation", cor[beam - 1][cell],
                              data["correlation"]["unit"],
                              frame="beam", source=data["source"],
                              ping=ping, beam=beam, cell=cell, t=t))
                add(**_record(f"amp:p{ping}:b{beam}:c{cell}",
                              "amplitude", amp[beam - 1][cell],
                              data["amplitude"]["unit"],
                              frame="beam", source=data["source"],
                              ping=ping, beam=beam, cell=cell, t=t))
            add(**_record(f"flg:p{ping}:b{beam}", "bad_beam_flag",
                          str(flags[beam - 1]).lower(),
                          data["bad_beam_flag"]["unit"],
                          frame="beam", source=data["source"],
                          ping=ping, beam=beam, t=t))
        add(**_record(f"bt:p{ping}:stbd", "bottom_track",
                      bt["ship:stbd_ms"], data["bottom_track"]["unit"],
                      frame="ship:stbd", source=data["source"],
                      ping=ping, t=t))
        add(**_record(f"bt:p{ping}:fore", "bottom_track",
                      bt["ship:fore_ms"], data["bottom_track"]["unit"],
                      frame="ship:fore", source=data["source"],
                      ping=ping, t=t))

    for override in data["amplitude"].get("overrides", []):
        rec = _record(f"amp:p{override['ping']}:b{override['beam']}:"
                      f"c{override['cell']}", "amplitude",
                      override["value_raw"], override["unit"],
                      frame="beam", source=data["source"],
                      ping=override["ping"], beam=override["beam"],
                      cell=override["cell"],
                      t=next(p["t"] for p in data["pings"]
                             if p["ping"] == override["ping"]))
        # same stable id: supersede the generated record within v1 import
        records = [r for r in records if r["id"] != rec["id"]]
        records.append({**rec, "valid_from": vid})
    return records


def import_baseline(store: Store) -> Dict:
    """Create the raw-import version and the first fully derived version."""
    if store.current_version() is not None:
        return {"skipped": True, "version": store.current_version()}
    data = load("baseline.json")
    with store.transaction():
        vid = store.insert_version(None, "baseline-import", {},
                                   status="building")
        for rec in records_from_baseline(data, vid):
            store.insert_raw(rec)
        store.insert_event("import", vid, {
            "step_id": "import-baseline",
            "case_id": data["case_id"], "source": data["source"],
            "separate_kinds": ["beam_velocity", "correlation", "amplitude",
                               "bad_beam_flag", "heading", "pitch", "roll",
                               "magnetic_declination", "mounting_angle",
                               "bottom_track", "transducer_config"],
            "note": "原始表示、单位与坐标基准原样保留"})
        store.complete_version(vid)
    result = engine.build_version(
        store, vid, "baseline-recompute",
        param_updates=data["params"],
        events=[("recompute", {"step_id": "baseline-recompute",
                               "rule_version": engine.RULE_VERSION,
                               "judgments": JUDGMENTS})]
        + [("judgment", j) for j in JUDGMENTS])
    return {"raw_version": vid, "version": result["version"],
            "changed": result["changed"]}


def steps() -> List[Dict]:
    return load("steps.json")["steps"]


def _step(step_id: str) -> Dict:
    for step in steps():
        if step["id"] == step_id:
            return step
    raise KeyError(step_id)


def run_step(store: Store, step_id: str) -> Dict:
    """Apply one replay step; idempotent."""
    if store.event_applied(step_id):
        return {"skipped": True, "step_id": step_id,
                "version": store.current_version()}
    step = _step(step_id)
    current = store.current_version()

    if step["kind"] == "param_revision":
        old = store.effective_raw(current, kind="mounting_angle")[-1]
        rec = step["record"]
        result = engine.build_version(
            store, current, f"param-revision:{step_id}",
            raw_changes=[{"supersede": old["id"], "record": rec}],
            events=[("param_revision", {
                "step_id": step_id, "parameter": "mounting_angle",
                "old_record": {"id": old["id"], "value_raw": old["value_raw"],
                               "version": old["source"]},
                "new_record": {"id": rec["id"], "value_raw": rec["value_raw"],
                               "version": rec["source"]},
                "affected": None,
                "rule": "only downstream nodes get recomputed",
                "judgment": "安装角版本进入 ship 派生指纹"})])
        store.conn.execute(
            "UPDATE events SET payload=json_set(payload, '$.affected', ?) "
            "WHERE version_id=? AND kind='param_revision'",
            (json.dumps(result["changed"]), result["version"]))
        store.conn.commit()
        return {"step_id": step_id, **result}

    if step["kind"] == "source_change":
        rec = step["record"]
        fragment = {"raw_id": rec["id"], "supersedes": step["supersede"],
                    "ping": rec["ping"], "beam": rec["beam"],
                    "value_raw": rec["value_raw"], "source": rec["source"]}
        result = engine.build_version(
            store, current, f"source-change:{step_id}",
            raw_changes=[{"supersede": step["supersede"],
                          "record": rec}],
            events=[("source_change", {
                "step_id": step_id, "input_fragment": fragment,
                "affected": None,
                "judgment": "只改动一个直接来源，传播范围由依赖图决定"})])
        store.conn.execute(
            "UPDATE events SET payload=json_set(payload, '$.affected', ?) "
            "WHERE version_id=? AND kind='source_change'",
            (json.dumps(result["changed"]), result["version"]))
        store.conn.commit()
        return {"step_id": step_id, **result}

    if step["kind"] == "decision_conflict":
        base = current
        sessions = {user: store.create_session(user, base)
                    for user in step["users"]}
        # first user saves cleanly
        first = engine.save_decision(
            store, sessions[step["users"][0]]["id"], step["node"],
            step["choices"][0])
        conflict = None
        try:
            engine.save_decision(
                store, sessions[step["users"][1]]["id"], step["node"],
                step["choices"][1])
        except ConflictError as exc:
            conflict = {"session": exc.session["id"],
                        "base_version": exc.session["base_version"],
                        "current_version": exc.current,
                        "diff_nodes": sorted(exc.diff.keys())}
        if conflict is None:
            raise RuntimeError("expected conflict did not happen")
        resolved = engine.build_version(
            store, store.current_version(),
            f"conflict-resolution:{step_id}",
            decisions={step["node"]: step["choices"][1]},
            events=[("conflict", {"step_id": step_id + ":detected",
                                  **conflict}),
                    ("conflict_resolution", {
                        "step_id": step_id,
                        "node": step["node"],
                        "base_version": base,
                        "choices": [
                            {"user": step["users"][0],
                             "adopted_candidate": step["choices"][0]},
                            {"user": step["users"][1],
                             "adopted_candidate": step["choices"][1]}],
                        "resolution": step["resolution"],
                        "adopted_candidate": step["choices"][1],
                        "retained": "未选候选仍保留在 instr 节点中",
                        "judgment": "三波束解与四波束解不混为一值；"
                                    "冲突可解决且不覆盖历史"})])
        return {"step_id": step_id,
                "first_version": first["version"],
                "conflict": conflict,
                "version": resolved["version"],
                "changed": resolved["changed"]}

    raise ValueError(f"unknown step kind {step['kind']}")


def run_scenario(store: Store) -> List[Dict]:
    import_baseline(store)
    outcomes = []
    for step in steps():
        outcomes.append(run_step(store, step["id"]))
    return outcomes
