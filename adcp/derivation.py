"""Dependency-graph ADCP derivation."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .fingerprint import derivation_fingerprint, stable_hash
from .geometry import Geometry, instrument_candidates, rdi_geometry
from .repository import current_run, latest_event_seq
from .store import transaction, utc_now
from .transforms import instrument_to_ship, interpolate_attitude, ship_to_earth
from .units import convert_angle, convert_velocity


def _rule(events: Sequence[Dict[str, Any]], selection_event: Optional[int]) -> Dict[str, Any]:
    order = "zyx"
    for e in reversed(events):
        if e["event_type"] == "set_transform_order" and e["seq"] <= (selection_event or e["seq"] + 1):
            order = e_payload(e).get("order", order)
            break
    return {"rule_version": "adcp-coordinate-evidence-v1", "rotation_order": order,
            "selection_event_seq": selection_event,
            "stages": ["beam->instrument", "instrument->ship", "ship->earth"]}


def e_payload(event: Dict[str, Any]) -> Dict[str, Any]:
    import json
    return json.loads(event["payload_json"])


def _geometry(state: Dict[str, Any]) -> Geometry:
    g = state["geom"][-1]
    return rdi_geometry(g["angle_unit"], g["beam_angle"], g["geometry_version"])


def _index(records: Sequence[Dict[str, Any]], keys: Sequence[str]) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    return {tuple(r[k] for k in keys): r for r in records}


def _attitude_for_ping(state: Dict[str, Any], ping_time: float) -> Dict[str, Any]:
    samples = []
    diagnostics = []
    for r in state["attitude"]:
        if r["raw_value"] is None:
            diagnostics.append({"code": "unconvertible-attitude", "record_id": r["attitude_record_id"],
                                "raw_repr": r["raw_repr"], "unit": r["raw_unit"]})
        else:
            samples.append({"record_id": r["attitude_record_id"], "ping_time": r["ping_time"],
                            "field": r["field"], "raw_value": r["raw_value"], "raw_unit": r["raw_unit"]})
    fields = {}
    bounds = []
    methods = set()
    crossed = False
    for field in ("heading", "pitch", "roll"):
        result = interpolate_attitude([s for s in samples if s["field"] == field], ping_time)
        methods.add(result.method)
        if result.bounded_by[0]:
            bounds.extend(result.bounded_by)
        crossed = crossed or result.crossed_zero_heading
        if result.diagnostic:
            diagnostics.append({"code": result.method, "field": field, "message": result.diagnostic,
                                "bounded_by": list(result.bounded_by)})
        if field in result.values:
            fields[field] = result.values[field]
    return {"fields": fields, "methods": sorted(methods), "bounds": sorted(set(bounds)),
            "crossed_zero_heading": crossed, "diagnostics": diagnostics, "samples": samples}


def _numeric_q(record: Dict[str, Any], kind: str) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
    if record.get("raw_value") is None:
        return None, {"code": "unconvertible-value", "record_id": record.get(f"{kind}_record_id"),
                      "raw_repr": record.get("raw_repr"), "unit": record.get("raw_unit")}
    try:
        if kind in ("beam", "bt"):
            value, unit = convert_velocity(record["raw_value"], record["raw_unit"])
        else:
            value, unit = float(record["raw_value"]), record["raw_unit"]
        return value, None
    except Exception as exc:
        return None, {"code": "unconvertible-value", "message": str(exc), "raw_repr": record.get("raw_repr")}


def _dep(table: str, records: Sequence[Dict[str, Any]], id_col: str) -> List[Dict[str, str]]:
    out = []
    for r in records:
        out.append({"source_table": table, "record_id": r[id_col], "version_hash": r["version_hash"]})
    return out


def _candidate_decisions(events: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    import json
    decisions = {}
    for event in events:
        if event["event_type"] != "select_candidate":
            continue
        p = json.loads(event["payload_json"])
        decisions[(p["ping_id"], p["cell_id"])] = {"seq": event["seq"], **p}
    return decisions


def _derive_cell(state, geometry, geom_rec, install_rec, decl_rec, rule, ping, cell, decisions):
    ping_id, cell_id = ping["ping_id"], cell["cell_id"]
    beam_idx = _index(state["beam"], ("ping_id", "cell_id", "beam"))
    corr_idx = _index(state["correlation"], ("ping_id", "cell_id", "beam"))
    amp_idx = _index(state["amplitude"], ("ping_id", "cell_id", "beam"))
    flag_idx = _index(state["flag"], ("ping_id", "cell_id", "beam"))
    cell_idx = _index(state["cell"], ("ping_id", "cell_id"))
    beams = geometry.beam_order
    measurements, flags, quality, diagnostics, dependencies = [], {}, [], [], []
    dependencies.append({"source_table": "geom_versions", "record_id": geom_rec["geom_id"],
                         "version_hash": geom_rec["version_hash"]})
    dependencies.append({"source_table": "install_versions", "record_id": install_rec["install_id"],
                         "version_hash": install_rec["version_hash"]})
    dependencies.append({"source_table": "declination_versions", "record_id": decl_rec["declination_id"],
                         "version_hash": decl_rec["version_hash"]})

    cellvalid = cell_idx.get((ping_id, cell_id))
    if cellvalid:
        dependencies.append({"source_table": "cell_versions", "record_id": cellvalid["cell_id"],
                             "version_hash": cellvalid["version_hash"]})
        if not cellvalid["valid"]:
            return [], [_exclusion(ping_id, cell_id, "invalid-depth-cell", cellvalid["reason"],
                                   {"cell_version_hash": cellvalid["version_hash"]})], dependencies

    for beam in beams:
        b = beam_idx.get((ping_id, cell_id, beam))
        c = corr_idx.get((ping_id, cell_id, beam))
        a = amp_idx.get((ping_id, cell_id, beam))
        f = flag_idx.get((ping_id, cell_id, beam))
        if b:
            dependencies.append({"source_table": "beam_record_versions", "record_id": b["beam_record_id"],
                                 "version_hash": b["version_hash"]})
            measurements.append({"record_id": b["beam_record_id"], "beam": beam,
                                 "raw_repr": b["raw_repr"], "raw_value": b["raw_value"],
                                 "raw_unit": b["raw_unit"]})
        else:
            diagnostics.append({"code": "missing-beam", "beam": beam})
        if c:
            dependencies.append({"source_table": "correlation_record_versions", "record_id": c["correlation_record_id"],
                                 "version_hash": c["version_hash"]})
        if a:
            dependencies.append({"source_table": "amplitude_record_versions", "record_id": a["amplitude_record_id"],
                                 "version_hash": a["version_hash"]})
        if f:
            dependencies.append({"source_table": "flag_record_versions", "record_id": f["flag_record_id"],
                                 "version_hash": f["version_hash"]})
            flags[beam] = f["flag"]
        quality.append({"beam": beam,
                        "correlation": None if not c else {"raw_repr": c["raw_repr"], "unit": c["raw_unit"],
                                                            "value": c["raw_value"]},
                        "amplitude": None if not a else {"raw_repr": a["raw_repr"], "unit": a["raw_unit"],
                                                          "value": a["raw_value"]},
                        "flag": f["flag"] if f else "ok",
                        "measurement": None if not b else {"raw_repr": b["raw_repr"], "unit": b["raw_unit"]}})

    candidates = instrument_candidates(geometry, flags, measurements)
    if not candidates:
        return [], [_exclusion(ping_id, cell_id, "insufficient-valid-beams",
                               "fewer than three beams can produce a legal velocity",
                               {"quality": quality, "diagnostics": diagnostics})], dependencies

    attitude = _attitude_for_ping(state, ping["ping_time"])
    for rec_id in attitude["bounds"]:
        rec = next((r for r in state["attitude"] if r["attitude_record_id"] == rec_id), None)
        if rec:
            dependencies.append({"source_table": "attitude_record_versions", "record_id": rec_id,
                                 "version_hash": rec["version_hash"]})
    diagnostics.extend(attitude["diagnostics"])
    pitch, roll, heading = (attitude["fields"].get("pitch"), attitude["fields"].get("roll"),
                            attitude["fields"].get("heading"))
    if None in (pitch, roll, heading):
        exclusion = _exclusion(ping_id, cell_id, "attitude-unavailable",
                               "do not interpolate attitude across a source gap",
                               {"attitude": {"methods": attitude["methods"],
                                             "crossed_zero_heading": attitude["crossed_zero_heading"],
                                             "diagnostics": attitude["diagnostics"]}})
        instrument_values, earth_available = [], False
    else:
        exclusion = None
        earth_available = True

    install_rad, _ = convert_angle(install_rec["angle"], install_rec["angle_unit"])
    decl_rad = math.radians(decl_rec["degrees"])
    order = rule["rotation_order"]
    values = []
    decision = decisions.get((ping_id, cell_id))
    selected = decision["candidate_id"] if decision else None
    legal_ids = set(candidates)
    for cid, candidate in candidates.items():
        if selected:
            if cid == selected:
                status = "accepted"
            elif decision.get("decision") == "reject" and cid != selected:
                status = "rejected"
            else:
                status = "retained-alternative"
        else:
            status = candidate["status"]
        dep_fingerprint = derivation_fingerprint(dependencies + [{"record_id": cid, "version_hash": "candidate"}], rule)
        values.append(_value(ping_id, cell_id, "instrument", cid, status,
                             {"vector_mps": candidate["vector_mps"], "axes": ["x", "y", "z"],
                              "coordinate_frame": "instrument", "velocity_unit": "m/s",
                              "beams": candidate["beams"], "mode": candidate["mode"],
                              "ambiguity": candidate.get("ambiguity"),
                              "residual_mps": candidate.get("residual_mps"),
                              "beam_quality": quality, "fingerprint_basis": ["beam geometry", "beam values"]},
                             dependencies, rule))
        if earth_available:
            ship_vec, stages = instrument_to_ship(candidate["vector_mps"], math.radians(pitch), math.radians(roll), order)
            values.append(_value(ping_id, cell_id, "ship", cid, status,
                                 {"vector_mps": ship_vec.tolist(), "axes": ["starboard", "forward", "up"],
                                  "coordinate_frame": "ship", "velocity_unit": "m/s",
                                  "stages": stages,
                                  "attitude": {"heading_deg": heading, "pitch_deg": pitch, "roll_deg": roll,
                                               "interpolation": attitude["methods"],
                                               "crossed_zero_heading": attitude["crossed_zero_heading"],
                                               "bounded_by": attitude["bounds"]}},
                                 dependencies, rule))
            heading_true = math.radians(heading) + decl_rad + install_rad
            earth_vec = ship_to_earth(ship_vec, heading_true)
            values.append(_value(ping_id, cell_id, "earth", cid, status,
                                 {"vector_mps": earth_vec.tolist(), "axes": ["east", "north", "up"],
                                  "coordinate_frame": "earth", "velocity_unit": "m/s",
                                  "rotation_order": order,
                                  "declination": {"degrees": decl_rec["degrees"],
                                                  "model_version": decl_rec["model_version"]},
                                  "installation": {"angle": install_rec["angle"],
                                                   "angle_unit": install_rec["angle_unit"]}},
                                 dependencies, rule))
    if not earth_available:
        values.append(_value(ping_id, cell_id, "earth", "none", "excluded",
                             {"reason": exclusion["reason"], "evidence": exclusion["evidence"]},
                             dependencies, rule))
    return values, ([exclusion] if exclusion else []), dependencies


def _exclusion(ping_id: str, cell_id: str, code: str, reason: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
    return {"exclusion_id": "ex_" + stable_hash([ping_id, cell_id, code, reason, evidence], 18),
            "ping_id": ping_id, "cell_id": cell_id, "code": code, "reason": reason, "evidence": evidence}


def _value(ping_id, cell_id, layer, candidate_id, status, payload, dependencies, rule):
    from .fingerprint import derived_value_id
    value_id = derived_value_id("DS_ADCP_2026", ping_id, cell_id, layer, candidate_id)
    fp_inputs = dependencies + [{"record_id": layer, "version_hash": stable_hash(payload, 18)}]
    return {"value_id": value_id, "ping_id": ping_id, "cell_id": cell_id, "layer": layer,
            "candidate_id": candidate_id, "status": status, "payload": payload,
            "fingerprint": derivation_fingerprint(fp_inputs, rule), "dependencies": dependencies}


def _bottom_track(state, geometry, geom_rec, install_rec, decl_rec, rule, ping):
    beam_idx = _index(state["bt"], ("ping_id", "beam"))
    flag_idx = _index(state["flag"], ("ping_id", "cell_id", "beam"))
    measurements, flags, deps = [], {}, [
        {"source_table": "geom_versions", "record_id": geom_rec["geom_id"], "version_hash": geom_rec["version_hash"]},
        {"source_table": "install_versions", "record_id": install_rec["install_id"], "version_hash": install_rec["version_hash"]},
        {"source_table": "declination_versions", "record_id": decl_rec["declination_id"], "version_hash": decl_rec["version_hash"]}]
    for beam in geometry.beam_order:
        rec = beam_idx.get((ping["ping_id"], beam))
        if rec:
            measurements.append({"record_id": rec["bt_record_id"], "beam": beam, "raw_repr": rec["raw_repr"],
                                 "raw_value": rec["raw_value"], "raw_unit": rec["raw_unit"]})
            deps.append({"source_table": "bt_record_versions", "record_id": rec["bt_record_id"],
                         "version_hash": rec["version_hash"]})
        f = flag_idx.get((ping["ping_id"], "bottom", beam))
        flags[beam] = f["flag"] if f else "ok"
    candidates = instrument_candidates(geometry, flags, measurements)
    attitude = _attitude_for_ping(state, ping["ping_time"])
    if not candidates or len(attitude["fields"]) < 3:
        return [], [_exclusion(ping["ping_id"], "bottom-track", "bottom-track-unavailable",
                               "bottom track requires complete beams and bounded attitude",
                               {"candidate_count": len(candidates), "attitude_methods": attitude["methods"]})]
    cid = next(iter(candidates))
    cand = candidates[cid]
    order = rule["rotation_order"]
    ship, _ = instrument_to_ship(cand["vector_mps"], math.radians(attitude["fields"]["pitch"]),
                                 math.radians(attitude["fields"]["roll"]), order)
    true_heading = math.radians(attitude["fields"]["heading"]) + math.radians(decl_rec["degrees"])
    true_heading += convert_angle(install_rec["angle"], install_rec["angle_unit"])[0]
    earth = ship_to_earth(ship, true_heading)
    return [_value(ping["ping_id"], "bottom-track", "earth", cid, cand["status"],
                   {"vector_mps": earth.tolist(), "coordinate_frame": "earth", "velocity_unit": "m/s",
                    "instrument_mps": cand["vector_mps"], "ship_mps": ship.tolist(),
                    "candidate_modes": {k: v["mode"] for k, v in candidates.items()}}, deps, rule)], []


def changed_records(conn, dataset: str, previous_seq: int, new_seq: int) -> List[Dict[str, str]]:
    out = []
    for table, id_col in [
        ("beam_record_versions", "beam_record_id"), ("bt_record_versions", "bt_record_id"),
        ("correlation_record_versions", "correlation_record_id"),
        ("amplitude_record_versions", "amplitude_record_id"), ("flag_record_versions", "flag_record_id"),
        ("attitude_record_versions", "attitude_record_id"), ("geom_versions", "geom_id"),
        ("install_versions", "install_id"), ("declination_versions", "declination_id"),
        ("cell_versions", "cell_id")]:
        for r in conn.execute(
            f"SELECT '{table}' AS source_table,{id_col} AS record_id,version_hash FROM {table} WHERE dataset_id=? AND first_seq>? AND first_seq<=?",
                (dataset, previous_seq, new_seq)):
            out.append(dict(r))
    return out


def derive_run(conn, dataset: str, selection_event_seq: Optional[int] = None,
               parent_run_id: Optional[str] = None, mark_failed: bool = False) -> Dict[str, Any]:
    state = None
    rule = None
    seq = latest_event_seq(conn, dataset)
    events = [dict(r) for r in conn.execute("SELECT * FROM events WHERE dataset_id=? AND seq<=? ORDER BY seq",
                                            (dataset, seq))]
    state = {"events": events, "seq": seq}
    for name, (table, _) in __import__("adcp.store", fromlist=["SOURCE_TABLES"]).SOURCE_TABLES.items():
        state[name] = [dict(r) for r in conn.execute(f"SELECT * FROM {table} WHERE dataset_id=?", (dataset,))]
    state["pings"] = [dict(r) for r in conn.execute("SELECT * FROM pings WHERE dataset_id=? ORDER BY ordinal", (dataset,))]
    state["cells"] = [dict(r) for r in conn.execute("SELECT * FROM cells WHERE dataset_id=? ORDER BY ordinal", (dataset,))]
    rule = _rule(events, selection_event_seq)
    geometry = _geometry(state)
    geom_rec, install_rec, decl_rec = state["geom"][-1], state["install"][-1], state["declination"][-1]
    decisions = _candidate_decisions(events)

    new_values, new_exclusions, all_deps = [], [], []
    affected = set()
    previous = current_run(conn, dataset)
    previous_seq = None
    global_event = previous is None
    if previous:
        previous_seq = previous["base_event_seq"]
        changed = changed_records(conn, dataset, previous_seq, seq)
        changed_keys = {(r["source_table"], r["record_id"]) for r in changed}
        old_deps = conn.execute("SELECT value_id,source_table,source_record_id FROM derived_dependencies WHERE run_id=?",
                                (previous["run_id"],)).fetchall()
        affected = {d["value_id"] for d in map(dict, old_deps)
                    if (d["source_table"], d["source_record_id"]) in changed_keys}
        recent_types = {r["event_type"] for r in conn.execute(
            "SELECT event_type FROM events WHERE dataset_id=? AND seq>? AND seq<=?", (dataset, previous_seq, seq))}
        global_event = bool(recent_types.intersection({"set_transform_order", "select_candidate", "resolve_conflict"}))
    if parent_run_id is None and previous:
        parent_run_id = previous["run_id"]

    for ping in state["pings"]:
        for cell in state["cells"]:
            vals, excs, deps = _derive_cell(state, geometry, geom_rec, install_rec, decl_rec, rule, ping, cell, decisions)
            all_deps.extend(deps)
            for v in vals:
                old_matches = _matching_old_ids(previous["run_id"], v, conn) if previous else []
                if previous and not global_event and not affected.intersection(old_matches):
                    old = conn.execute("SELECT value_json FROM derived_values WHERE run_id=? AND value_id=?",
                                       (previous["run_id"], v["value_id"])).fetchone()
                    if old:
                        import json
                        v = json.loads(old["value_json"])
                        v["propagation"] = "still-valid: dependency hashes unchanged"
                else:
                    v["propagation"] = "recomputed: changed dependency or rule"
                new_values.append(v)
            new_exclusions.extend(excs)
        bt_vals, bt_excs = _bottom_track(state, geometry, geom_rec, install_rec, decl_rec, rule, ping)
        new_values.extend(bt_vals)
        new_exclusions.extend(bt_excs)

    unique_deps = {(d["source_table"], d["record_id"], d["version_hash"]) for d in all_deps}
    manifest = [{"source_table": a, "record_id": b, "version_hash": c} for a, b, c in sorted(unique_deps)]
    fingerprint = derivation_fingerprint(manifest, rule)
    run_id = "run_" + stable_hash([dataset, seq, rule, fingerprint, status], 20)
    status = "failed" if mark_failed else "complete"
    diagnostics = []
    with transaction(conn):
        conn.execute(
            "INSERT INTO derivation_runs(run_id,dataset_id,base_event_seq,rule_json,fingerprint,status,created_at,diagnostic_json,replaced_run_id,selection_event_seq) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (run_id, dataset, seq, json_dumps(rule), fingerprint, status, utc_now(), json_dumps(diagnostics),
             parent_run_id, selection_event_seq))
        conn.execute("INSERT INTO derived_inputs(run_id,manifest_json) VALUES(?,?)", (run_id, json_dumps(manifest)))
        if status == "complete":
            for v in new_values:
                conn.execute(
                    "INSERT INTO derived_values(value_id,run_id,ping_id,cell_id,layer,candidate_id,status,value_json,fingerprint) VALUES(?,?,?,?,?,?,?,?,?)",
                    (v["value_id"], run_id, v["ping_id"], v["cell_id"], v["layer"], v["candidate_id"], v["status"],
                     json_dumps(v), v["fingerprint"]))
                for d in v["dependencies"]:
                    conn.execute(
                        "INSERT OR IGNORE INTO derived_dependencies(value_id,run_id,source_table,source_record_id,source_version_hash) VALUES(?,?,?,?,?)",
                        (v["value_id"], run_id, d["source_table"], d["record_id"], d["version_hash"]))
            for e in new_exclusions:
                conn.execute(
                    "INSERT INTO exclusions(exclusion_id,run_id,ping_id,cell_id,code,reason,evidence_json) VALUES(?,?,?,?,?,?,?)",
                    (e["exclusion_id"], run_id, e["ping_id"], e["cell_id"], e["code"], e["reason"], json_dumps(e["evidence"])))
            conn.execute("INSERT INTO current_pointer(dataset_id,run_id) VALUES(?,?) ON CONFLICT(dataset_id) DO UPDATE SET run_id=excluded.run_id",
                         (dataset, run_id))
    return {"run_id": run_id, "status": status, "fingerprint": fingerprint, "values": len(new_values),
            "exclusions": len(new_exclusions), "base_event_seq": seq, "rule": rule}


def _matching_old_ids(run_id: str, new_value: Dict[str, Any], conn) -> List[str]:
    rows = conn.execute(
        "SELECT value_id FROM derived_values WHERE run_id=? AND ping_id=? AND cell_id=? AND layer=? AND candidate_id=?",
        (run_id, new_value["ping_id"], new_value["cell_id"], new_value["layer"], new_value["candidate_id"])).fetchall()
    return [r[0] for r in rows]


def json_dumps(value: Any) -> str:
    import json
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
