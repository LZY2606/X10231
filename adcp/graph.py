"""Dependency graph: sources -> beam-quality -> instrument -> ship -> earth.

Materialised nodes are content-addressed by fingerprint and immutable.
Recompute walks a versioned source snapshot; only nodes whose fingerprint
changes are reported as ``recomputed``. A branch replay reproduces every
historical conclusion because nodes are never overwritten and decisions
point at the exact fingerprint that was accepted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .attitude import AttitudeSeries, GAP, EXTRAP, INVALID
from .casepack import (N_BEAMS, config_v1, GEOMETRY_V1, DECL_V1, INSTALL_V1,
                       ORDER_V1, POLICY_V1, CELLS_V1)
from .fingerprint import fingerprint, rule_versions as rv
from .geometry import TransducerConfig, solve_radial
from .quality import (QualityPolicy, BeamObservation, build_cell_candidates,
                      cell_depth_ok, ADOPTED)
from .transforms import correct_attitude, instrument_to_ship_matrix, ship_to_earth_matrix


# Source record kinds kept separately (storage schema matches this list).
SOURCE_KINDS = ("ping", "beam_velocity", "correlation", "amplitude", "bad_flag",
                "attitude", "bt_velocity", "bt_range", "bt_flag",
                "depth_cell_bounds")


def _cfg_from_rule(payload: dict) -> TransducerConfig:
    return TransducerConfig(
        config_id="geom", n_beams=4,
        beam_angle_deg=payload["beam_angle_deg"],
        azimuths_deg=tuple(payload["azimuths_deg"]),
        radial_sign=payload.get("radial_sign", 1.0),
        convex=True, orientation="down")


@dataclass
class Snapshot:
    sources: dict          # kind -> key -> {version,payload}
    rules: dict            # rule -> {version_id,payload}
    nodes: dict = field(default_factory=dict)       # node_id -> node dict
    index: dict = field(default_factory=dict)       # (ping,cell,beam,candidate) -> node_id

    def rule_ids(self) -> dict:
        return {k: v["version_id"] for k, v in self.rules.items()}

    def rule_payload(self, name: str) -> dict:
        return self.rules[name]["payload"]


def build_snapshot(committed_sources: list[dict], committed_rules: list[dict]) -> Snapshot:
    sources: dict = {k: {} for k in SOURCE_KINDS}
    for r in committed_sources:
        latest = sources.setdefault(r["kind"], {}).get(r["record_key"])
        if latest is None or r["version"] > latest["version"]:
            sources.setdefault(r["kind"], {})[r["record_key"]] = {
                "version": r["version"], "payload": r["payload"]}
    rules = {}
    for r in committed_rules:
        latest = rules.get(r["rule"])
        if latest is None or r["version"] > latest["version"]:
            rules[r["rule"]] = {"version_id": r["payload"].get("version_id"),
                                "payload": r["payload"]}
    return Snapshot(sources=sources, rules=rules)


def _source_versions(snap: Snapshot, keys_by_kind: dict) -> dict:
    out = {}
    for kind, keys in keys_by_kind.items():
        table = snap.sources.get(kind, {})
        out[kind] = {k: table[k]["version"] for k in keys if k in table}
    return out


def _beam_observations(snap: Snapshot, ping_id: str, cell_idx: int):
    obs = []
    for b in range(N_BEAMS):
        rk = f"{ping_id}:c{cell_idx + 1:02d}:b{b + 1}"
        bv = snap.sources["beam_velocity"].get(rk, {}).get("payload", {})
        co = snap.sources["correlation"].get(rk, {}).get("payload", {})
        am = snap.sources["amplitude"].get(rk, {}).get("payload", {})
        flag = snap.sources["bad_flag"].get(f"{ping_id}:b{b + 1}", {}).get("payload", {})
        q = bv.get("q", {})
        radial_si = q["value"] / 100.0 if q.get("unit") == "cm/s" and q.get("value") is not None else None
        corr = co.get("corr", {}).get("value")
        amp = am.get("amp", {})
        obs.append(BeamObservation(
            beam=b, radial_si=radial_si, bad=bool(flag.get("bad", False)),
            corr_counts=corr, amp_counts=amp.get("value"),
            amp_raw=amp.get("raw"), amp_unit=amp.get("unit")))
    return obs


def _attitude_for(snap: Snapshot, ping_id: str):
    samples = [v["payload"] for v in snap.sources.get("attitude", {}).values()]
    samples = [s for s in samples if s.get("heading") is not None]
    series = AttitudeSeries(samples)
    ping = snap.sources["ping"][ping_id]["payload"]
    threshold = snap.rule_payload("quality_policy")["gap_threshold_s"]
    return series.at(ping_id, ping["t"], threshold)


def _dep_fingerprint(snap: Snapshot, kind: str, inputs: dict) -> str:
    rids = snap.rule_ids()
    return fingerprint(kind, inputs, rv(
        beam_geometry=rids.get("geometry", GEOMETRY_V1),
        declination=rids.get("declination", DECL_V1),
        install=rids.get("install", INSTALL_V1),
        transform_order=snap.rules.get("transform_order", {}).get("payload", {}).get(
            "order", ORDER_V1),
        quality_policy=rids.get("quality_policy", POLICY_V1)))


def _candidate_node(snap, ping_id, cell_idx, depth_m, depth_ok, depth_reason,
                    att, candidate, decl_p, install_p, order):
    beams = candidate["beams"]
    src_versions = _source_versions(snap, {
        "beam_velocity": [f"{ping_id}:c{cell_idx + 1:02d}:b{b + 1}" for b in beams],
        "bad_flag": [f"{ping_id}:b{b + 1}" for b in beams],
        "correlation": [f"{ping_id}:c{cell_idx + 1:02d}:b{b + 1}" for b in beams],
        "amplitude": [f"{ping_id}:c{cell_idx + 1:02d}:b{b + 1}" for b in beams],
        "ping": [ping_id]})
    inputs = {"source_versions": src_versions, "beams": beams,
              "candidate": candidate["candidate_id"], "cell_depth_m": depth_m,
              "attitude_t": att.t if att.status == "ok" else None}
    fp = _dep_fingerprint(snap, "earth_velocity_cell", inputs)
    v_inst = candidate["v_instrument"]
    if att.status == "ok":
        corr = correct_attitude(att.heading, att.pitch, att.roll,
                                decl_p["declination_deg"], install_p)
        v_ship = (instrument_to_ship_matrix(corr["pitch"], corr["roll"], order)
                  @ np.asarray(v_inst)).tolist()
        v_earth = (ship_to_earth_matrix(corr["heading_true"])
                   @ np.asarray(v_ship)).tolist()
        att_out = {"status": att.status, "bounding": att.bounding,
                   "gap_seconds": att.gap_seconds, "corrected": corr,
                   "heading_unit": "deg"}
    else:
        v_ship = v_earth = None
        att_out = {"status": att.status, "bounding": att.bounding,
                   "gap_seconds": att.gap_seconds,
                   "diagnostics": att.diagnostics}
    status = candidate["status"]
    if att.status != "ok":
        status = "excluded_attitude_" + att.status
    elif not depth_ok:
        status = "excluded_depth"
    return {
        "node_id": f"n:{ping_id}:c{cell_idx + 1:02d}:{candidate['candidate_id']}",
        "fingerprint": fp, "kind": "earth_velocity_cell",
        "ping_id": ping_id, "cell_idx": cell_idx,
        "payload": {
            "ping_id": ping_id, "cell_idx": cell_idx,
            "depth_m": depth_m, "depth_unit": "m",
            "candidate_id": candidate["candidate_id"],
            "candidate_kind": candidate["kind"], "beams": beams,
            "v_instrument": {"value": v_inst, "frame": "instrument",
                             "unit": "m/s", "axes": ["forward", "starboard", "up"]},
            "v_ship": ({"value": v_ship, "frame": "ship", "unit": "m/s",
                        "axes": ["forward", "starboard", "up"]}
                       if v_ship is not None else None),
            "v_earth": ({"value": v_earth, "frame": "earth", "unit": "m/s",
                         "datum": "ENU", "axes": ["east", "north", "up"]}
                        if v_earth is not None else None),
            "error_velocity_mps": candidate.get("error_velocity"),
            "attitude": att_out, "candidate_status": candidate["status"],
            "status": status,
            "depth_check": {"ok": depth_ok, "reason": depth_reason},
            "rule_versions": {
                "beam_geometry": snap.rule_ids().get("geometry"),
                "declination": snap.rule_ids().get("declination"),
                "install": snap.rule_ids().get("install"),
                "transform_order": order,
                "quality_policy": snap.rule_ids().get("quality_policy")},
            "source_versions": src_versions,
        }}


def _bt_node(snap: Snapshot, ping_id: str, att, cfg: TransducerConfig,
             decl_p, install_p, order):
    radial, flags = [], []
    for b in range(N_BEAMS):
        p = snap.sources["bt_velocity"].get(f"{ping_id}:b{b + 1}", {}).get("payload", {})
        f = snap.sources["bt_flag"].get(ping_id, {}).get("payload", {})
        q = p.get("q", {})
        radial.append(q["value"] / 100.0 if q.get("unit") == "cm/s" else None)
        flags.append(bool((f.get("flags") or [False] * N_BEAMS)[b]))
    good = [b for b in range(N_BEAMS) if not flags[b] and radial[b] is not None]
    src_versions = _source_versions(snap, {
        "bt_velocity": [f"{ping_id}:b{b + 1}" for b in good],
        "bt_flag": [ping_id], "ping": [ping_id]})
    inputs = {"source_versions": src_versions, "beams": good}
    fp = _dep_fingerprint(snap, "bottom_track_earth", inputs)
    payload = {"ping_id": ping_id, "beams_radial": radial, "radial_unit": "m/s",
               "bad_flags": flags, "attitude": None, "rule_versions": {
                   "beam_geometry": snap.rule_ids().get("geometry"),
                   "declination": snap.rule_ids().get("declination"),
                   "install": snap.rule_ids().get("install"),
                   "transform_order": order},
               "source_versions": src_versions}
    if len(good) >= 3 and att.status == "ok":
        sol = solve_radial(cfg, [radial[b] for b in good], tuple(good))
        corr = correct_attitude(att.heading, att.pitch, att.roll,
                                decl_p["declination_deg"], install_p)
        v_ship = (instrument_to_ship_matrix(corr["pitch"], corr["roll"], order)
                  @ np.asarray(sol["velocity"])).tolist()
        v_earth = (ship_to_earth_matrix(corr["heading_true"])
                   @ np.asarray(v_ship)).tolist()
        payload.update({
            "v_instrument": {"value": sol["velocity"], "frame": "instrument",
                             "unit": "m/s"},
            "v_ship": {"value": v_ship, "frame": "ship", "unit": "m/s"},
            "v_earth": {"value": v_earth, "frame": "earth", "datum": "ENU",
                        "unit": "m/s"},
            "kind": sol["kind"], "status": "adopted" if sol["kind"] == "4beam"
            else "adopted_3beam"})
    else:
        payload.update({"v_instrument": None, "v_ship": None, "v_earth": None,
                        "status": "undetermined",
                        "diagnostics": [f"{len(good)} valid bottom-track beams"]})
    return {"node_id": f"n:{ping_id}:bottomtrack", "fingerprint": fp,
            "kind": "bottom_track_earth", "ping_id": ping_id, "payload": payload}


def materialize(snap: Snapshot) -> list[dict]:
    """Compute every derived node for the snapshot (deterministic)."""
    cfg = _cfg_from_rule(snap.rule_payload("geometry"))
    policy_p = snap.rule_payload("quality_policy")
    policy = QualityPolicy(policy_p["corr_min_counts"], policy_p["amp_min_counts"],
                           policy_p["err_velocity_max"])
    cells_p = snap.rule_payload("cells")
    decl_p = snap.rule_payload("declination")
    install_p = snap.rule_payload("install")
    order = snap.rule_payload("transform_order")["order"]
    depths = cells_p["cell_depths_m"]
    blank = cells_p.get("blanking_m", 0.0)
    nodes: list[dict] = []
    for ping_rec in sorted(snap.sources["ping"].values(),
                           key=lambda r: r["payload"]["t"]):
        ping_id = ping_rec["payload"]["ping_id"]
        att = _attitude_for(snap, ping_id)
        bt_range = (snap.sources["bt_range"].get(ping_id, {})
                    .get("payload", {}).get("range", {}).get("value"))
        for ci, depth in enumerate(depths):
            obs = _beam_observations(snap, ping_id, ci)
            result = build_cell_candidates(cfg, obs, policy)
            depth_ok, depth_reason = cell_depth_ok(depth, bt_range, blank)
            for cand in result["candidates"]:
                nodes.append(_candidate_node(
                    snap, ping_id, ci, depth, depth_ok, depth_reason,
                    att, cand, decl_p, install_p, order))
        nodes.append(_bt_node(snap, ping_id, att, cfg, decl_p, install_p, order))
    return nodes


# ----- recompute / invalidation reporting -----------------------------
def recompute(old_snap: Snapshot | None, new_snap: Snapshot) -> dict:
    """Propagate changes; report unchanged, recomputed and new nodes."""
    old_map = {n["node_id"]: n for n in (old_snap.nodes.values() if old_snap else [])}
    new_nodes = materialize(new_snap)
    unchanged, recomputed, added = [], [], []
    for n in new_nodes:
        prev = old_map.get(n["node_id"])
        if prev is None:
            added.append(n["node_id"])
        elif prev["fingerprint"] == n["fingerprint"]:
            unchanged.append(n["node_id"])
        else:
            recomputed.append(n["node_id"])
    new_snap.nodes = {n["node_id"]: n for n in new_nodes}
    return {"unchanged": sorted(unchanged), "recomputed": sorted(recomputed),
            "added": sorted(added), "nodes": new_snap.nodes}


def explain(nodes: dict, node_id: str) -> dict:
    """Human/machine readable reason a node is valid or must recompute."""
    n = nodes.get(node_id)
    if not n:
        return {"node_id": node_id, "state": "absent"}
    return {
        "node_id": node_id, "fingerprint": n["fingerprint"],
        "state": "valid",
        "bound_inputs": n["payload"].get("source_versions", {}),
        "rule_versions": n["payload"].get("rule_versions", {}),
        "note": ("fingerprint binds beam geometry, declination, install "
                 "version, transform order, attitude window and source versions"),
    }


# ----- event replay ----------------------------------------------------
def effective_source_overrides(events: list[dict]) -> dict:
    """Fold SOURCE_FIX events down to {kind: {key: {version,payload}}}."""
    overrides: dict = {}
    for e in events:
        if e["type"] != "source_fix":
            continue
        for rec in e["payload"].get("sources", []):
            overrides.setdefault(rec["kind"], {})[rec["record_key"]] = rec
    return overrides


def effective_rule_override(events: list[dict]) -> dict:
    out: dict = {}
    for e in events:
        if e["type"] == "rule_revision":
            out[e["payload"]["rule"]] = e["payload"]["record"]
    return out


def project_state(base_snapshot: Snapshot, events: list[dict]) -> Snapshot:
    """Apply branch events onto a baseline snapshot, then materialise."""
    sources = {k: dict(v) for k, v in base_snapshot.sources.items()}
    for kind, table in effective_source_overrides(events).items():
        sources.setdefault(kind, {}).update(table)
    rules = {k: {"version_id": v["version_id"], "payload": dict(v["payload"])}
             for k, v in base_snapshot.rules.items()}
    for rule, rec in effective_rule_override(events).items():
        rules[rule] = {"version_id": rec["payload"].get("version_id"),
                       "payload": rec["payload"]}
    snap = Snapshot(sources=sources, rules=rules)
    snap.nodes = {n["node_id"]: n for n in materialize(snap)}
    return snap


def accepted_decisions(events: list[dict]) -> dict:
    """Map node_id -> {candidate_id, fingerprint} from decision events."""
    accepted: dict = {}
    for e in events:
        if e["type"] == "decision":
            p = e["payload"]
            accepted[p["node_id"]] = {"candidate_id": p["candidate_id"],
                                      "fingerprint": p["fingerprint"],
                                      "event_id": e["event_id"],
                                      "verdict": p.get("verdict", "accepted")}
    return accepted
