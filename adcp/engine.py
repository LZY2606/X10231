"""Dependency-graph propagation engine.

Each derivation records its inputs as (kind, natural_key, content_id,
fingerprint, rule_id) tuples. When a source or rule changes, the graph is
walked bottom-up: only derivations whose input fingerprint actually changed
are recomputed, and the explanation says exactly which edge fired.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from . import transforms as T
from .store import Store, content_hash, source_uid
from .units import quantity

# ---------------------------------------------------------------- ingestion

SOURCE_KINDS = (
    "beam_velocity", "correlation", "amplitude", "bad_beam",
    "attitude", "declination", "mounting", "geometry", "bottom_track",
    "cell_layout", "ping_clock",
)


def natural_key(kind: str, ping: Optional[str] = None,
                cell: Optional[int] = None, beam: Optional[int] = None) -> str:
    parts = [kind]
    if ping is not None:
        parts.append(str(ping))
    if cell is not None:
        parts.append(f"cell{int(cell)}")
    if beam is not None:
        parts.append(f"b{int(beam)}")
        return ":".join(parts)
    return ":".join(parts)


def fingerprint_for(payload: dict) -> str:
    return content_hash("fp", "payload", "v1", payload)


@dataclass
class _Cfg:
    corr_floor: float = 60.0
    corr_good: float = 150.0
    max_attitude_gap_s: float = 2.0
    earth_order: str = T.EARTH_DEFAULT_RULE_ID


def _q(payload: dict, field: str, kind: str):
    unit = payload.get(f"{field}_unit")
    datum = payload.get(f"{field}_datum")
    return quantity(payload.get(field), unit, datum, kind=kind)


def _load_geometry(store: Store) -> tuple[T.BeamGeometry, dict, str]:
    rows = [s for s in store.current_sources() if s["kind"] == "geometry"]
    if not rows:
        geom = T.default_geometry()
        return geom, {"azimuth_deg": list(geom.azimuth_deg),
                      "tilt_deg": list(geom.tilt_deg)}, "default"
    src = rows[0]
    p = src["payload"]
    geom = T.geometry(p["azimuth_deg"], p["tilt_deg"])
    return geom, p, src["content_id"]


def _load_scalar_source(store: Store, kind: str):
    rows = [s for s in store.current_sources() if s["kind"] == kind]
    if not rows:
        return None
    return rows[0]


def _load_ping_source(store: Store, kind: str, ping: str):
    rows = [s for s in store.current_sources()
            if s["kind"] == kind and s["payload"].get("ping") == ping]
    return rows[0] if rows else None


def _load_beam_value(store: Store, kind: str, ping: str, cell: int, beam: int):
    nk = natural_key(kind, ping, cell, beam)
    cid = store.current_source_content(kind, nk)
    if cid is None:
        return None
    src = store.get_source(cid)
    return src["payload"] if src else None


# ------------------------------------------------------------- input helpers

def _fp_of_source(src: dict) -> str:
    return fingerprint_for(src["payload"])


def _input(kind: str, src: dict, rule_id: str) -> dict:
    return {"kind": kind, "natural_key": src["natural_key"],
            "content_id": src["content_id"],
            "fingerprint": _fp_of_source(src), "rule_id": rule_id}


def _cell_index(store: Store, ping: str, cell: int) -> Optional[dict]:
    layout = _load_scalar_source(store, "cell_layout")
    if layout is None:
        return None
    cells = layout["payload"].get("cells", [])
    if cell >= len(cells):
        return None
    entry = dict(cells[cell])
    entry["layout_content_id"] = layout["content_id"]
    return entry


def _gather_cell(store: Store, ping: str, cell: int, beam_count: int,
                 radial_kind: str = "beam_velocity") -> dict:
    radials, corrs, amps, bads = [], [], [], []
    radial_srcs, corr_srcs, amp_srcs, bad_srcs = [], [], [], []
    missing = []
    for beam in range(1, beam_count + 1):
        rv = _load_beam_value(store, radial_kind, ping, cell, beam)
        co = _load_beam_value(store, "correlation", ping, cell, beam)
        am = _load_beam_value(store, "amplitude", ping, cell, beam)
        bd = _load_beam_value(store, "bad_beam", ping, cell, beam)
        for collection, src, key in ((radial_srcs, rv, "radial"),
                                     (corr_srcs, co, "correlation"),
                                     (amp_srcs, am, "amplitude"),
                                     (bad_srcs, bd, "bad")):
            if src is None:
                missing.append(f"{key}.beam{beam}")
        if rv is not None:
            q = _q(rv, "v", "speed")
            radials.append(q)
        if co is not None:
            q = quantity(co.get("correlation"), co.get("correlation_unit"), kind="counts")
            corrs.append(q.value)
        if am is not None:
            q = quantity(am.get("amplitude"), am.get("amplitude_unit"), kind="counts")
            amps.append(q.value)
        if bd is not None:
            bads.append(bool(bd.get("bad", False)))
        else:
            bads.append(False)
    return {"radial_q": radials, "correlation": corrs, "amplitude": amps,
            "bad": bads, "radial_srcs": radial_srcs, "corr_srcs": corr_srcs,
            "amp_srcs": amp_srcs, "bad_srcs": bad_srcs, "missing": missing}


def _diag(code: str, message: str, details: dict) -> dict:
    return {"code": code, "message": message, "details": details}


# -------------------------------------------------------------- derivations

def _cell_exclusion(layout_src: Optional[dict], cell: int) -> Optional[dict]:
    if layout_src is None:
        return None
    cells = layout_src["payload"].get("cells", [])
    if cell >= len(cells):
        return _diag("cell_out_of_range",
                     f"cell index {cell} is outside the declared profile",
                     {"cell": cell, "cell_count": len(cells)})
    entry = cells[cell]
    depth = entry.get("depth_m")
    if depth is None or (isinstance(depth, (int, float)) and not isinstance(depth, bool)
                         and (depth < 0 or math.isnan(float(depth)))):
        return _diag("invalid_depth_cell",
                     "cell has no valid depth; excluded from the profile",
                     {"cell": cell, "raw_depth": depth})
    return None


def compute_instrument(store: Store, ping: str, cell: int,
                       cfg: Optional[_Cfg] = None, branch_id: str = "main",
                       note: str = "instrument recompute") -> Optional[dict]:
    cfg = cfg or _Cfg()
    geom, geom_payload, geom_cid = _load_geometry(store)
    layout = _load_scalar_source(store, "cell_layout")
    exclusion = _cell_exclusion(layout, cell)
    scope = f"ping={ping}:cell={cell}:frame=instrument"
    inputs: list[dict] = []
    diagnostics: list[dict] = []
    if geom_cid != "default":
        inputs.append(_input("geometry", {"natural_key": "geometry",
                                          "content_id": geom_cid,
                                          "payload": geom_payload},
                             T.GEOMETRY_RULE_ID))
    if layout is not None:
        inputs.append(_input("cell_layout", layout, T.CELL_RULE_ID))

    if exclusion is not None:
        fingerprint = content_hash("fp", "instrument", scope,
                                   {"exclusion": exclusion,
                                    "layout": layout["content_id"] if layout else None})
        rule_set = {"beam_geometry": T.GEOMETRY_RULE_ID,
                    "beam4": T.BEAM4_RULE_ID, "beam3": T.BEAM3_RULE_ID,
                    "cell": T.CELL_RULE_ID}
        payload = {"ping": ping, "cell": cell, "frame": "instrument",
                   "unit": "m/s", "velocity": None,
                   "excluded": True, "exclusion": exclusion}
        return store.put_derivation(
            "instrument", scope, payload, "excluded", rule_set, fingerprint,
            inputs, [], [exclusion], branch_id=branch_id, note=note)

    data = _gather_cell(store, ping, cell, geom.beam_count)
    for srcs, kind in ((data["radial_srcs"], "beam_velocity"),
                       (data["corr_srcs"], "correlation"),
                       (data["amp_srcs"], "amplitude"),
                       (data["bad_srcs"], "bad_beam")):
        for src in srcs:
            inputs.append(_input(kind, src, T.BEAM4_RULE_ID))
    radial = np.array([q.value if q is not None else np.nan
                       for q in data["radial_q"]], dtype=float)
    for q in data["radial_q"]:
        if q is not None and q.diagnostic is not None:
            diagnostics.append(_diag(q.diagnostic.code, q.diagnostic.message,
                                     {"raw_repr": q.diagnostic.raw_repr,
                                      "unit": q.diagnostic.unit}))
    quality = T.beam_quality(data["correlation"], data["amplitude"],
                             data["bad"], cfg.corr_floor)
    if data["missing"]:
        diagnostics.append(_diag("missing_beam_record",
                                 "beam-level source records are missing",
                                 {"missing": data["missing"]}))

    candidates: list[dict] = []
    status = "indeterminate"
    chosen = None
    usable_count = sum(quality["usable"])
    if usable_count == geom.beam_count and not np.isnan(radial).any():
        sol = T.solve_four_beam(geom, radial, data["correlation"],
                                data["amplitude"], data["bad"], cfg.corr_floor)
        cfp = content_hash("cfp", "candidate", "four",
                           {"velocity": sol.velocity,
                            "error_velocity": sol.error_velocity,
                            "beams": sol.beams})
        candidates.append({"candidate_id": "four_beam",
                           "rule_id": T.BEAM4_RULE_ID, "fingerprint": cfp,
                           "payload": {"velocity": list(sol.velocity),
                                       "error_velocity": sol.error_velocity,
                                       "beams": list(sol.beams),
                                       "label": "four-beam solution"}})
        chosen = sol
        status = "four_beam"
    subsets = T.legal_three_beam_subsets(quality, cfg.corr_good)
    for subset in subsets:
        idx = np.array([b - 1 for b in subset])
        if np.isnan(_beam_values(radial, idx)).any():
            continue
        sol = T.solve_three_beam(geom, radial, subset)
        cid_name = "three_" + "_".join(str(b) for b in subset)
        cfp = content_hash("cfp", "candidate", cid_name,
                           {"velocity": sol.velocity, "beams": sol.beams})
        candidates.append({"candidate_id": cid_name,
                           "rule_id": T.BEAM3_RULE_ID, "fingerprint": cfp,
                           "payload": {"velocity": list(sol.velocity),
                                       "error_velocity": None,
                                       "beams": list(sol.beams),
                                       "label": f"three-beam {list(subset)}"}})
        if status == "indeterminate":
            chosen = sol
    if len(candidates) > 1 or (status == "indeterminate" and len(candidates) > 1):
        status = "ambiguous"
    elif status == "indeterminate" and len(candidates) == 1:
        status = "three_beam"
    elif not candidates:
        status = "indeterminate"
        diagnostics.append(_diag(
            "no_legal_solution",
            "no four-beam solution and no legal three-beam subset",
            {"quality": quality}))

    velocity = list(chosen.velocity) if chosen is not None else None
    error_velocity = chosen.error_velocity if chosen is not None else None
    rule_set = {"beam_geometry": T.GEOMETRY_RULE_ID + f"@{geom_cid[:12]}",
                "beam4": T.BEAM4_RULE_ID, "beam3": T.BEAM3_RULE_ID,
                "cell": T.CELL_RULE_ID,
                "corr_floor": cfg.corr_floor, "corr_good": cfg.corr_good}
    fingerprint = content_hash(
        "fp", "instrument", scope,
        {"inputs": sorted((i["kind"], i["natural_key"], i["fingerprint"])
                          for i in inputs),
         "candidates": sorted((c["candidate_id"], c["fingerprint"])
                              for c in candidates),
         "status": status})
    payload = {"ping": ping, "cell": cell, "frame": "instrument",
               "unit": "m/s", "velocity": velocity,
               "error_velocity": error_velocity, "status": status,
               "quality": quality, "candidate_count": len(candidates)}
    result = store.put_derivation(
        "instrument", scope, payload, status, rule_set, fingerprint,
        inputs, candidates, diagnostics, branch_id=branch_id, note=note)
    result["candidates"] = candidates
    return result


def _beam_values(radial: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return radial[idx]


def compute_ship(store: Store, ping: str, cell: int,
                 cfg: Optional[_Cfg] = None, branch_id: str = "main",
                 note: str = "ship recompute") -> Optional[dict]:
    cfg = cfg or _Cfg()
    inst_scope = f"ping={ping}:cell={cell}:frame=instrument"
    inst = store.derivation_by_scope("instrument", inst_scope, branch_id)
    mounting = _load_scalar_source(store, "mounting")
    scope = f"ping={ping}:cell={cell}:frame=ship"
    if inst is None or mounting is None:
        return None
    alignment_q = quantity(mounting["payload"].get("alignment_deg"),
                           mounting["payload"].get("alignment_unit"),
                           kind="angle")
    diagnostics = []
    if alignment_q.value is None:
        diagnostics.append(_diag("mounting_alignment_unconvertible",
                                 "mounting alignment cannot be normalised",
                                 {"payload": mounting["payload"]}))
        alignment = 0.0
    else:
        alignment = alignment_q.value
    candidates = []
    for cand in inst["candidates"]:
        vel = cand["payload"]["velocity"]
        ship_vel = T.instrument_to_ship(vel, alignment)
        candidates.append({
            "candidate_id": cand["candidate_id"],
            "rule_id": T.INST2SHIP_RULE_ID,
            "fingerprint": content_hash(
                "cfp", "ship", cand["candidate_id"],
                {"from": cand["fingerprint"], "alignment": alignment,
                 "rule": T.INST2SHIP_RULE_ID}),
            "payload": {"velocity": [float(v) for v in ship_vel],
                        "error_velocity": cand["payload"].get("error_velocity"),
                        "beams": cand["payload"]["beams"],
                        "source_candidate": cand["candidate_id"],
                        "label": cand["payload"].get("label", cand["candidate_id"])}})
    rule_set = {"inst2ship": T.INST2SHIP_RULE_ID,
                "alignment_deg": alignment,
                "alignment_version": mounting["content_id"],
                "upstream": inst["content_id"]}
    fingerprint = content_hash(
        "fp", "ship", scope,
        {"instrument": inst["fingerprint"], "mounting": _fp_of_source(mounting),
         "alignment": alignment, "rule": T.INST2SHIP_RULE_ID,
         "candidates": sorted((c["candidate_id"], c["fingerprint"]) for c in candidates)})
    inputs = [{"kind": "instrument", "natural_key": inst_scope,
               "content_id": inst["content_id"],
               "fingerprint": inst["fingerprint"],
               "rule_id": T.INST2SHIP_RULE_ID},
              _input("mounting", mounting, T.INST2SHIP_RULE_ID)]
    payload = {"ping": ping, "cell": cell, "frame": "ship", "unit": "m/s",
               "status": inst["status"], "candidate_count": len(candidates)}
    return store.put_derivation(
        "ship", scope, payload, inst["status"], rule_set, fingerprint,
        inputs, candidates, inst["diagnostics"] + diagnostics,
        branch_id=branch_id, note=note)


def _attitude_samples(store: Store, ping: str):
    rows = [s for s in store.current_sources()
            if s["kind"] == "attitude" and s["payload"].get("ping") == ping]
    samples = []
    srcs = []
    for src in sorted(rows, key=lambda r: r["payload"].get("time_s", 0)):
        p = src["payload"]
        h = quantity(p.get("heading_mag_deg"), p.get("heading_unit"), kind="angle")
        pt = quantity(p.get("pitch_deg"), p.get("pitch_unit"), kind="angle")
        rl = quantity(p.get("roll_deg"), p.get("roll_unit"), kind="angle")
        if None not in (h.value, pt.value, rl.value):
            samples.append(T.AttitudeSample(float(p["time_s"]), h.value,
                                            pt.value, rl.value))
            srcs.append(src)
    return samples, srcs


def compute_earth(store: Store, ping: str, cell: int,
                  cfg: Optional[_Cfg] = None, branch_id: str = "main",
                  note: str = "earth recompute") -> Optional[dict]:
    cfg = cfg or _Cfg()
    ship_scope = f"ping={ping}:cell={cell}:frame=ship"
    ship = store.derivation_by_scope("ship", ship_scope, branch_id)
    decl = _load_scalar_source(store, "declination")
    clock = _load_ping_source(store, "ping_clock", ping)
    scope = f"ping={ping}:cell={cell}:frame=earth"
    if ship is None or decl is None or clock is None:
        return None
    decl_q = quantity(decl["payload"].get("declination_deg"),
                      decl["payload"].get("declination_unit"), kind="angle")
    time_s = float(clock["payload"]["time_s"])
    samples, attitude_srcs = _attitude_samples(store, ping)
    diagnostics: list[dict] = []
    attitude = T.attitude_at(samples, time_s,
                             decl_q.value or 0.0, cfg.max_attitude_gap_s)
    inputs = [{"kind": "ship", "natural_key": ship_scope,
               "content_id": ship["content_id"],
               "fingerprint": ship["fingerprint"], "rule_id": cfg.earth_order},
              _input("declination", decl, T.DECLINATION_RULE_ID),
              _input("ping_clock", clock, T.ATTITUDE_RULE_ID)]
    for src in attitude_srcs:
        inputs.append(_input("attitude", src, T.ATTITUDE_RULE_ID))
    rule_set = {"ship2earth": cfg.earth_order,
                "attitude": T.ATTITUDE_RULE_ID,
                "declination": T.DECLINATION_RULE_ID,
                "declination_version": decl["content_id"]}
    if decl_q.value is None:
        diagnostics.append(_diag("declination_unconvertible",
                                 "declination cannot be normalised",
                                 {"payload": decl["payload"]}))
    if attitude.diagnostic is not None:
        diagnostics.append(_diag(attitude.diagnostic["code"],
                                 attitude.diagnostic["message"],
                                 attitude.diagnostic))
        fingerprint = content_hash(
            "fp", "earth", scope,
            {"ship": ship["fingerprint"], "attitude": "blocked",
             "diag": attitude.diagnostic, "order": cfg.earth_order,
             "declination": _fp_of_source(decl)})
        payload = {"ping": ping, "cell": cell, "frame": "earth",
                   "unit": "m/s", "status": "blocked",
                   "heading_true_deg": None, "candidate_count": 0}
        return store.put_derivation(
            "earth", scope, payload, "blocked", rule_set, fingerprint,
            inputs, [], diagnostics, branch_id=branch_id, note=note)

    candidates = []
    for cand in ship["candidates"]:
        earth_vel = T.ship_to_earth(cand["payload"]["velocity"],
                                    attitude.heading_true_deg,
                                    attitude.pitch_deg, attitude.roll_deg,
                                    cfg.earth_order)
        candidates.append({
            "candidate_id": cand["candidate_id"],
            "rule_id": cfg.earth_order,
            "fingerprint": content_hash(
                "cfp", "earth", cand["candidate_id"],
                {"from": cand["fingerprint"],
                 "heading": attitude.heading_true_deg,
                 "pitch": attitude.pitch_deg, "roll": attitude.roll_deg,
                 "order": cfg.earth_order,
                 "decl_version": decl["content_id"]}),
            "payload": {"velocity": [float(v) for v in earth_vel],
                        "error_velocity": cand["payload"].get("error_velocity"),
                        "beams": cand["payload"]["beams"],
                        "source_candidate": cand["candidate_id"],
                        "heading_true_deg": attitude.heading_true_deg,
                        "pitch_deg": attitude.pitch_deg,
                        "roll_deg": attitude.roll_deg,
                        "attitude_segment": attitude.segment,
                        "label": cand["payload"].get("label", cand["candidate_id"])}})
    fingerprint = content_hash(
        "fp", "earth", scope,
        {"ship": ship["fingerprint"],
         "declination": _fp_of_source(decl),
         "order": cfg.earth_order,
         "attitude_inputs": sorted((s["content_id"] for s in attitude_srcs)),
         "time_s": time_s,
         "candidates": sorted((c["candidate_id"], c["fingerprint"]) for c in candidates)})
    payload = {"ping": ping, "cell": cell, "frame": "earth", "unit": "m/s",
               "status": ship["status"],
               "heading_true_deg": attitude.heading_true_deg,
               "pitch_deg": attitude.pitch_deg, "roll_deg": attitude.roll_deg,
               "attitude_segment": attitude.segment,
               "candidate_count": len(candidates)}
    return store.put_derivation(
        "earth", scope, payload, ship["status"], rule_set, fingerprint,
        inputs, candidates, diagnostics, branch_id=branch_id, note=note)


# ------------------------------------------------------------- bottom track

def compute_bottom_track(store: Store, ping: str,
                         cfg: Optional[_Cfg] = None, branch_id: str = "main",
                         note: str = "bottom-track recompute") -> Optional[dict]:
    cfg = cfg or _Cfg()
    geom, geom_payload, geom_cid = _load_geometry(store)
    bt = _load_ping_source(store, "bottom_track", ping)
    decl = _load_scalar_source(store, "declination")
    clock = _load_ping_source(store, "ping_clock", ping)
    if bt is None or decl is None or clock is None:
        return None
    scope = f"ping={ping}:frame=earth:bottom_track"
    radial_q = [quantity(bt["payload"].get(f"b{i}"),
                         bt["payload"].get("b_unit", "m/s"), kind="speed")
                for i in range(1, geom.beam_count + 1)]
    bad = [bool(bt["payload"].get(f"b{i}_bad", False))
           for i in range(1, geom.beam_count + 1)]
    corr = bt["payload"].get("correlation", [None] * geom.beam_count)
    amp = bt["payload"].get("amplitude", [None] * geom.beam_count)
    quality = T.beam_quality(corr, amp, bad, cfg.corr_floor)
    radial = np.array([q.value if q.value is not None else np.nan
                       for q in radial_q], dtype=float)
    diagnostics = [_diag(q.diagnostic.code, q.diagnostic.message,
                         {"raw_repr": q.diagnostic.raw_repr})
                   for q in radial_q if q.diagnostic is not None]
    candidates = []
    if sum(quality["usable"]) == geom.beam_count:
        sol = T.solve_four_beam(geom, radial, corr, amp, bad, cfg.corr_floor)
        inst_vel = np.array(sol.velocity)
        mounting = _load_scalar_source(store, "mounting")
        alignment = 0.0
        if mounting is not None:
            aq = quantity(mounting["payload"].get("alignment_deg"),
                          mounting["payload"].get("alignment_unit"), kind="angle")
            alignment = aq.value or 0.0
        ship_vel = T.instrument_to_ship(inst_vel, alignment)
        samples, attitude_srcs = _attitude_samples(store, ping)
        decl_q = quantity(decl["payload"].get("declination_deg"),
                          decl["payload"].get("declination_unit"), kind="angle")
        attitude = T.attitude_at(samples, float(clock["payload"]["time_s"]),
                                 decl_q.value or 0.0, cfg.max_attitude_gap_s)
        if attitude.diagnostic is None:
            earth_vel = T.ship_to_earth(ship_vel, attitude.heading_true_deg,
                                        attitude.pitch_deg, attitude.roll_deg,
                                        cfg.earth_order)
            fp = content_hash("cfp", "bt", ping,
                              {"radial": list(radial), "order": cfg.earth_order,
                               "decl": decl["content_id"]})
            candidates.append({"candidate_id": "four_beam",
                               "rule_id": T.BEAM4_RULE_ID, "fingerprint": fp,
                               "payload": {"instrument_velocity": list(inst_vel),
                                           "ship_velocity": [float(v) for v in ship_vel],
                                           "velocity": [float(v) for v in earth_vel],
                                           "error_velocity": sol.error_velocity,
                                           "beams": list(sol.beams),
                                           "heading_true_deg": attitude.heading_true_deg,
                                           "label": "bottom-track four-beam"}})
            status = "four_beam"
        else:
            status = "blocked"
            diagnostics.append(_diag(attitude.diagnostic["code"],
                                     attitude.diagnostic["message"],
                                     attitude.diagnostic))
    else:
        status = "indeterminate"
        diagnostics.append(_diag("bt_no_four_beam",
                                 "bottom track needs four good beams", {}))
    inputs = [_input("bottom_track", bt, T.BEAM4_RULE_ID),
              _input("declination", decl, T.DECLINATION_RULE_ID),
              _input("ping_clock", clock, T.ATTITUDE_RULE_ID)]
    mounting = _load_scalar_source(store, "mounting")
    if mounting is not None:
        inputs.append(_input("mounting", mounting, T.INST2SHIP_RULE_ID))
    rule_set = {"beam4": T.BEAM4_RULE_ID, "beam_geometry": T.GEOMETRY_RULE_ID,
                "inst2ship": T.INST2SHIP_RULE_ID, "ship2earth": cfg.earth_order,
                "declination": T.DECLINATION_RULE_ID}
    fingerprint = content_hash("fp", "bottom_track", scope,
                               {"inputs": sorted((i["kind"], i["natural_key"],
                                                  i["fingerprint"]) for i in inputs),
                                "order": cfg.earth_order})
    payload = {"ping": ping, "frame": "earth", "component": "bottom_track",
               "unit": "m/s", "status": status,
               "candidate_count": len(candidates), "quality": quality}
    return store.put_derivation(
        "bottom_track", scope, payload, status, rule_set, fingerprint,
        inputs, candidates, diagnostics, branch_id=branch_id, note=note)


# ------------------------------------------------------------- graph update

FRAME_CHAIN = ("instrument", "ship", "earth")


def _pings(store: Store) -> list[str]:
    return sorted({s["payload"]["ping"] for s in store.current_sources()
                   if "ping" in s["payload"]})


def _cells(store: Store) -> int:
    layout = _load_scalar_source(store, "cell_layout")
    return len(layout["payload"]["cells"]) if layout else 0


def recompute_all(store: Store, cfg: Optional[_Cfg] = None,
                  branch_id: str = "main", note: str = "full recompute") -> list[dict]:
    cfg = cfg or _Cfg()
    results = []
    for ping in _pings(store):
        results.append(compute_bottom_track(store, ping, cfg, branch_id, note))
        for cell in range(_cells(store)):
            results.append(compute_instrument(store, ping, cell, cfg, branch_id, note))
            results.append(compute_ship(store, ping, cell, cfg, branch_id, note))
            results.append(compute_earth(store, ping, cell, cfg, branch_id, note))
    return [r for r in results if r is not None]


def affected_scopes(store: Store, changed_content_ids: list[str]) -> set[tuple[str, str]]:
    """Return (kind, scope) derivations whose input fingerprint changed."""
    affected: set[tuple[str, str]] = set()
    frontier = list(changed_content_ids)
    seen = set()
    while frontier:
        cid = frontier.pop()
        if cid in seen:
            continue
        seen.add(cid)
        for dep_cid in store.dependents(cid):
            deriv = store.get_derivation(dep_cid)
            if deriv is None:
                continue
            changed = any(inp["content_id"] == cid for inp in deriv["inputs"])
            if changed:
                affected.add((deriv["kind"], deriv["scope_key"]))
                frontier.append(dep_cid)
    return affected


def explain_validity(store: Store, content_id: str) -> dict:
    deriv = store.get_derivation(content_id)
    if deriv is None:
        return {"content_id": content_id, "state": "missing"}
    edges = []
    stale = False
    for inp in deriv["inputs"]:
        current = None
        if inp["kind"] in SOURCE_KINDS:
            current = store.current_source_content(inp["kind"], inp["natural_key"])
        else:
            current = store.pointer(inp["kind"], inp["natural_key"])
        edge_state = "current" if current == inp["content_id"] else "changed"
        if edge_state == "changed":
            stale = True
        edges.append({"input_kind": inp["kind"],
                      "natural_key": inp["natural_key"], "rule_id": inp["rule_id"],
                      "recorded_content_id": inp["content_id"],
                      "current_content_id": current, "state": edge_state})
    return {"content_id": content_id, "kind": deriv["kind"],
            "scope_key": deriv["scope_key"], "status": deriv["status"],
            "state": "needs_recompute" if stale else "still_valid",
            "edges": edges}


def recompute_propagate(store: Store, changed_content_ids: list[str],
                        cfg: Optional[_Cfg] = None,
                        branch_id: str = "main",
                        note: str = "propagated recompute") -> dict:
    """Recompute only scopes downstream of the changed content versions."""
    cfg = cfg or _Cfg()
    affected = affected_scopes(store, changed_content_ids)
    recomputed = []
    skipped_still_valid = []
    for kind in ("bottom_track",):
        for k, scope in sorted(affected):
            if k != kind:
                continue
            ping = scope.split("ping=")[1].split(":")[0]
            out = compute_bottom_track(store, ping, cfg, branch_id, note)
            if out:
                recomputed.append(out["content_id"])
    for cell_frame in FRAME_CHAIN:
        for kind, scope in sorted(affected):
            if kind != cell_frame:
                continue
            ping = scope.split("ping=")[1].split(":")[0]
            cell = int(scope.split("cell=")[1].split(":")[0])
            if kind == "instrument":
                out = compute_instrument(store, ping, cell, cfg, branch_id, note)
            elif kind == "ship":
                out = compute_ship(store, ping, cell, cfg, branch_id, note)
            else:
                out = compute_earth(store, ping, cell, cfg, branch_id, note)
            if out:
                recomputed.append(out["content_id"])
    all_scopes = {(d["kind"], d["scope_key"]) for d in store.all_derivations()}
    for kind, scope in sorted(all_scopes - affected):
        skipped_still_valid.append(f"{kind}:{scope}")
    return {"changed_inputs": changed_content_ids,
            "affected_scopes": sorted(f"{k}:{s}" for k, s in affected),
            "recomputed": recomputed,
            "still_valid": skipped_still_valid}
