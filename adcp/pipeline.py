"""Dependency-graph derivation engine.

Stages per depth cell per ping:

  beam measurements + quality evidence
      -> four-beam solution  OR  set of legal three-beam candidates  OR excluded
      -> instrument velocity
      -> ship velocity (mounting angles + explicit order)
      -> earth velocity (true heading/pitch/roll + explicit order + declination)

All context that could change the answer enters the derivation fingerprint:
rule version, beam geometry, mounting revision/order, declination model
version, transform order, and every input record id.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .angles import (
    ANGLES_SCHEMA_VERSION,
    instrument_to_ship_matrix,
    ship_to_earth_matrix,
    true_heading,
)
from .dataset import DatasetModel
from .fingerprint import derive_fingerprint, digest
from .geometry import (
    canonical_radials,
    solve_four_beam,
    three_beam_candidates,
)
from .quality import (
    QUALITY_RULE_VERSION,
    assess_ping_beams,
    cell_exclusion,
    solution_decision,
)

PIPELINE_RULE_VERSION = "pipeline/2"


def _vec(arr, frame, datum=None, unit="m/s"):
    return {"x": float(arr[0]), "y": float(arr[1]), "z": float(arr[2]),
            "frame": frame, "datum": datum, "unit": unit}


def cell_node(dataset_id, ping, cell):
    return digest(
        {"node": "cell_velocity", "dataset": dataset_id,
         "ping": ping, "cell": cell},
        prefix="node",
    )


def attitude_node(dataset_id, ping):
    return digest(
        {"node": "attitude_resolved", "dataset": dataset_id, "ping": ping},
        prefix="node",
    )


def profile_node(dataset_id):
    return digest({"node": "profile", "dataset": dataset_id}, prefix="node")


def _input_record_ids(model: DatasetModel, ping, cell) -> Dict[str, Any]:
    ids = {"beam": [], "correlation": [], "amplitude": [], "badflag": []}
    for b in model.beams:
        if b.ping == ping and b.cell == cell:
            ids["beam"].append({"beam": b.beam, "raw": b.raw})
    for s in model.correlations:
        if s.ping == ping and s.cell == cell:
            ids["correlation"].append({"beam": s.beam, "raw": s.raw})
    for s in model.amplitudes:
        if s.ping == ping and s.cell == cell:
            ids["amplitude"].append({"beam": s.beam, "raw": s.raw})
    for s in model.bad_flags:
        if s.ping == ping and s.cell == cell:
            ids["badflag"].append({"beam": s.beam, "raw": s.raw})
    return ids


def resolve_attitude(model: DatasetModel, ping: int) -> Dict[str, Any]:
    """Attitude at ping time: exact sample, unwrapped interpolation, or gap."""
    t = ping * model.time_step_s
    result = model.attitude_at(t)
    payload: Dict[str, Any] = {
        "ping": ping,
        "t_s": t,
        "status": result.status,
    }
    if result.attitude is not None:
        decl = model.declination
        payload["heading_deg"] = result.attitude.heading_deg
        payload["heading_datum"] = result.attitude.heading_datum
        payload["pitch_deg"] = result.attitude.pitch_deg
        payload["roll_deg"] = result.attitude.roll_deg
        payload["interpolated"] = result.interpolated
        payload["bracket"] = list(result.bracket) if result.bracket else None
        payload["fraction"] = result.fraction
        if result.attitude.heading_datum == "magnetic":
            if decl is None:
                payload["status"] = "gap"
                payload["reason"] = "magnetic_heading_without_declination"
                payload["true_heading_deg"] = None
            else:
                payload["true_heading_deg"] = true_heading(result.attitude, decl)
        else:
            payload["true_heading_deg"] = result.attitude.heading_deg
    else:
        payload["reason"] = "attitude_gap" if result.status == "gap" else result.status
    payload["declination"] = model.declination.to_fingerprint() if model.declination else None
    fp = derive_fingerprint(
        "attitude_resolved",
        {"t_s": t, "result": {k: v for k, v in payload.items()
                              if k not in ("declination",)}},
        ANGLES_SCHEMA_VERSION,
        declination_fp=model.declination.to_fingerprint() if model.declination else None,
    )
    payload["fingerprint"] = fp
    return payload


def derive_cell(model: DatasetModel, ping: int, cell: int) -> Dict[str, Any]:
    """Run every stage for one ping/cell and return the full evidence payload."""
    n_beams = model.geometry.n_beams
    corrs = [model.correlation(ping, cell, i) for i in range(n_beams)]
    amps = [model.amplitude(ping, cell, i) for i in range(n_beams)]
    flags = [model.bad_flag(ping, cell, i) for i in range(n_beams)]
    quality = assess_ping_beams(corrs, amps, flags, model.thresholds)
    decision = solution_decision(quality)

    payload: Dict[str, Any] = {
        "dataset_id": model.dataset_id,
        "ping": ping,
        "cell": cell,
        "cell_depth_m": model.cell_depths_m[cell] if cell < len(model.cell_depths_m) else None,
        "beam_quality": [q.to_dict() for q in quality],
        "decision": decision,
        "radials_mps": [model.beam_value(ping, cell, i) for i in range(n_beams)],
        "raw_inputs": _input_record_ids(model, ping, cell),
        "geometry": model.geometry.to_fingerprint(),
        "mounting": model.mounting.to_fingerprint(),
        "declination": model.declination.to_fingerprint() if model.declination else None,
        "transform_order": model.transform_order,
        "stages": [],
        "candidates": [],
        "excluded": None,
    }

    # Stage 0: cell depth validity.
    depth = payload["cell_depth_m"]
    bottom = model.bottom_depth(ping)
    exclusion = cell_exclusion(
        depth, bottom, sidelobe_fraction=model.sidelobe_fraction
    )
    if exclusion is not None:
        payload["excluded"] = exclusion
        payload["status"] = "excluded"
    if any(v is None for v in payload["radials_mps"]):
        missing = [i for i, v in enumerate(payload["radials_mps"]) if v is None]
        if payload["excluded"] is None:
            payload["excluded"] = {
                "code": "beam_velocity_unconvertible",
                "message": "one or more beam velocities could not be normalised",
                "beams": missing,
            }
            payload["status"] = "excluded"

    attitude = resolve_attitude(model, ping)
    payload["attitude"] = attitude

    if payload["excluded"] is not None:
        payload["status"] = "excluded"
        payload["fingerprint"] = _cell_fp(model, payload)
        return payload

    radials = canonical_radials(payload["radials_mps"], model.geometry)

    # Stage 1: beam -> instrument solutions.
    primary: Optional[Dict[str, Any]] = None
    candidates: List[Dict[str, Any]] = []
    if decision["kind"] == "four_beam":
        sol = solve_four_beam(radials, model.geometry)
        sol["candidate_id"] = digest(
            {"node": cell_node(model.dataset_id, ping, cell), "sol": sol,
             "kind": "four"}, prefix="cand")
        primary = sol
        candidates.append(sol)
    elif decision["kind"] == "unusable":
        payload["excluded"] = {
            "code": "insufficient_good_beams",
            "message": "two or more hard-bad beams; no velocity solution",
            "hard_bad_beams": decision["hard_bad_beams"],
        }
        payload["status"] = "excluded"
    else:
        # three_beam (forced) and three_beam_ambiguous (all legal candidates).
        cands = three_beam_candidates(
            radials, model.geometry,
            decision["hard_bad_beams"], ambiguous=decision["ambiguous"],
        )
        for c in cands:
            c["candidate_id"] = digest(
                {"node": cell_node(model.dataset_id, ping, cell), "sol": c},
                prefix="cand",
            )
            candidates.append(c)
        if len(candidates) == 1 and not decision["ambiguous"]:
            primary = candidates[0]
        else:
            # Multiplicity preserved: no single value auto-chosen.
            primary = None

    payload["candidates"] = candidates

    # Stages 2 and 3 require a resolved attitude and a unique primary.
    if primary is not None and attitude["status"] in ("sample", "interpolated") \
            and attitude.get("true_heading_deg") is not None:
        u_inst = np.array(primary["u_instrument"])
        payload["stages"].append({
            "stage": "beam_to_instrument",
            "kind": primary["solution_kind"],
            "vector": _vec(u_inst, "instrument"),
            "detail": {k: v for k, v in primary.items()
                       if k not in ("u_instrument",)},
        })
        m_is = instrument_to_ship_matrix(model.mounting)
        u_ship = m_is @ u_inst
        payload["stages"].append({
            "stage": "instrument_to_ship",
            "vector": _vec(u_ship, "ship"),
            "matrix": m_is.tolist(),
        })
        m_se = ship_to_earth_matrix(
            attitude["true_heading_deg"], attitude["pitch_deg"], attitude["roll_deg"],
            order=model.transform_order, earth_datum=model.earth_datum,
        )
        u_earth = m_se @ u_ship
        payload["stages"].append({
            "stage": "ship_to_earth",
            "vector": _vec(u_earth, "earth", datum=model.earth_datum),
            "matrix": m_se.tolist(),
            "true_heading_deg": attitude["true_heading_deg"],
        })
        payload["u_instrument"] = _vec(u_inst, "instrument")
        payload["u_ship"] = _vec(u_ship, "ship")
        payload["u_earth"] = _vec(u_earth, "earth", datum=model.earth_datum)
        payload["status"] = "resolved"
    elif primary is None and payload["excluded"] is None:
        payload["status"] = "ambiguous"
        payload["ambiguous_reason"] = decision["reason"]
    else:
        payload.setdefault("status", "excluded")
        if payload["excluded"] is None and attitude["status"] not in (
                "sample", "interpolated"):
            payload["excluded"] = {
                "code": "attitude_unavailable",
                "message": "attitude interpolation would cross a gap",
                "attitude_status": attitude["status"],
            }

    payload["fingerprint"] = _cell_fp(model, payload)
    return payload


def _cell_fp(model: DatasetModel, payload: Dict[str, Any]) -> str:
    fp_inputs = {
        "radials_mps": payload["radials_mps"],
        "beam_quality": payload["beam_quality"],
        "cell_depth_m": payload["cell_depth_m"],
        "raw_inputs": payload["raw_inputs"],
        "attitude": {
            k: v for k, v in payload["attitude"].items() if k != "fingerprint"
        },
        "excluded": payload["excluded"],
        "thresholds": model.thresholds.to_fingerprint(),
    }
    return derive_fingerprint(
        "cell_velocity", fp_inputs, PIPELINE_RULE_VERSION,
        geometry_fp=model.geometry.to_fingerprint(),
        mounting_fp=model.mounting.to_fingerprint(),
        declination_fp=model.declination.to_fingerprint() if model.declination else None,
        transform_order=model.transform_order,
    )


def derive_all(model: DatasetModel) -> Dict[str, Any]:
    cells: List[Dict[str, Any]] = []
    attitudes: List[Dict[str, Any]] = []
    for ping in model.pings:
        attitudes.append(resolve_attitude(model, ping))
        for cell in model.cells:
            cells.append(derive_cell(model, ping, cell))
    profile = {
        "dataset_id": model.dataset_id,
        "n_pings": len(model.pings),
        "n_cells": len(model.cells),
        "earth_datum": model.earth_datum,
        "statuses": {},
    }
    for c in cells:
        profile["statuses"][c["status"]] = profile["statuses"].get(c["status"], 0) + 1
    profile["fingerprint"] = digest(
        {"profile": profile,
         "cells": [c["fingerprint"] for c in cells],
         "attitudes": [a["fingerprint"] for a in attitudes]},
        prefix="node",
    )
    return {"cells": cells, "attitudes": attitudes, "profile": profile}
