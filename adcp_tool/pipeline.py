"""Pure derivation pipeline: raw ping/cell evidence -> velocity candidates.

The pipeline never discards ambiguity:

* a 4-beam result and 3-beam results are separate candidates with separate
  fingerprints; they are never merged into one value;
* every legal 3-beam candidate is retained;
* an attitude gap leaves the instrument solution valid but ship/earth
  undetermined with a diagnostic, instead of being interpolated across gap;
* transform order, beam geometry, mount version and declination version all
  enter the derivation fingerprint.
"""
from __future__ import annotations

from typing import Optional

from .geometry import BeamGeometry, fingerprint
from .quality import (QualityPolicy, classify_beam, valid_for_4beam,
                      valid_for_3beam, cell_status)
from .rotation import MountAngles, TransformConfig, instrument_to_ship, ship_to_earth
from .units import convert
from .attitude import AttitudeSeries


def _norm_velocity(q):
    if q is None:
        return None, "velocity-missing"
    val, diag = convert(q, "m/s")
    return val, diag


def _candidate_fingerprint(cell_identity: str, cand: dict, provenance: dict) -> str:
    payload = {
        "cell": cell_identity,
        "candidate": cand["candidate_id"],
        "mode": cand["mode"],
        "beams": cand["beams"],
        "instrument": [round(cand["u"], 9), round(cand["v"], 9),
                       round(cand["w"], 9)],
        "provenance": provenance,
    }
    return fingerprint("deriv", payload)


def derive_cell(ping_id: str, ping_t: float, ping_idx: int, cell_idx: int,
                cell: dict, geom: BeamGeometry, mount: MountAngles,
                declination: dict, qpolicy: QualityPolicy,
                cfg: TransformConfig, attitude: AttitudeSeries,
                source_versions: dict,
                invalid_cells: Optional[set] = None) -> dict:
    """Derive all legal velocity candidates for one depth cell of one ping."""
    depth = cell.get("depth")  # Q or None
    status, cell_reasons = cell_status(
        ping_idx, cell_idx,
        None if depth is None or depth.missing() else depth.to("m").value,
        set() if invalid_cells is None else set(map(tuple, invalid_cells)),
        cell.get("min_depth_m", 0.5), cell.get("blank_m", 0.0))

    raw_beams = cell["beams"]  # list of Q (velocity) with original unit/raw
    corrs = cell.get("correlation", [None] * geom.nbeams)
    amps = cell.get("amplitude", [None] * geom.nbeams)
    bad_flags = cell.get("bad_beam", [False] * geom.nbeams)

    verdicts, bvals, bdiag = [], [], []
    for k in range(geom.nbeams):
        val, diag = _norm_velocity(raw_beams[k])
        bvals.append(val)
        bdiag.append(diag)
        verdicts.append(classify_beam(
            k, val, corrs[k], amps[k], bool(bad_flags[k]), qpolicy))

    att = attitude.at(ping_t)
    provenance = {
        "beam_velocity": source_versions.get("beam"),
        "correlation": source_versions.get("correlation"),
        "amplitude": source_versions.get("amplitude"),
        "bad_beam": source_versions.get("bad_beam"),
        "attitude": source_versions.get("attitude"),
        "attitude_status": att.status,
        "attitude_bracket": list(att.bracket) if att.bracket else None,
        "declination": declination["id"],
        "mount": mount.version,
        "geometry": geom.fingerprint_inputs(),
        "transform_order": cfg.id(),
        "quality_policy": qpolicy.id,
    }

    identity = f"{ping_id}:c{cell_idx+1}"
    candidates = []
    good4 = valid_for_4beam(verdicts)
    usable3 = valid_for_3beam(verdicts)

    def expand(sol):
        cand = dict(sol)
        cand.setdefault("beams", list(range(geom.nbeams)))
        cand.setdefault("missing_beams", [])
        cand.setdefault("candidate_id", "b1234")
        stage = {"instrument": {"u": cand["u"], "v": cand["v"], "w": cand["w"],
                                "unit": "m/s", "basis": "instrument",
                                "error_velocity": cand.get("error_velocity")}}
        if att.status in ("gap", "missing"):
            stage.update({
                "ship": None, "earth": None,
                "downstream_status": att.status,
                "diagnostic": f"attitude-{att.status}",
            })
        else:
            ship = instrument_to_ship([cand["u"], cand["v"], cand["w"]],
                                      att.roll_deg, att.pitch_deg, mount, cfg)
            earth = ship_to_earth(ship["v_ship"], att.heading_deg, mount,
                                  declination["declination_deg"], cfg)
            stage.update({"ship": {"vector": ship["v_ship"], "unit": "m/s",
                                   "basis": "ship",
                                   "R_tilt": ship["R_tilt"],
                                   "R_mount": ship["R_mount"]},
                          "earth": {"vector": earth["v_earth"], "unit": "m/s",
                                    "basis": "earth",
                                    "basis_detail": "earth-true-north",
                                    "heading_used_deg": earth["heading_used_deg"],
                                    "declination_deg": earth["declination_deg"],
                                    "R_head": earth["R_head"],
                                    "R_decl": earth["R_decl"]},
                          "downstream_status": "ok"})
        cand["stages"] = stage
        cand["fingerprint"] = _candidate_fingerprint(identity, cand, provenance)
        return cand

    n_bad = sum(v.tier == "bad" for v in verdicts)
    if all(good4):
        candidates.append(expand(geom.solve4(bvals)))
    elif n_bad == 0:
        # only marginal beams: keep the 4-beam least-squares result AND every
        # legal exact 3-beam solution -> genuinely multi-valued, never merged
        candidates.append(expand(geom.solve4(bvals)))
        for sol in geom.solve3_candidates(bvals, usable3):
            candidates.append(expand(sol))
    else:
        for sol in geom.solve3_candidates(bvals, usable3):
            candidates.append(expand(sol))
    for c in candidates:
        c["ambiguous"] = len(candidates) > 1

    return {
        "id": identity,
        "ping_id": ping_id,
        "ping_t": ping_t,
        "cell_idx": cell_idx,
        "depth": None if depth is None or depth.missing()
        else {"value": depth.to("m").value, "unit": "m",
              "raw": depth.raw, "raw_unit": depth.unit},
        "status": status,
        "cell_reasons": cell_reasons,
        "raw": {"beam_velocity": [
            {"raw": q.raw if q is not None else None,
             "unit": None if q is None else q.unit,
             "value_mps": bvals[k], "diagnostic": bdiag[k]}
            for k, q in enumerate(raw_beams)],
            "correlation": corrs, "amplitude": amps,
            "bad_beam": list(map(bool, bad_flags))},
        "verdicts": [{"beam": v.beam, "tier": v.tier, "reasons": v.reasons,
                      "correlation": v.correlation, "amplitude": v.amplitude,
                      "flagged_bad": v.flagged_bad,
                      "velocity_missing": v.velocity_missing}
                     for v in verdicts],
        "attitude": {"status": att.status,
                     "heading_deg": att.heading_deg,
                     "pitch_deg": att.pitch_deg, "roll_deg": att.roll_deg,
                     "bracket": list(att.bracket) if att.bracket else None,
                     "sources": list(att.source_span) if att.source_span else None},
        "candidates": candidates,
        "provenance": provenance,
        "excluded": status == "cell-invalid",
    }


def derive_bottom_track(ping_id: str, ping_t: float, bt: dict,
                        geom: BeamGeometry, mount: MountAngles,
                        declination: dict, cfg: TransformConfig,
                        attitude: AttitudeSeries,
                        source_versions: dict) -> dict:
    qpolicy = QualityPolicy()
    cell = {"beams": bt["beams"],
            "correlation": bt.get("correlation"),
            "amplitude": bt.get("amplitude"),
            "bad_beam": bt.get("bad_beam"),
            "min_depth_m": 0.0, "blank_m": -1.0, "depth": None}
    res = derive_cell(ping_id, ping_t, 0, 0, cell, geom, mount, declination,
                      qpolicy, cfg, attitude, source_versions)
    res["id"] = f"{ping_id}:bt"
    res["kind"] = "bottom-track"
    res["status"] = "valid"
    res["excluded"] = False
    return res


def relative_to_bottom(cell_candidates: list[dict],
                       bt_candidates: list[dict]) -> list[dict]:
    """Absolute (referenced to bottom) earth velocities per matching candidate."""
    out = []
    bt_by_id = {c["candidate_id"]: c for c in bt_candidates}
    for c in cell_candidates:
        bt = bt_by_id.get(c["candidate_id"])
        entry = {"candidate_id": c["candidate_id"],
                 "mode": c["mode"], "status": "no-matching-bottom-track"}
        if bt and c.get("stages", {}).get("earth") and bt.get("stages", {}).get("earth"):
            we = c["stages"]["earth"]["vector"]
            be = bt["stages"]["earth"]["vector"]
            entry.update({"vector": [we[i] - be[i] for i in range(3)],
                          "unit": "m/s", "basis": "earth-referenced-to-bottom",
                          "status": "ok"})
        out.append(entry)
    return out
