"""Dependency-graph derivation engine.

Inputs are the immutable source versions visible at one commit. Outputs are
content-addressed derivations plus per-slot status, so only fingerprints that
changed are recomputed and every remaining slot can explain why it is still
valid (same input versions, rule version, geometry and declination versions).
"""
from __future__ import annotations

from typing import Any, Optional

from . import RULE_VERSION
from .fingerprint import content_id, derivation_fingerprint, event
from .geometry import (
    TransducerGeometry,
    apply_transform_chain,
    three_beam_candidates,
)
from .quality import QualityThresholds, judge_beam, judge_cell, judge_ping
from .units import (
    interpolate_gap_aware,
    normalize_heading_deg,
    unwrap_angle_deg,
)


def _geo(transducer: dict) -> TransducerGeometry:
    payload = transducer["payload"]
    return TransducerGeometry(
        beam_angle_deg=payload["beam_angle_deg"],
        azimuths_deg=tuple(payload["azimuths_deg"]),
    )


def _thresholds(rules: dict) -> QualityThresholds:
    p = rules["payload"]
    return QualityThresholds(
        correlation_min=float(p["correlation_min"]),
        amplitude_min=float(p["amplitude_min"]),
        attitude_max_gap_s=float(p["attitude_max_gap_s"]),
        sidelobe_margin_m=float(p.get("sidelobe_margin_m", 0.5)),
    )


def _canon(quantity: dict) -> Optional[float]:
    return quantity.get("canonical") if quantity.get("diagnostic") is None else None


def build_context(sources: dict[str, dict]) -> dict:
    docs = {rid: rec for rid, rec in sources.items()
            if rid.startswith("doc:")}
    pings = {rid: rec for rid, rec in sources.items()
             if rec["record_kind"] == "ping"}
    attitude = [rec for rid, rec in sources.items()
                if rec["record_kind"] == "attitude"]
    attitude.sort(key=lambda rec: rec["payload"]["t_s"])
    return {
        "transducer": docs["doc:transducer"],
        "mount": docs["doc:mount"],
        "declination": docs["doc:declination"],
        "rules": docs["doc:rules"],
        "pings": pings,
        "attitude": attitude,
    }


def input_versions(ctx: dict, ping_id: Optional[str] = None) -> dict:
    versions = {
        "transducer": ctx["transducer"]["version"],
        "mount": ctx["mount"]["version"],
        "declination": ctx["declination"]["version"],
        "rules": ctx["rules"]["version"],
        "attitude_samples": sorted(
            rec["version"] for rec in ctx["attitude"]
        ),
    }
    if ping_id is not None:
        rec = next((r for rid, r in ctx["pings"].items()
                    if rid == f"ping:{ping_id}"), None)
        if rec is not None:
            versions["ping"] = rec["version"]
    return versions


def align_attitude(ctx: dict) -> dict:
    """Gap-aware attitude alignment for every ping timestamp.

    Headings are unwrapped only within each continuous segment (a missing
    sample splits the segments), interpolated, then normalized back to
    [0,360). True heading adds the versioned magnetic declination.
    """
    thresholds = _thresholds(ctx["rules"])
    times = [rec["payload"]["t_s"] for rec in ctx["attitude"]]
    head_mag = [
        None if rec["payload"]["heading"].get("diagnostic")
        else rec["payload"]["heading"]["canonical"]
        for rec in ctx["attitude"]
    ]
    pitch = [
        None if rec["payload"]["pitch"].get("diagnostic")
        else rec["payload"]["pitch"]["canonical"]
        for rec in ctx["attitude"]
    ]
    roll = [
        None if rec["payload"]["roll"].get("diagnostic")
        else rec["payload"]["roll"]["canonical"]
        for rec in ctx["attitude"]
    ]
    unwrapped = unwrap_angle_deg(head_mag)
    ping_times = sorted(rec["payload"]["t_s"] for rec in ctx["pings"].values())
    head_i, head_diag = interpolate_gap_aware(
        times, unwrapped, ping_times, thresholds.attitude_max_gap_s
    )
    pitch_i, pitch_diag = interpolate_gap_aware(
        times, pitch, ping_times, thresholds.attitude_max_gap_s
    )
    roll_i, roll_diag = interpolate_gap_aware(
        times, roll, ping_times, thresholds.attitude_max_gap_s
    )
    decl = float(ctx["declination"]["payload"]["declination_deg"])
    aligned = {}
    for i, t in enumerate(ping_times):
        diags = [d for d in (head_diag[i], pitch_diag[i], roll_diag[i]) if d]
        entry = {"t_s": t, "diagnostics": diags}
        if head_i[i] is not None:
            true_h = normalize_heading_deg(head_i[i] + decl)
            entry.update({
                "heading_magnetic_deg": normalize_heading_deg(head_i[i]),
                "heading_true_deg": true_h,
                "pitch_deg": pitch_i[i],
                "roll_deg": roll_i[i],
            })
        aligned[f"{t:.6g}"] = entry
    return aligned


def _mount_angles(mount: dict) -> tuple[float, float, float]:
    p = mount["payload"]
    return (float(p["alpha_deg"]), float(p["beta_deg"]),
            float(p["gamma_deg"]))


def compute_ping(ctx: dict, record: dict, aligned: dict,
                 order: str = "v1_separated") -> dict:
    thresholds = _thresholds(ctx["rules"])
    geom = _geo(ctx["transducer"])
    mount = _mount_angles(ctx["mount"])
    payload = record["payload"]
    pid = payload["ping_id"]
    att = aligned.get(f"{payload['t_s']:.6g}", {})
    cells_out = []
    for cell in payload["cells"]:
        idx = cell["index"]
        bt_depth = None
        if "bottom_track" in payload:
            bt_depth = _canon(payload["bottom_track"]["depth_m"])
        cell_verdict = judge_cell(
            pid, idx,
            cell["center_m"].get("canonical") or 0.0,
            cell["configured"], bt_depth,
            ctx["transducer"]["payload"]["beam_angle_deg"], thresholds,
        )
        verdicts = []
        velocities = []
        for b in range(4):
            bv = cell["beam_velocity"][b]
            velocities.append(_canon(bv))
            diag = bv.get("diagnostic")
            velocity_unusable = diag in (
                "missing", "non_finite"
            ) or (isinstance(diag, str) and diag.startswith("unparseable"))
            verdicts.append(judge_beam(
                b, not velocity_unusable,
                _canon(cell["correlation"][b]),
                _canon(cell["amplitude"][b]),
                cell["bad_flag"][b], thresholds,
            ))
        ping_verdict = judge_ping(verdicts)
        valid = [v["valid"] for v in verdicts]
        slot_name = f"ping:{pid}:cell:{idx}"
        result = {
            "ping_id": pid, "cell": idx,
            "cell_verdict": cell_verdict,
            "beam_quality": ping_verdict,
        }
        if not cell_verdict["included"]:
            result["status"] = "cell_excluded"
            cells_out.append(result)
            continue
        weak = [
            i for i, v in enumerate(verdicts)
            if v["valid"] and v["marginal"]
        ]
        three_candidates = []
        if ping_verdict["state"] == "four_beam_marginal" and weak:
            # Every way of dropping one weak beam is a legal 3-beam answer;
            # the system must not silently choose one.
            three_candidates = [
                cand for cand in three_beam_candidates(
                    geom, velocities, [True] * 4
                )
                if cand["dropped_beam"] in weak
            ]
        if att.get("diagnostics") or "heading_true_deg" not in att:
            result["status"] = "attitude_unavailable"
            result["attitude_diagnostics"] = att.get("diagnostics",
                                                      ["no_sample"])
            cells_out.append(result)
            continue
        chain = apply_transform_chain(
            velocities, valid, geom, mount,
            att["heading_true_deg"], att["pitch_deg"], att["roll_deg"],
            order=order,
        )
        result["transform"] = chain
        result["attitude_used"] = att
        if three_candidates:
            full_cands = []
            for cand in three_candidates:
                cand_valid = [i != cand["dropped_beam"] for i in range(4)]
                cand_chain = apply_transform_chain(
                    velocities, cand_valid, geom, mount,
                    att["heading_true_deg"], att["pitch_deg"],
                    att["roll_deg"], order=order,
                )
                full_cands.append({**cand, "chain": cand_chain})
            result["three_beam_candidates"] = full_cands
            result["four_beam_transform"] = chain
            result["status"] = "ambiguous_three_or_four"
        else:
            result["status"] = chain["beam_solution"]["kind"]
        cells_out.append(result)
    return {"ping_id": pid, "t_s": payload["t_s"], "cells": cells_out,
            "bottom_track": payload.get("bottom_track")}


def _fp(slot: str, ctx: dict, payload_inputs: dict, order: str) -> str:
    return derivation_fingerprint(
        slot=slot,
        rule_version=RULE_VERSION,
        transform_order=order,
        geometry=ctx["transducer"]["payload"],
        thresholds=_thresholds(ctx["rules"]).fingerprint_parts(),
        declination_version=ctx["declination"]["version"],
        mount_version=ctx["mount"]["version"],
        inputs=payload_inputs,
    )


def compute_state(sources: dict[str, dict], order: str = "v1_separated",
                  ) -> dict:
    """Compute every slot from an immutable source view."""
    ctx = build_context(sources)
    aligned = align_attitude(ctx)
    thresholds = _thresholds(ctx["rules"])
    derivations: list[dict] = []
    slot_status: dict[str, dict] = {}
    events: list[dict] = []
    graph: dict[str, list[str]] = {}

    dep_doc = ["doc:transducer", "doc:mount", "doc:declination", "doc:rules"]
    pings_sorted = sorted(ctx["pings"].values(),
                          key=lambda rec: rec["payload"]["t_s"])
    for record in pings_sorted:
        rid = f"ping:{record['payload']['ping_id']}"
        computed = compute_ping(ctx, record, aligned, order=order)
        for cell in computed["cells"]:
            slot = f"{rid}:cell:{cell['cell']}"
            inputs = input_versions(ctx, record["payload"]["ping_id"])
            fp = _fp(slot, ctx, inputs, order)
            status = cell["status"]
            derivations.append({
                "fingerprint": fp, "slot": slot, "status": status,
                "payload": cell, "rule_version": RULE_VERSION,
                "transform_order": order,
            })
            slot_status[slot] = {"status": status, "fingerprint": fp}
            graph[slot] = dep_doc + [rid]
            for v in cell["beam_quality"]["verdicts"]:
                if not v["valid"]:
                    events.append(event(
                        "beam_excluded", f"{slot}:beam:{v['beam']}",
                        principles=["P-raw-preserved"],
                        detail={"reasons": v["reasons"],
                                "cell_status": status},
                    ))
            if status == "cell_excluded":
                events.append(event(
                    "cell_excluded", slot,
                    principles=["P-cell-excluded"],
                    detail={"reasons": cell["cell_verdict"]["reasons"]},
                ))
            if status == "attitude_unavailable":
                events.append(event(
                    "attitude_gap_blocked", slot,
                    principles=["P-no-cross-gap"],
                    detail={"diagnostics": cell["attitude_diagnostics"]},
                ))
            if status == "three_beam":
                events.append(event(
                    "three_beam_solution", slot,
                    principles=["P-3v4-separate"],
                    detail={"valid_beams": cell["beam_quality"]["valid_beams"]},
                ))
            if status == "ambiguous_three_or_four":
                events.append(event(
                    "multi_solution_retained", slot,
                    principles=["P-3v4-separate", "P-multiple-candidates"],
                    detail={
                        "candidate_count": len(cell["three_beam_candidates"]),
                        "four_beam_present": True,
                    },
                ))

    # Attitude alignment slots (per ping time) make gap handling visible.
    for key, att in aligned.items():
        slot = f"attitude:t:{key}"
        fp = _fp(slot, ctx, input_versions(ctx), order)
        ok = "heading_true_deg" in att
        derivations.append({
            "fingerprint": fp, "slot": slot,
            "status": "aligned" if ok else "blocked_gap",
            "payload": att, "rule_version": RULE_VERSION,
            "transform_order": order,
        })
        slot_status[slot] = {
            "status": "aligned" if ok else "blocked_gap",
            "fingerprint": fp,
        }
        graph[slot] = dep_doc + [
            f"attitude:{r['payload']['sample_id']}" for r in ctx["attitude"]
        ]
        if not ok:
            events.append(event(
                "attitude_gap", slot,
                principles=["P-no-cross-gap"],
                detail={"diagnostics": att["diagnostics"]},
            ))

    # Profile slot: one ensemble per cell depth using included slots only.
    profile = _build_profile(slot_status, derivations)
    fp = _fp("profile:earth_enu", ctx, input_versions(ctx), order)
    derivations.append({
        "fingerprint": fp, "slot": "profile:earth_enu",
        "status": "ok", "payload": profile,
        "rule_version": RULE_VERSION, "transform_order": order,
    })
    slot_status["profile:earth_enu"] = {"status": "ok", "fingerprint": fp}
    graph["profile:earth_enu"] = sorted(slot_status.keys())
    return {"derivations": derivations, "slot_status": slot_status,
            "events": events, "graph": graph, "aligned": aligned}


def _build_profile(slot_status: dict, derivations: list[dict]) -> dict:
    by_fp = {d["fingerprint"]: d for d in derivations}
    series: dict[int, list[dict]] = {}
    for slot, spec in slot_status.items():
        if not slot.startswith("ping:") or ":cell:" not in slot:
            continue
        der = by_fp.get(spec["fingerprint"])
        if der is None:
            continue
        payload = der["payload"]
        cell = payload["cell"]
        if payload["status"] in ("cell_excluded", "attitude_unavailable",
                                 "insufficient_beams"):
            continue
        chain = payload.get("transform") or payload.get("four_beam_transform")
        if chain and chain.get("earth_ms") is not None:
            series.setdefault(cell, []).append({
                "ping": payload["ping_id"],
                "status": payload["status"],
                "kind": chain["beam_solution"]["kind"],
                "earth_ms": chain["earth_ms"],
                "error_velocity_ms":
                    chain["beam_solution"].get("error_velocity_ms"),
            })
    depths = sorted(series)
    ensembles = []
    import statistics

    for depth in depths:
        members = series[depth]
        # 3-beam and 4-beam members are never averaged together.
        four = [m for m in members if m["kind"] == "four_beam"]
        three = [m for m in members if m["kind"] == "three_beam"]
        ambig = [m for m in members if m["status"] == "ambiguous_three_or_four"]
        for group_name, group in (("four_beam", four), ("three_beam", three)):
            if not group:
                continue
            comps = list(zip(*(m["earth_ms"] for m in group)))
            ensembles.append({
                "cell": depth,
                "solution_kind": group_name,
                "n": len(group),
                "earth_mean_ms": [statistics.fmean(c) for c in comps],
                "members": [m["ping"] for m in group],
            })
        if ambig:
            ensembles.append({
                "cell": depth,
                "solution_kind": "ambiguous",
                "n": len(ambig),
                "earth_mean_ms": None,
                "members": [m["ping"] for m in ambig],
            })
    return {"ensembles": ensembles}
