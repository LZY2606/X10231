"""Dependency-graph derivation.

Graph per batch:

    sources (beam_velocity / correlation / amplitude / bad_beam /
             heading, pitch, roll / declination / mount / transducer /
             bottom_track / sample_time)
        -> inst nodes  (per ping, per cell; distinct 3- and 4-beam candidates)
        -> ship nodes  (per ping, per cell)
        -> earth nodes (per ping, per cell)
        -> profile node (per ping; vertical section + excluded-cell evidence)
        -> bottom-track node (per ping)

Only nodes whose dependencies changed are recomputed; unaffected nodes are
copied from the parent batch with ``reused_from`` set, so the UI can explain
why a result is still valid.  Three- and four-beam solutions are never merged
into one value: when no beam is proven bad, every legal 3-beam candidate is
retained.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from typing import Any

import numpy as np

from . import transforms, units
from .attitude import ATTITUDE_RULE, AttitudeSample, interpolate_at
from .cases import CORR_BAD, N_CELLS, N_PINGS, MAX_GAP_S
from .fingerprint import derivation_fp, fingerprint
from .geometry import GEOMETRY_RULE, BeamGeometry, selected_matrix, solve_radial
from .transforms import TRANSFORM_RULE

N_BEAMS = 4


def source_content_uid(row: dict[str, Any]) -> str:
    """Content-derived source uid, stable across export/import/rebuild."""
    body = "|".join(
        str(x)
        for x in (
            row["kind"],
            row.get("ping_idx"),
            row.get("beam"),
            row.get("cell"),
            row.get("raw_repr", row.get("raw")),
            row.get("unit"),
            row.get("ref"),
        )
    )
    return "src_" + hashlib.sha1(body.encode("utf-8")).hexdigest()[:12]


def _num(row: dict[str, Any]) -> float | None:
    if row.get("raw_value") is not None:
        return float(row["raw_value"])
    try:
        return float(row["raw_repr"])
    except (TypeError, ValueError):
        return None


def _stable_config(eff: dict[tuple, dict], kind: str) -> dict[str, Any]:
    row = eff[(kind, None, None, None)]
    if kind == "transducer":
        return {"uid": row["uid"], "value": _num(row), "unit": row["unit"],
                "ref": row["ref"]}
    return {"uid": row["uid"], "value": _num(row), "unit": row["unit"],
            "ref": row["ref"]}


def _mount(eff: dict[tuple, dict]) -> list[dict[str, Any]]:
    rows = [r for r in eff.values() if r["kind"] == "mount"]
    by_ref = {r["ref"]: r for r in rows}
    order = ("mount@install-survey-r1", "mount@install-survey-r2")
    heading = next(
        (by_ref[k] for k in order if k in by_ref),
        next((r for r in rows if "heading" in (r["ref"] or "") or r["ref"] in order), rows[0]),
    )
    pitch = by_ref.get("mount-pitch")
    roll = by_ref.get("mount-roll")
    out = [heading]
    if pitch is not None:
        out.append(pitch)
    if roll is not None:
        out.append(roll)
    if len(out) < 3:  # fallback for fixtures with a single mount record
        pitch = pitch or rows[min(1, len(rows) - 1)]
        roll = roll or rows[-1]
        out = [heading, pitch, roll]
    return out


def _attitude_samples(eff: dict[tuple, dict]) -> list[AttitudeSample]:
    times: dict[int, float] = {}
    head: dict[int, dict] = {}
    pitch: dict[int, dict] = {}
    roll: dict[int, dict] = {}
    for r in eff.values():
        idx = r["ping_idx"]
        if r["kind"] == "sample_time":
            times[idx] = _num(r)
        elif r["kind"] == "heading":
            head[idx] = r
        elif r["kind"] == "pitch":
            pitch[idx] = r
        elif r["kind"] == "roll":
            roll[idx] = r
    samples: list[AttitudeSample] = []
    for idx in sorted(times):
        h = units.angle_rad(head[idx]["raw_repr"], head[idx]["unit"])
        p = units.angle_rad(pitch[idx]["raw_repr"], pitch[idx]["unit"])
        rl = units.angle_rad(roll[idx]["raw_repr"], roll[idx]["unit"])
        if not (h.ok and p.ok and rl.ok):
            continue
        samples.append(AttitudeSample(times[idx], h.value, p.value, rl.value, head[idx]["unit"]))
    return samples


def _cell_inputs(eff, p: int, c: int) -> dict[int, dict[str, Any]]:
    out = {}
    for b in range(N_BEAMS):
        out[b] = {
            "velocity": eff[("beam_velocity", p, b, c)],
            "correlation": eff[("correlation", p, b, c)],
            "amplitude": eff[("amplitude", p, b, c)],
            "bad": eff[("bad_beam", p, b, c)],
        }
    return out


def _cell_exclusion(inputs: dict[int, dict]) -> list[dict[str, Any]]:
    """Return exclusion diagnostics (empty => the cell may be computed)."""
    reasons: list[dict[str, Any]] = []
    for b, d in inputs.items():
        vr = units.velocity_ms(d["velocity"]["raw_repr"], d["velocity"]["unit"])
        if not vr.ok:
            reasons.append({
                "beam": b, "code": "unconvertible_velocity",
                "raw": d["velocity"]["raw_repr"],
                "unit": d["velocity"]["unit"],
                "diagnostic": vr.diagnostic,
            })
        flag = int(float(d["bad"]["raw_repr"]))
        corr = _num(d["correlation"])
        if flag >= 2:
            reasons.append({"beam": b, "code": "hard_bad_beam", "flag": flag})
        if corr is not None and corr < CORR_BAD:
            reasons.append({
                "beam": b, "code": "correlation_below_threshold",
                "value": corr, "threshold": CORR_BAD,
            })
    return reasons


def _beam_states(inputs: dict[int, dict], exclusions: list[dict[str, Any]]):
    """Classify beams as usable (good/suspect) or hard-excluded."""
    excluded_beams = {r["beam"] for r in exclusions if r["code"] in
                      ("hard_bad_beam", "unconvertible_velocity",
                       "correlation_below_threshold")}
    good: list[int] = []
    suspect: list[int] = []
    for b in range(N_BEAMS):
        if b in excluded_beams:
            continue
        flag = int(float(inputs[b]["bad"]["raw_repr"]))
        if flag == 1:
            suspect.append(b)
        else:
            good.append(b)
    return excluded_beams, good, suspect


def build_candidates(
    geom: BeamGeometry,
    inputs: dict[int, dict],
    exclusions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """Build every legal 3-/4-beam candidate for one ping/cell.

    Rules:
    * hard-excluded beams never appear in any candidate;
    * exactly 3 usable beams -> one uniquely-determined 3-beam candidate
      (this is the single-failed-beam case);
    * >=4 usable with any suspect beam -> one 4-beam LS solution plus one
      candidate for every legal 3-beam combination (multi-solution is kept);
    * 4 good beams -> a single 4-beam LS solution with error velocity.
    """
    _, good, suspect = _beam_states(inputs, exclusions)
    usable = sorted(good + suspect)
    cands: list[dict[str, Any]] = []
    if len(usable) < 3:
        return [], "underconstrained"
    radial = {
        b: units.velocity_ms(inputs[b]["velocity"]["raw_repr"],
                             inputs[b]["velocity"]["unit"]).value
        for b in usable
    }

    def threebeam(combo: tuple[int, ...], suspect_beam: int | None) -> None:
        B = selected_matrix(geom, combo)
        v, _ = solve_radial(B, np.array([radial[b] for b in combo]))
        cands.append({
            "ctype": "three-beam",
            "beam_combo": list(combo),
            "suspect_beam": suspect_beam,
            "v_instrument": [float(x) for x in v],
            "error_velocity": None,
            "unique": suspect_beam is not None or len(usable) == 3,
        })

    if len(usable) >= 4:
        B4 = selected_matrix(geom, tuple(usable[:4])) if len(usable) == 4 else None
        if B4 is not None:
            v4, err = solve_radial(B4, np.array([radial[b] for b in usable[:4]]))
            cands.append({
                "ctype": "four-beam",
                "beam_combo": usable[:4],
                "suspect_beams": suspect,
                "v_instrument": [float(x) for x in v4],
                "error_velocity": err,
                "unique": len(suspect) == 0,
            })
        if suspect:
            # Each suspect could be the bad one: retain all legal alternatives.
            for drop in usable:
                combo = tuple(b for b in usable if b != drop)
                if len(combo) == 3:
                    threebeam(combo, drop if drop in suspect else None)
    else:  # exactly 3 usable
        threebeam(tuple(usable), None)
    status = "ambiguous" if any(not c["unique"] for c in cands) or (
        len(cands) > 1 and suspect
    ) else "unique"
    return cands, status


def _vec_q(v: list[float]) -> dict[str, Any]:
    return {
        "components": [
            units.Q(v[0], "m/s").as_dict(),
            units.Q(v[1], "m/s").as_dict(),
            units.Q(v[2], "m/s").as_dict(),
        ],
        "frame_refs": {
            "instrument": "x-fwd/y-star/z-down",
            "ship": "x-fwd/y-star/z-down",
            "earth": "ENU/up",
        },
    }


def _inst_uid(p: int, c: int) -> str:
    return f"inst_p{p}_c{c}"


def _ship_uid(p: int, c: int) -> str:
    return f"ship_p{p}_c{c}"


def _earth_uid(p: int, c: int) -> str:
    return f"earth_p{p}_c{c}"


def _profile_uid(p: int) -> str:
    return f"profile_p{p}"


def _bt_uid(p: int) -> str:
    return f"bottomtrack_p{p}"


def _candidate_uid(p: int, c: int, cand: dict[str, Any]) -> str:
    combo = "".join(str(b) for b in cand["beam_combo"])
    tag = "4b" if cand["ctype"] == "four-beam" else f"3b{combo}"
    return f"cand_p{p}_c{c}_{tag}"


class Engine:
    """Computes nodes/candidates for one effective source snapshot."""

    def __init__(
        self,
        eff_rows: list[dict[str, Any]],
        *,
        rule: str = TRANSFORM_RULE,
        accept: dict[str, list[int]] | None = None,
    ):
        self.eff = {
            (r["kind"], r["ping_idx"], r["beam"], r["cell"]): r for r in eff_rows
        }
        self.rule = rule
        self.accept = accept or {}
        self.transducer = _stable_config(self.eff, "transducer")
        self.declination = _stable_config(self.eff, "declination")
        self.mount = _mount(self.eff)
        beam_rad = math.radians(self.transducer["value"]) \
            if self.transducer["unit"] == "deg" else self.transducer["value"]
        self.geom = BeamGeometry(beam_rad)
        self.attitude = _attitude_samples(self.eff)
        self._mount_rad = [
            units.angle_rad(r["raw_repr"], r["unit"]).value for r in self.mount
        ]
        self._decl_rad = units.angle_rad(self.declination["value"],
                                         self.declination["unit"]).value

    # -------------------------------------------------------------- fingerprint
    def _base_includes(self, p: int, c: int) -> dict[str, str]:
        keys = {}
        for kind in ("beam_velocity", "correlation", "amplitude", "bad_beam"):
            for b in range(N_BEAMS):
                r = self.eff[(kind, p, b, c)]
                keys[f"{kind}.b{b}"] = r["uid"]
        return keys

    def _config_fp_inputs(self) -> dict[str, str]:
        return {
            "declination": self.declination["uid"],
            "mount_heading": self.mount[0]["uid"],
            "mount_pitch": self.mount[1]["uid"],
            "mount_roll": self.mount[2]["uid"],
            "transducer": self.transducer["uid"],
            "rule": self.rule,
            "geom": GEOMETRY_RULE,
            "attitude_rule": ATTITUDE_RULE,
        }

    def compute_all(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        nodes: list[dict[str, Any]] = []
        candidates_out: list[dict[str, Any]] = []
        seq = 0

        def nxt() -> int:
            nonlocal seq
            seq += 1
            return seq

        for p in range(N_PINGS):
            att = interpolate_at(self.attitude, p * 10.0, MAX_GAP_S)
            profile_rows = []
            excluded_rows = []
            for c in range(N_CELLS):
                inputs = _cell_inputs(self.eff, p, c)
                exclusions = _cell_exclusion(inputs, )
                depth_m = (c + 1) * 2.0
                bt_ranges = [
                    _num(self.eff[("bottom_track", p, b, None)])
                    if ("bottom_track", p, b, None) in self.eff else None
                    for b in range(N_BEAMS)
                ]
                # Range records are stored on beam rows with ref=range; find them.
                bt_ranges = self._bt_ranges(p)
                too_shallow = min(bt_ranges) < depth_m if all(
                    x is not None for x in bt_ranges
                ) else False
                if too_shallow:
                    exclusions.append({
                        "code": "invalid_depth_cell",
                        "depth_m": depth_m,
                        "min_bottom_range_m": min(bt_ranges),
                    })

                inst_node, cands = self._inst_node(p, c, inputs, exclusions, nxt)
                nodes.append(inst_node)
                candidates_out.extend(cands)
                chosen = inst_node["payload"].get("chosen_candidate")
                ship_node = self._ship_node(p, c, chosen, inst_node, nxt)
                nodes.append(ship_node)
                earth_node = self._earth_node(
                    p, c, chosen, att, inst_node, ship_node, nxt
                )
                nodes.append(earth_node)

                row = {
                    "cell": c,
                    "depth_m": {"value": depth_m, "unit": "m", "ref": "z-down"},
                    "excluded": bool(exclusions),
                    "reasons": exclusions,
                    "inst_uid": inst_node["uid"],
                    "ship_uid": ship_node["uid"],
                    "earth_uid": earth_node["uid"],
                    "inst_status": inst_node["status"],
                    "earth_status": earth_node["status"],
                }
                if exclusions:
                    excluded_rows.append(row)
                else:
                    row["earth_velocity"] = earth_node["payload"].get("vector")
                    profile_rows.append(row)

            nodes.append(self._profile_node(p, profile_rows, excluded_rows, nxt))
            nodes.append(self._bottomtrack_node(p, nxt))
        return nodes, candidates_out

    def _bt_ranges(self, p: int) -> list[float | None]:
        out = []
        for b in range(N_BEAMS):
            row = self.eff.get(("bottom_track", p, b, None))
            if row is not None and row.get("ref") == "range":
                out.append(_num(row))
        return out if out else [None] * N_BEAMS

    def _bt_velocities(self, p: int) -> list[float | None]:
        out = []
        for b in range(N_BEAMS):
            row = self.eff.get(("bottom_track", p, b, None))
            if row is not None and row.get("ref") == "radial":
                q = units.velocity_ms(row["raw_repr"], row["unit"])
                out.append(q.value if q.ok else None)
        return out

    # ------------------------------------------------------------------ inst
    def _inst_node(self, p, c, inputs, exclusions, nxt) -> tuple[dict, list]:
        includes = self._base_includes(p, c)
        includes["transducer"] = self.transducer["uid"]
        includes["geom_rule"] = GEOMETRY_RULE
        if exclusions:
            fp = derivation_fp(
                rule=self.rule, geom_rule=GEOMETRY_RULE,
                declination_uid=self.declination["uid"],
                mount_uid=self.mount[0]["uid"], sources=includes,
                extra={"node": "inst", "excluded": True},
            )
            return {
                "uid": _inst_uid(p, c), "kind": "instrument", "ping_idx": p,
                "cell": c, "status": "excluded", "fingerprint": fp,
                "includes": includes, "seq": nxt(),
                "payload": {"reasons": exclusions, "candidates": []},
            }, []
        cands, cand_status = build_candidates(self.geom, inputs, exclusions)
        chosen_combo = self.accept.get(_inst_uid(p, c))
        chosen = None
        for cand in cands:
            cand["v_ship"] = self._to_ship(cand["v_instrument"]).tolist()
            cf = derivation_fp(
                rule=self.rule, geom_rule=GEOMETRY_RULE,
                declination_uid=self.declination["uid"],
                mount_uid=self.mount[0]["uid"], sources=includes,
                extra={"combo": cand["beam_combo"], "ctype": cand["ctype"],
                       "v": [round(x, 9) for x in cand["v_instrument"]]},
            )
            cand["fingerprint"] = cf
            cand["uid"] = _candidate_uid(p, c, cand)
            if chosen_combo is not None and cand["beam_combo"] == list(chosen_combo):
                chosen = cand
        if chosen is None:
            unique = [x for x in cands if x.get("unique")]
            if len(unique) == 1:
                chosen = unique[0]
        ambiguous = chosen is None and len(cands) > 1
        status = "unresolved" if ambiguous else "valid"
        fp = derivation_fp(
            rule=self.rule, geom_rule=GEOMETRY_RULE,
            declination_uid=self.declination["uid"],
            mount_uid=self.mount[0]["uid"], sources=includes,
            extra={
                "candidate_fps": sorted(x["fingerprint"] for x in cands),
                "chosen": chosen["fingerprint"] if chosen else None,
            },
        )
        payload = {
            "candidates": [self._cand_public(x) for x in cands],
            "chosen_candidate": chosen["uid"] if chosen else None,
            "ambiguous": ambiguous,
            "beam_quality": {
                str(b): {
                    "correlation_counts": _num(inputs[b]["correlation"]),
                    "amplitude_counts": _num(inputs[b]["amplitude"]),
                    "bad_flag": int(float(inputs[b]["bad"]["raw_repr"])),
                }
                for b in range(N_BEAMS)
            },
            "vector_ref": "instrument x-fwd/y-star/z-down",
        }
        node = {
            "uid": _inst_uid(p, c), "kind": "instrument", "ping_idx": p,
            "cell": c, "status": status, "fingerprint": fp,
            "includes": includes, "seq": nxt(), "payload": payload,
        }
        return node, [self._cand_row(p, c, x, chosen is x) for x in cands]

    def _to_ship(self, v_inst: list[float]) -> np.ndarray:
        mh, mp, mr = self._mount_rad
        return transforms.instrument_to_ship(mh, mp, mr) @ np.asarray(v_inst)

    @staticmethod
    def _cand_public(cand: dict) -> dict:
        return {
            "uid": cand["uid"],
            "type": cand["ctype"],
            "beam_combo": cand["beam_combo"],
            "suspect_beam": cand.get("suspect_beam"),
            "error_velocity": cand["error_velocity"],
            "v_instrument": _vec_q(cand["v_instrument"]),
            "v_ship": _vec_q(cand["v_ship"]),
            "fingerprint": cand["fingerprint"],
            "unique": cand["unique"],
        }

    @staticmethod
    def _cand_row(p, c, cand, accepted) -> dict:
        return {
            "uid": cand["uid"], "node_uid": _inst_uid(p, c),
            "ctype": cand["ctype"], "beam_combo": cand["beam_combo"],
            "fingerprint": cand["fingerprint"], "accepted": bool(accepted),
            "seq": p * 1000 + c * 20 + (0 if cand["ctype"] == "four-beam" else 1)
                   + cand["beam_combo"][-1],
            "payload": {
                "v_instrument": cand["v_instrument"],
                "v_ship": cand["v_ship"],
                "error_velocity": cand["error_velocity"],
                "unique": cand["unique"],
            },
        }

    # ------------------------------------------------------------------ ship
    def _ship_node(self, p, c, chosen, inst_node, nxt) -> dict:
        includes = dict(inst_node["includes"])
        includes["mount_heading"] = self.mount[0]["uid"]
        includes["mount_pitch"] = self.mount[1]["uid"]
        includes["mount_roll"] = self.mount[2]["uid"]
        includes["rule"] = self.rule
        if inst_node["status"] == "excluded":
            status = "excluded"
            vector = None
        elif chosen is None:
            status = "unresolved"
            vector = None
        else:
            status = "valid"
            vector = chosen["v_ship"]
        fp = derivation_fp(
            rule=self.rule, geom_rule=GEOMETRY_RULE,
            declination_uid=self.declination["uid"],
            mount_uid=self.mount[0]["uid"], sources=includes,
            extra={"node": "ship", "inst_fp": inst_node["fingerprint"]},
        )
        return {
            "uid": _ship_uid(p, c), "kind": "ship", "ping_idx": p, "cell": c,
            "status": status, "fingerprint": fp, "includes": includes,
            "seq": nxt(),
            "payload": {
                "vector": _vec_q(vector) if vector is not None else None,
                "chosen_candidate": chosen["uid"] if chosen else None,
                "candidate_vectors": {
                    x["uid"]: x["v_ship"]
                    for x in inst_node["payload"]["candidates"]
                },
                "vector_ref": "ship x-fwd/y-star/z-down",
                "mount_heading_deg": self._mount_rad[0] * 180.0 / math.pi,
            },
        }

    # ----------------------------------------------------------------- earth
    def _earth_node(self, p, c, chosen, att, inst_node, ship_node, nxt) -> dict:
        includes = dict(ship_node["includes"])
        includes["declination"] = self.declination["uid"]
        includes["attitude_rule"] = ATTITUDE_RULE
        if att.bracket:
            includes["attitude_bracket"] = f"t{att.bracket[0]:g}-t{att.bracket[1]:g}"
        in_gap = att.in_gap
        if inst_node["status"] == "excluded":
            status, vector, cmap = "excluded", None, {}
            reason = "cell excluded upstream"
        elif in_gap:
            status, vector, cmap = "unresolved", None, {}
            reason = {
                "code": "attitude_gap",
                "bracket_t": att.bracket,
                "gap_s": att.gap_s,
                "max_gap_s": MAX_GAP_S,
                "policy": ATTITUDE_RULE,
            }
        elif chosen is None:
            status, vector, cmap = "unresolved", None, {}
            reason = {"code": "ambiguous_three_beam"}
        else:
            r_se = transforms.ship_to_earth(att.heading_rad, att.pitch_rad,
                                            att.roll_rad) if False else None
            h_true = transforms.true_heading(att.heading_rad, self._decl_rad)
            R = transforms.ship_to_earth(h_true, att.pitch_rad, att.roll_rad)
            cmap = {}
            for x in inst_node["payload"]["candidates"]:
                cmap[x["uid"]] = [float(q) for q in (R @ np.asarray(x["v_ship"]))]
            vector = cmap[chosen["uid"]]
            status = "valid"
            reason = None
        fp = derivation_fp(
            rule=self.rule, geom_rule=GEOMETRY_RULE,
            declination_uid=self.declination["uid"],
            mount_uid=self.mount[0]["uid"], sources=includes,
            extra={"node": "earth", "ship_fp": ship_node["fingerprint"],
                   "in_gap": in_gap},
        )
        payload: dict[str, Any] = {
            "vector": _vec_q(vector) if vector is not None else None,
            "chosen_candidate": chosen["uid"] if chosen else None,
            "candidate_vectors": {k: _vec_q(v) for k, v in cmap.items()},
            "vector_ref": "earth ENU/up",
            "true_heading_deg": None
            if (in_gap or chosen is None or inst_node["status"] == "excluded")
            else h_true * 180.0 / math.pi,
            "attitude_interpolated": att.interpolated,
            "attitude_bracket_t": att.bracket,
            "declination_ref": self.declination["ref"],
        }
        if reason is not None:
            payload["unresolved"] = reason
        return {
            "uid": _earth_uid(p, c), "kind": "earth", "ping_idx": p, "cell": c,
            "status": status, "fingerprint": fp, "includes": includes,
            "seq": nxt(), "payload": payload,
        }
