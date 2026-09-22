"""Derivation engine: raw sources -> instrument/ship/earth velocities.

The engine is a pure function over one version's sources plus the previous
version's node revisions.  It returns exactly the node revisions that must
be written for the new commit; everything else stays valid by graph
propagation.

Guaranteed semantics (tested):
  * a three-beam solution and a four-beam solution are never collapsed into
    one value; every legal candidate is kept with its support set;
  * attitude interpolation never crosses a time gap wider than the rule;
  * transform order, beam geometry and declination version are part of the
    derivation fingerprint.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .fingerprint import canonical, fingerprint
from .geometry import BeamGeometry, geometry_from_config
from .quality import QualityRules, cell_valid, grade_beam
from .rotation import (instrument_to_ship, ship_to_earth, true_heading,
                       unwrap_angles)
from . import dag

STATUS_OK = "ok"
STATUS_MULTI = "multi"               # legal candidates, awaiting decision
STATUS_BLOCKED = "blocked"           # missing/invalid upstream input
STATUS_EXCLUDED = "excluded"         # cell or beam excluded with evidence


@dataclass
class RuleSet:
    geometry: BeamGeometry
    mounting: dict
    declination: dict
    quality: QualityRules
    rotation_recipe: str = "HPR"
    mount_recipe: str = "ALPHA_BETA"

    def fingerprints(self) -> Dict[str, str]:
        return {
            "geometry": fingerprint("geo", self.geometry.fingerprint_inputs()),
            "mounting": fingerprint("mount", self.mounting),
            "declination": fingerprint("decl", self.declination),
            "rules": fingerprint("rules", self.quality.fingerprint()),
            "rotation_recipe": fingerprint("rot", {
                "ship_earth": self.rotation_recipe,
                "instrument_ship": self.mount_recipe}),
        }


def ruleset_from_sources(configs: dict) -> RuleSet:
    geo = geometry_from_config(configs.get("geometry", {}))
    mounting = configs.get("mounting", {
        "heading_offset_deg": 0.0, "tilt_deg": 0.0, "recipe": "ALPHA_BETA"})
    declination = configs.get("declination", {
        "value_deg": 0.0, "convention": "east_positive", "epoch": "baseline"})
    qr = configs.get("rules", {})
    rules = QualityRules(**{k: v for k, v in qr.items()
                            if k in QualityRules.__dataclass_fields__})
    return RuleSet(geo, mounting, declination, rules,
                   rotation_recipe=configs.get("rotation_recipe", "HPR"),
                   mount_recipe=mounting.get("recipe", "ALPHA_BETA"))


# ---------------------------------------------------------------------- helpers
def _qval(q):
    """Return (value, None) from a stored quantity dict or (None, diag)."""
    if q is None:
        return None, "missing"
    if isinstance(q, dict):
        if q.get("valid"):
            return q["value"], None
        return None, q.get("diagnostic", "invalid")
    return None, "not_a_quantity"


def _vec(v: np.ndarray, unit: str, datum: str) -> dict:
    return {"components": [float(x) for x in np.asarray(v, dtype=float)],
            "unit": unit, "datum": datum}


def node_id(kind: str, scope: str, tag: Optional[str] = None) -> str:
    return "%s:%s%s" % (kind, scope, ("#" + tag) if tag else "")


def _candidate(cid: str, support: List[int], v: np.ndarray, datum: str,
               error_velocity: Optional[float], provenance: str) -> dict:
    return {"candidate_id": cid, "support_beams": support,
            "velocity": _vec(v, "m/s", datum),
            "error_velocity": (None if error_velocity is None
                               else float(error_velocity)),
            "provenance": provenance}


class Evaluator:
    """Evaluate one version.

    sources : {source_id: {"kind","datum","raw"}} where raw is either a
              stored quantity dict (import layer) or a plain value.
    previous: {node_id: node_rev dict} from the parent version.
    decisions: {scope: chosen candidate_id} carried from decision events.
    """

    def __init__(self, sources: Dict[str, dict],
                 previous: Optional[Dict[str, dict]] = None,
                 decisions: Optional[Dict[str, str]] = None,
                 changed: Optional[Sequence[str]] = None):
        self.sources = sources
        self.previous = previous or {}
        self.decisions = decisions or {}
        self.changed = set(changed or [])
        self.nodes: List[dict] = []
        self._by_id: Dict[str, dict] = {}
        self.diagnostics: List[dict] = []
        self.scopes = self._discover_scopes()
        self.attitude_samples = self._discover_attitude()
        self.configs = self._discover_configs()
        self.ruleset = ruleset_from_sources(self.configs)
        self.fp = self.ruleset.fingerprints()

    # ------------------------------------------------------------- discovery
    def _raw(self, sid: str):
        s = self.sources.get(sid)
        return None if s is None else s.get("raw")

    def _discover_scopes(self) -> List[dict]:
        pings: Dict[str, set] = {}
        for sid in self.sources:
            if sid.startswith("attitude."):
                continue
            parts = sid.split(".")
            if parts[0].startswith("ping-"):
                pings.setdefault(parts[0], set()).add(sid)
        out = []
        for ping in sorted(pings):
            cells = set()
            beams = set()
            for sid in pings[ping]:
                parts = sid.split(".")
                if "cell" in parts:
                    ci = parts.index("cell")
                    cells.add(int(parts[ci + 1]))
                if "beam" in parts:
                    bi = parts.index("beam")
                    beams.add(int(parts[bi + 1]))
            out.append({"ping": ping,
                        "cells": sorted(cells),
                        "beams": sorted(beams),
                        "time_source": "%s.time" % ping})
        return out

    def _discover_attitude(self) -> List[dict]:
        sample_ids = set()
        for sid in self.sources:
            if sid.startswith("attitude.") and sid.endswith(".time"):
                sample_ids.add(sid.rsplit(".", 1)[0])
        samples = []
        for base in sorted(sample_ids):
            t, td = _qval(self._raw(base + ".time"))
            h, hd = _qval(self._raw(base + ".heading"))
            p, pd = _qval(self._raw(base + ".pitch"))
            r, rd = _qval(self._raw(base + ".roll"))
            samples.append({"id": base, "t": t, "heading": h,
                            "pitch": p, "roll": r,
                            "diags": [d for d in (td, hd, pd, rd) if d]})
        return samples

    def _discover_configs(self) -> dict:
        cfg = {}
        for sid, s in self.sources.items():
            if sid in ("geometry", "mounting", "declination", "rules",
                       "rotation_recipe"):
                cfg[sid] = s.get("raw")
        return cfg

    # -------------------------------------------------------------- attitude
    def evaluate_attitude(self) -> Dict[str, dict]:
        """Unwrap heading through the 360 crossover; interpolate per segment.

        Interpolation is segment-local: samples separated by a gap larger
        than rules.gap_seconds are never bridged.  A ping landing in a gap
        gets a BLOCKED attitude node carrying the gap evidence.
        """
        out: Dict[str, dict] = {}
        samples = self.attitude_samples
        valid = [s for s in samples if s["t"] is not None]
        if not valid:
            return out
        headings = unwrap_angles([s["heading"] for s in valid])
        gap = self.ruleset.quality.gap_seconds
        segments: List[List[int]] = []
        for i, s in enumerate(valid):
            if i == 0 or (s["t"] - valid[i - 1]["t"]) > gap:
                segments.append([i])
            else:
                segments[-1].append(i)

        # one node per attitude sample (unwrapped value + segment id)
        for seg_i, idxs in enumerate(segments):
            for j in idxs:
                s = valid[j]
                payload = {
                    "t": s["t"], "unit_t": "unix_s",
                    "heading_unwrapped_deg": float(headings[j]),
                    "heading_raw_deg": s["heading"],
                    "pitch_deg": s["pitch"], "roll_deg": s["roll"],
                    "segment": seg_i,
                    "gap_seconds_rule": gap,
                    "crossed_360": bool(abs(headings[j] - s["heading"]) > 1e-9),
                }
                nid = node_id("attitude", s["id"])
                out[nid] = self._node(nid, STATUS_OK, payload,
                                      scope=s["id"], reason="attitude_sample")

        # attitude at each ping: interpolate only inside its segment
        for sc in self.scopes:
            t, td = _qval(self._raw(sc["time_source"]))
            nid = node_id("attitude", sc["ping"])
            if t is None:
                out[nid] = self._node(
                    nid, STATUS_BLOCKED, {"diagnostic": td or "no_time"},
                    scope=sc["ping"], reason="no_ping_time")
                continue
            placed = False
            for seg_i, idxs in enumerate(segments):
                ts = [valid[j]["t"] for j in idxs]
                if ts[0] - gap <= t <= ts[-1] + gap:
                    # exact sample?
                    hit = next((j for j in idxs if abs(valid[j]["t"] - t)
                                < 1e-9), None)
                    if hit is not None:
                        s = valid[hit]
                        h = float(headings[hit])
                    elif len(idxs) >= 2 and ts[0] <= t <= ts[-1]:
                        h = float(np.interp(t, ts, [headings[j] for j in idxs]))
                        p = float(np.interp(t, ts, [valid[j]["pitch"]
                                                    for j in idxs]))
                        r = float(np.interp(t, ts, [valid[j]["roll"]
                                                    for j in idxs]))
                        s = {"pitch": p, "roll": r}
                    else:
                        break  # inside guard band but outside real coverage
                    payload = {
                        "t": t, "unit_t": "unix_s",
                        "heading_unwrapped_deg": h,
                        "heading_raw_deg": (h + 360.0) % 360.0,
                        "pitch_deg": s["pitch"], "roll_deg": s["roll"],
                        "segment": seg_i, "interpolated": hit is None,
                        "gap_seconds_rule": gap,
                    }
                    out[nid] = self._node(nid, STATUS_OK, payload,
                                          scope=sc["ping"],
                                          reason="attitude_segment_%d" % seg_i)
                    placed = True
                    break
            if not placed:
                out[nid] = self._node(
                    nid, STATUS_BLOCKED,
                    {"t": t, "unit_t": "unix_s",
                     "diagnostic": "attitude_gap",
                     "gap_seconds_rule": gap,
                     "evidence": "no two attitude samples within rule window"},
                    scope=sc["ping"], reason="attitude_gap_not_bridged")
        return out

    # --------------------------------------------------------- cells/beams
    def evaluate_cell_and_beams(self, ping: str, cell: int) -> Tuple[dict, Dict[int, dict]]:
        scope = "%s.cell.%d" % (ping, cell)
        depth, ddiag = _qval(self._raw("%s.cell.depth" % scope))
        valid = cell_valid(depth, self.ruleset.quality)
        if depth is None:
            reason = "depth:" + (ddiag or "missing")
            status = STATUS_EXCLUDED
        elif not valid:
            reason = "depth_out_of_range"
            status = STATUS_EXCLUDED
        else:
            reason = "depth_ok"
            status = STATUS_OK
        cell_node = self._node(
            node_id("cell_validity", scope), status,
            {"depth_m": depth, "depth_diagnostic": ddiag,
             "valid_range_m": [self.ruleset.quality.min_depth_m,
                               self.ruleset.quality.max_depth_m]},
            scope=scope, reason=reason)
        beam_nodes: Dict[int, dict] = {}
        for b in range(
                1, self.ruleset.geometry.n_beams + 1):
            corr, cd = _qval(self._raw("%s.beam.%d.corr" % (scope, b)))
            amp, ad = _qval(self._raw("%s.beam.%d.amp" % (scope, b)))
            bad_raw = self._raw("%s.beam.%d.bad" % (scope, b))
            if isinstance(bad_raw, dict):
                bad = bad_raw.get("value", bad_raw.get("raw"))
            else:
                bad = bad_raw
            try:
                bad = bool(bad)
            except (TypeError, ValueError):
                bad = None
            tier = grade_beam(corr, amp, bad, self.ruleset.quality)
            reasons = []
            if bad is True:
                reasons.append("bad_flag")
            if corr is None:
                reasons.append("corr:" + (cd or "missing"))
            elif corr < self.ruleset.quality.min_correlation:
                reasons.append("corr_below_%g" %
                               self.ruleset.quality.min_correlation)
            if amp is None:
                reasons.append("amp:" + (ad or "missing"))
            elif amp < self.ruleset.quality.min_amplitude:
                reasons.append("amp_below_%g" %
                               self.ruleset.quality.min_amplitude)
            beam_nodes[b] = self._node(
                node_id("beam_quality", scope, str(b)), tier,
                {"beam": b, "correlation": corr, "corr_diagnostic": cd,
                 "amplitude": amp, "amp_diagnostic": ad,
                 "bad_flag": bad, "tier": tier,
                 "limits": {"min_correlation":
                            self.ruleset.quality.min_correlation,
                            "min_amplitude":
                            self.ruleset.quality.min_amplitude}},
                scope="%s.beam.%d" % (scope, b),
                reason=";".join(reasons) or "within_limits")
        return cell_node, beam_nodes

    # ------------------------------------------------------------- velocity
    def evaluate_instrument_velocity(self, ping: str, cell: int,
                                     cell_node: dict,
                                     beam_nodes: Dict[int, dict]) -> dict:
        scope = "%s.cell.%d" % (ping, cell)
        nid = node_id("velocity_instrument", scope)
        if cell_node["status"] == STATUS_EXCLUDED:
            return self._node(
                nid, STATUS_EXCLUDED,
                {"diagnostic": cell_node["payload"].get(
                    "depth_diagnostic", "cell_excluded"),
                 "cell_node": cell_node["node_id"]},
                scope=scope, reason="excluded_cell_not_transformed")
        radials = {}
        rad_diag = {}
        for b in range(1, self.ruleset.geometry.n_beams + 1):
            v, d = _qval(self._raw("%s.beam.%d.radial" % (scope, b)))
            radials[b] = v
            rad_diag[b] = d
        good = [b for b, n in beam_nodes.items() if n["status"] == "good"]
        suspect = [b for b, n in beam_nodes.items()
                   if n["status"] == "suspect"]
        bad = [b for b, n in beam_nodes.items() if n["status"] == "bad"]
        usable = sorted(good + suspect)
        candidates: List[dict] = []
        if len(usable) >= 4:
            v4, _res = self.ruleset.geometry.solve(
                {b: radials[b] for b in usable[:self.ruleset.geometry.n_beams]})
            ev = self.ruleset.geometry.error_velocity(
                [radials[b] if b in usable else 0.0
                 for b in range(1, self.ruleset.geometry.n_beams + 1)])
            candidates.append(_candidate(
                "cand-4beam", usable, v4, "instrument_xyz",
                ev, "least_squares_all_usable"))
            # every suspect beam also yields a legal 3-beam alternative;
            # these are kept, never averaged into the 4-beam value.
            for s in suspect:
                triad = [b for b in usable if b != s]
                vt, _ = self.ruleset.geometry.solve(
                    {b: radials[b] for b in triad})
                candidates.append(_candidate(
                    "cand-3beam-drop-%d" % s, triad, vt,
                    "instrument_xyz", None,
                    "exactly_determined_no_error_velocity"))
        elif len(usable) == 3:
            vt, _ = self.ruleset.geometry.solve(
                {b: radials[b] for b in usable})
            candidates.append(_candidate(
                "cand-3beam", usable, vt, "instrument_xyz",
                None, "exactly_determined_no_error_velocity"))
        elif len(usable) < 3:
            return self._node(
                nid, STATUS_BLOCKED,
                {"usable_beams": usable, "bad_beams": bad,
                 "diagnostic": "fewer_than_three_beams",
                 "radial_diagnostics": rad_diag},
                scope=scope, reason="underdetermined_%d_beams" % len(usable))

        status = STATUS_OK
        reason = "unique_4beam"
        if len(candidates) > 1:
            status = STATUS_MULTI
            reason = ("multiple_legal_candidates:" +
                      ",".join(c["candidate_id"] for c in candidates))
        chosen = self.decisions.get(scope)
        chosen_candidate = None
        if chosen:
            chosen_candidate = next(
                (c for c in candidates if c["candidate_id"] == chosen), None)
            if chosen_candidate is not None:
                status = STATUS_OK
                reason = "decision:%s" % chosen
        payload = {
            "candidates": candidates,
            "chosen_candidate_id": (chosen_candidate["candidate_id"]
                                    if chosen_candidate else None),
            "good_beams": good, "suspect_beams": suspect, "bad_beams": bad,
            "geometry_fingerprint": self.fp["geometry"],
            "rules_fingerprint": self.fp["rules"],
            "radial_diagnostics": rad_diag,
        }
        return self._node(nid, status, payload, scope=scope, reason=reason)

    def evaluate_ship_earth(self, ping: str, cell: int,
                            instrument_node: dict,
                            attitude_node: dict) -> Tuple[dict, dict]:
        scope = "%s.cell.%d" % (ping, cell)
        mount = self.ruleset.mounting
        decl = self.ruleset.declination
        R_is = instrument_to_ship(
            mount.get("heading_offset_deg", 0.0),
            mount.get("tilt_deg", 0.0),
            self.ruleset.mount_recipe)
        ship_id = node_id("velocity_ship", scope)
        earth_id = node_id("velocity_earth", scope)

        if instrument_node["status"] in (STATUS_BLOCKED, STATUS_EXCLUDED):
            blocked = self._node(
                ship_id, instrument_node["status"],
                {"diagnostic": "upstream:" + instrument_node["status"],
                 "upstream_node": instrument_node["node_id"],
                 "mounting_fingerprint": self.fp["mounting"]},
                scope=scope, reason="upstream_not_ok")
            earth = self._node(
                earth_id, instrument_node["status"],
                {"diagnostic": "upstream:" + instrument_node["status"],
                 "upstream_node": ship_id,
                 "declination_fingerprint": self.fp["declination"],
                 "rotation_fingerprint": self.fp["rotation_recipe"]},
                scope=scope, reason="upstream_not_ok")
            return blocked, earth

        cand_id = instrument_node["payload"].get("chosen_candidate_id")
        if instrument_node["status"] == STATUS_MULTI or cand_id is None:
            multi = self._node(
                ship_id, STATUS_MULTI,
                {"diagnostic": "instrument_candidate_not_decided",
                 "upstream_node": instrument_node["node_id"],
                 "candidate_ids": [c["candidate_id"] for c in
                                   instrument_node["payload"]["candidates"]],
                 "mounting_fingerprint": self.fp["mounting"]},
                scope=scope, reason="awaiting_candidate_decision")
            earth = self._node(
                earth_id, STATUS_BLOCKED,
                {"diagnostic": "blocked_by_multi_candidate",
                 "upstream_node": ship_id,
                 "declination_fingerprint": self.fp["declination"],
                 "rotation_fingerprint": self.fp["rotation_recipe"]},
                scope=scope, reason="no_unique_candidate")
            return multi, earth

        cand = next(c for c in instrument_node["payload"]["candidates"]
                    if c["candidate_id"] == cand_id)
        vi = np.array(cand["velocity"]["components"])
        vs = R_is @ vi
        ship = self._node(
            ship_id, STATUS_OK,
            {"velocity": _vec(vs, "m/s", "ship_sfu"),
             "from_candidate": cand_id,
             "mount": mount,
             "mounting_fingerprint": self.fp["mounting"]},
            scope=scope, reason="instrument_to_ship:%s"
            % self.ruleset.mount_recipe)

        if attitude_node["status"] != STATUS_OK:
            earth = self._node(
                earth_id, STATUS_BLOCKED,
                {"diagnostic": attitude_node["payload"].get(
                    "diagnostic", "attitude_unavailable"),
                 "attitude_node": attitude_node["node_id"],
                 "declination_fingerprint": self.fp["declination"],
                 "rotation_fingerprint": self.fp["rotation_recipe"]},
                scope=scope, reason="attitude_blocked")
            return ship, earth

        att = attitude_node["payload"]
        true_h = true_heading(att["heading_unwrapped_deg"] % 360.0,
                              decl.get("value_deg", 0.0),
                              decl.get("convention", "east_positive"))
        R_se = ship_to_earth(true_h, att["pitch_deg"], att["roll_deg"],
                             self.ruleset.rotation_recipe)
        ve = R_se @ vs
        earth = self._node(
            earth_id, STATUS_OK,
            {"velocity": _vec(ve, "m/s", "earth_enu"),
             "true_heading_deg": true_h,
             "magnetic_heading_deg": att["heading_raw_deg"],
             "declination": decl,
             "attitude_segment": att.get("segment"),
             "attitude_interpolated": att.get("interpolated", False),
             "transform_chain": [
                 "beam->instrument:%s" % cand_id,
                 "instrument->ship:%s" % self.ruleset.mount_recipe,
                 "ship->earth:%s" % self.ruleset.rotation_recipe],
             "declination_fingerprint": self.fp["declination"],
             "rotation_fingerprint": self.fp["rotation_recipe"],
             "geometry_fingerprint": self.fp["geometry"]},
            scope=scope, reason="earth_chain_ok")
        return ship, earth

    # ------------------------------------------------------- bottom track
    def evaluate_bottom_track(self, ping: str,
                              attitude_node: dict) -> List[dict]:
        nodes = []
        radials, beam_nodes = {}, {}
        for b in range(1, self.ruleset.geometry.n_beams + 1):
            v, d = _qval(self._raw("%s.bt.beam.%d.radial" % (ping, b)))
            corr, cd = _qval(self._raw("%s.bt.beam.%d.corr" % (ping, b)))
            amp, ad = _qval(self._raw("%s.bt.beam.%d.amp" % (ping, b)))
            bad_raw = self._raw("%s.bt.beam.%d.bad" % (ping, b))
            bad = (bad_raw.get("value", bad_raw.get("raw"))
                   if isinstance(bad_raw, dict) else bad_raw)
            radials[b] = v
            beam_nodes[b] = grade_beam(corr, amp,
                                       None if bad is None else bool(bad),
                                       self.ruleset.quality)
        usable = sorted(b for b, t in beam_nodes.items() if t != "bad"
                        and radials[b] is not None)
        bi_id = node_id("bt_instrument", ping)
        if len(usable) == 4:
            vi, ev = self.ruleset.geometry.solve(
                {b: radials[b] for b in usable})
            inst = self._node(
                bi_id, STATUS_OK,
                {"velocity": _vec(vi, "m/s", "instrument_xyz"),
                 "support_beams": usable, "error_velocity": ev,
                 "geometry_fingerprint": self.fp["geometry"]},
                scope=ping, reason="bt_4beam")
        elif len(usable) == 3:
            vi, _ = self.ruleset.geometry.solve(
                {b: radials[b] for b in usable})
            inst = self._node(
                bi_id, STATUS_OK,
                {"velocity": _vec(vi, "m/s", "instrument_xyz"),
                 "support_beams": usable, "error_velocity": None,
                 "note": "3-beam bottom track, no error velocity",
                 "geometry_fingerprint": self.fp["geometry"]},
                scope=ping, reason="bt_3beam")
        else:
            inst = self._node(
                bi_id, STATUS_BLOCKED,
                {"usable_beams": usable,
                 "diagnostic": "fewer_than_three_bt_beams"},
                scope=ping, reason="bt_underdetermined")
        nodes.append(inst)

        mount = self.ruleset.mounting
        R_is = instrument_to_ship(mount.get("heading_offset_deg", 0.0),
                                  mount.get("tilt_deg", 0.0),
                                  self.ruleset.mount_recipe)
        bs_id = node_id("bt_ship", ping)
        be_id = node_id("bt_earth", ping)
        if inst["status"] != STATUS_OK:
            nodes.append(self._node(
                bs_id, inst["status"], {"diagnostic": "upstream"},
                scope=ping, reason="bt_upstream"))
            nodes.append(self._node(
                be_id, inst["status"], {"diagnostic": "upstream"},
                scope=ping, reason="bt_upstream"))
            return nodes
        vs = R_is @ np.array(inst["payload"]["velocity"]["components"])
        nodes.append(self._node(
            bs_id, STATUS_OK,
            {"velocity": _vec(vs, "m/s", "ship_sfu"),
             "mounting_fingerprint": self.fp["mounting"]},
            scope=ping, reason="bt_instrument_to_ship"))
        if attitude_node["status"] != STATUS_OK:
            nodes.append(self._node(
                be_id, STATUS_BLOCKED,
                {"diagnostic": "attitude_unavailable",
                 "attitude_node": attitude_node["node_id"]},
                scope=ping, reason="bt_attitude_blocked"))
            return nodes
        att = attitude_node["payload"]
        decl = self.ruleset.declination
        true_h = true_heading(att["heading_unwrapped_deg"] % 360.0,
                              decl.get("value_deg", 0.0),
                              decl.get("convention", "east_positive"))
        R_se = ship_to_earth(true_h, att["pitch_deg"], att["roll_deg"],
                             self.ruleset.rotation_recipe)
        ve = R_se @ vs
        nodes.append(self._node(
            be_id, STATUS_OK,
            {"velocity": _vec(ve, "m/s", "earth_enu"),
             "true_heading_deg": true_h,
             "declination_fingerprint": self.fp["declination"],
             "rotation_fingerprint": self.fp["rotation_recipe"]},
            scope=ping, reason="bt_earth_chain_ok"))
        return nodes

    def evaluate_relative(self, ping: str, cell: int,
                          earth_node: dict, bt_node: Optional[dict]) -> dict:
        scope = "%s.cell.%d" % (ping, cell)
        nid = node_id("velocity_relative", scope)
        if earth_node["status"] != STATUS_OK:
            return self._node(
                nid, earth_node["status"],
                {"diagnostic": "water_earth_not_ok"}, scope=scope,
                reason="upstream_not_ok")
        if bt_node is None or bt_node["status"] != STATUS_OK:
            return self._node(
                nid, STATUS_BLOCKED,
                {"diagnostic": "bottom_track_unavailable"},
                scope=scope, reason="no_bottom_reference")
        vw = np.array(earth_node["payload"]["velocity"]["components"])
        vb = np.array(bt_node["payload"]["velocity"]["components"])
        return self._node(
            nid, STATUS_OK,
            {"velocity": _vec(vw - vb, "m/s", "earth_enu"),
             "water_node": earth_node["node_id"],
             "bottom_node": bt_node["node_id"]},
            scope=scope, reason="water_minus_bottom")

    # ------------------------------------------------------------ orchestration
    def _node(self, node_id_: str, status: str, payload: dict, scope: str,
              reason: str) -> dict:
        fp = fingerprint("node", {"id": node_id_, "status": status,
                                  "payload": payload})
        node = {"node_id": node_id_, "status": status,
                "fingerprint": fp, "payload": payload,
                "scope_key": scope, "reason": reason,
                "kind": node_id_.split(":")[0]}
        self._by_id[node_id_] = node
        return node

    # ------------------------------------------------------------ orchestration
    def evaluate(self) -> Tuple[List[dict], Dict[str, dict]]:
        """Evaluate the graph for one version.

        Each node is either recomputed (returned with a new revision) or
        carried over unchanged when no changed source or changed upstream
        node touches its dependency set.  `propagation` records, for every
        carried node, why it stayed valid; callers surface that as the
        "why still valid / why recompute" explanation.
        """
        self.propagation: List[dict] = []
        new_nodes: Dict[str, dict] = {}
        changed_downstream: set = set()

        def upstream_changed(nid: str, kind: str, scope: str) -> bool:
            deps = dag.node_dependencies(kind, scope)
            hit = None
            for pat in deps:
                if pat.startswith("node:"):
                    target = pat.split(":", 1)[1]
                    if target in new_nodes:
                        hit = pat
                        break
                else:
                    for ch in self.changed:
                        if dag.matches(pat, ch):
                            hit = pat
                            break
                if hit:
                    break
            if hit is None and nid in self.previous and not self.changed:
                pass
            return hit is not None

        def reuse_or(nid: str, kind: str, scope: str, compute):
            deps = dag.node_dependencies(kind, scope)
            recompute = True
            if self.changed and nid in self.previous:
                touched = False
                for pat in deps:
                    if pat.startswith("node:"):
                        target = pat.split(":", 1)[1]
                        if target in new_nodes:
                            touched = True
                            break
                    else:
                        if any(dag.matches(pat, ch) for ch in self.changed):
                            touched = True
                            break
                if not touched:
                    recompute = False
            if not recompute:
                prev = self.previous[nid]
                self.propagation.append({
                    "node_id": nid, "action": "kept",
                    "previous_fingerprint": prev.get("fingerprint"),
                    "because": "no changed source/upstream matches deps",
                    "dependencies": deps})
                self._by_id[nid] = prev
                new_nodes[nid] = prev
                return prev
            node = compute()
            new_nodes[node["node_id"]] = node
            return node

        attitudes = {}
        # attitude is a family: recompute all attitude nodes if any attitude
        # source or the rules change (segments can shift globally).
        att_touched = any(
            ch.startswith("attitude.") or ch == "rules" for ch in self.changed)
        if not self.changed or att_touched or not self.previous:
            attitudes = self.evaluate_attitude()
            for n in attitudes.values():
                new_nodes[n["node_id"]] = n
        else:
            for nid, prev in self.previous.items():
                if nid.startswith("attitude:"):
                    self.propagation.append({
                        "node_id": nid, "action": "kept",
                        "previous_fingerprint": prev.get("fingerprint"),
                        "because": "attitude inputs unchanged",
                        "dependencies": dag.node_dependencies(
                            "attitude", prev["scope_key"])})
                    self._by_id[nid] = prev
                    new_nodes[nid] = prev
            attitudes = {nid: n for nid, n in new_nodes.items()
                         if nid.startswith("attitude:")}

        for sc in self.scopes:
            ping = sc["ping"]
            att_nid = node_id("attitude", ping)
            att = new_nodes.get(att_nid) or attitudes.get(att_nid)
            if att is None:
                # attitude may be entirely absent in a propagated run;
                # recover from previous
                att = self.previous.get(att_nid)
            # bottom track
            def _bt():
                ns = self.evaluate_bottom_track(ping, att)
                for n in ns:
                    new_nodes[n["node_id"]] = n
                return ns
            if not self.changed or any(
                    ch.startswith("%s.bt." % ping) or ch in
                    ("geometry", "rules", "mounting", "declination")
                    or (ch.startswith("attitude.")) for ch in self.changed):
                bt_list = self.evaluate_bottom_track(ping, att)
                for n in bt_list:
                    new_nodes[n["node_id"]] = n
            else:
                for kind in ("bt_instrument", "bt_ship", "bt_earth"):
                    nid = node_id(kind, ping)
                    if nid in self.previous:
                        prev = self.previous[nid]
                        self.propagation.append({
                            "node_id": nid, "action": "kept",
                            "previous_fingerprint": prev.get("fingerprint"),
                            "because": "bt inputs unchanged",
                            "dependencies": dag.node_dependencies(kind, ping)})
                        self._by_id[nid] = prev
                        new_nodes[nid] = prev
            bt_earth = new_nodes.get(node_id("bt_earth", ping))

            for cell in sc["cells"]:
                scope = "%s.cell.%d" % (ping, cell)
                cell_nid = node_id("cell_validity", scope)
                cell_node = reuse_or(
                    cell_nid, "cell_validity", scope,
                    lambda: self.evaluate_cell_and_beams(ping, cell)[0])
                # beam quality nodes
                for b in range(1, self.ruleset.geometry.n_beams + 1):
                    bnid = node_id("beam_quality", scope, str(b))
                    bscope = "%s.beam.%d" % (scope, b)
                    if bnid in new_nodes:
                        continue
                    reuse_or(
                        bnid, "beam_quality", bscope,
                        lambda b=b: self.evaluate_cell_and_beams(ping, cell)[1][b])
                beam_nodes = {b: new_nodes[node_id("beam_quality", scope,
                                                   str(b))]
                              for b in range(1, self.ruleset.geometry.n_beams + 1)}
                vi_nid = node_id("velocity_instrument", scope)
                vi = reuse_or(
                    vi_nid, "velocity_instrument", scope,
                    lambda: self.evaluate_instrument_velocity(
                        ping, cell, cell_node, beam_nodes))
                vs_nid = node_id("velocity_ship", scope)
                vs = reuse_or(
                    vs_nid, "velocity_ship", scope,
                    lambda: self.evaluate_ship_earth(ping, cell, vi, att)[0])
                ve_nid = node_id("velocity_earth", scope)
                ve = reuse_or(
                    ve_nid, "velocity_earth", scope,
                    lambda: self.evaluate_ship_earth(ping, cell, vi, att)[1])
                vr_nid = node_id("velocity_relative", scope)
                reuse_or(
                    vr_nid, "velocity_relative", scope,
                    lambda: self.evaluate_relative(ping, cell, ve, bt_earth))

        nodes = list(new_nodes.values())
        for n in nodes:
            if "prev_rev" not in n:
                prev = self.previous.get(n["node_id"])
                n["prev_rev"] = prev["rev"] if prev else None
        self.nodes = nodes
        return nodes, attitudes

    def propagation_report(self) -> List[dict]:
        return getattr(self, "propagation", [])


def evaluate_version(sources, previous=None, decisions=None, changed=None):
    ev = Evaluator(sources, previous, decisions, changed)
    nodes, _att = ev.evaluate()
    return nodes, ev.propagation_report()
