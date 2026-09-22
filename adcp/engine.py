"""Derivation engine: beam -> instrument -> ship -> earth, per version.

The computer rebuilds a DependencyGraph for a frozen set of source rows and
configuration versions. Fingerprints bind raw representations (so a purely
cosmetic representation change still counts), geometry, declination version,
mount version and the chosen transform order. Three- and four-beam candidates
are always separate nodes with separate stable identities.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .fingerprint import fingerprint, RULE_VERSION
from .geometry import (
    Geometry, ORDER_HPR, attitude_rotation, mount_rotation,
)
from .graph import DependencyGraph, Node
from .quality import QualityPolicy, judge_beam
from .quantity import Quantity
from .attitude import interpolate_attitude


def qjson(q: Optional[Quantity]) -> Dict[str, Any]:
    return None if q is None else q.to_json()


def raw_key(q: Optional[Quantity]) -> Any:
    if q is None:
        return None
    return {"raw": q.raw, "unit": q.unit, "datum": q.datum}


class Computer:
    def __init__(self, bundle: Dict[str, Any]):
        self.case_id = bundle["case_id"]
        self.pings = bundle["pings"]
        self.attitude = bundle.get("attitude", [])
        self.geom = Geometry.from_json(bundle["geometry"])
        self.mount = bundle["mount"]
        self.declination = bundle["declination"]
        self.policy = QualityPolicy(**bundle["policy"])
        self.transform_order = bundle.get("transform_order", ORDER_HPR)
        self.gap_factor = float(bundle.get("gap_factor", 2.5))
        self.nb = len(self.pings)
        self.nbins = max((len(p["cells"]) for p in self.pings), default=0)
        self._interp = None
        self.graph = DependencyGraph()

    # ---- input fingerprints -------------------------------------------------
    def _input_snapshot(self) -> Dict[str, Any]:
        return {
            "pings": self.pings,
            "attitude": self.attitude,
            "geometry": self.geom.to_json(),
            "mount": self.mount,
            "declination": self.declination,
            "policy": self.policy.to_json(),
            "transform_order": self.transform_order,
            "gap_factor": self.gap_factor,
        }

    def attitude_for(self) -> List[Dict[str, Any]]:
        if self._interp is None:
            st = [s["time"] for s in self.attitude]
            hd = [s["heading_q"]["raw"] for s in self.attitude]
            pt = [s["pitch_q"]["raw"] for s in self.attitude]
            rl = [s["roll_q"]["raw"] for s in self.attitude]
            self._interp = interpolate_attitude(
                [p["time"] for p in self.pings], st, hd, pt, rl, self.gap_factor)
        return self._interp

    # ---- node id helpers ----------------------------------------------------
    def _src(self, kind: str, *parts: Any) -> str:
        return "src:%s:%s:%s" % (self.case_id, kind, ":".join(map(str, parts)))

    def _der(self, kind: str, *parts: Any) -> str:
        return "der:%s:%s:%s" % (self.case_id, kind, ":".join(map(str, parts)))

    def _cell_quantity(self, ping: int, cell: int, beam: int, name: str):
        return Quantity.from_json(self.pings[ping]["cells"][cell][name][beam])

    # ---- graph construction -------------------------------------------------
    def build(self) -> DependencyGraph:
        g = self.graph
        # Configuration source nodes (each separately versioned & fingerprinted).
        geo_payload = self.geom.to_json()
        g.add_source(self._src("geometry", self.geom.geometry_version),
                     "geometry", geo_payload,
                     fingerprint("src", {"geometry": geo_payload}), RULE_VERSION)
        g.add_source(self._src("mount", self.mount["mount_version"]),
                     "mount", self.mount,
                     fingerprint("src", {"mount": self.mount}), RULE_VERSION)
        g.add_source(self._src("decl", self.declination["declination_version"]),
                     "declination", self.declination,
                     fingerprint("src", {"declination": self.declination}), RULE_VERSION)
        g.add_source(self._src("qpol", self.policy.policy_version),
                     "quality-policy", self.policy.to_json(),
                     fingerprint("src", {"policy": self.policy.to_json()}), RULE_VERSION)
        # Attitude samples (separate records).
        for j, sample in enumerate(self.attitude):
            payload = {
                "time": sample["time"],
                "heading": sample["heading_q"], "pitch": sample["pitch_q"],
                "roll": sample["roll_q"],
            }
            g.add_source(self._src("att", j), "attitude-sample", payload,
                         fingerprint("src", {"attitude": payload}), RULE_VERSION)

        attitude = self.attitude_for()
        for pi, ping in enumerate(self.pings):
            self._build_ping(pi, attitude[pi])
        return g

    def _build_ping(self, pi: int, att: Dict[str, Any]) -> None:
        g = self.graph
        geo_id = self._src("geometry", self.geom.geometry_version)
        qpol_id = self._src("qpol", self.policy.policy_version)
        mt_id = self._src("mount", self.mount["mount_version"])
        dc_id = self._src("decl", self.declination["declination_version"])

        # Beam source nodes (velocity / correlation / amplitude / bad flag kept apart).
        for ci in range(self.nbins):
            for bi in range(4):
                cell = self.pings[pi]["cells"][ci]
                vel = Quantity.from_json(cell["velocity_q"][bi])
                cor = Quantity.from_json(cell["correlation_q"][bi])
                amp = Quantity.from_json(cell["amplitude_q"][bi])
                bad = bool(cell["bad_beam"][bi])
                for kind, payload in (
                    ("beamvel", {"q": qjson(vel)}),
                    ("beamcor", {"q": qjson(cor)}),
                    ("beamamp", {"q": qjson(amp)}),
                    ("beambad", {"bad": bad}),
                ):
                    g.add_source(
                        self._src(kind, pi, ci, bi), kind, payload,
                        fingerprint("src", payload), RULE_VERSION)

        # Derived attitude-at-ping (explicitly does not cross gaps).
        att_inputs = {"samples": [s for s in self.attitude], "ping_time": self.pings[pi]["time"]}
        att_fp = fingerprint("attitude-at-ping", att_inputs)
        att_id = self._der("att", pi)
        g.upsert_derived(att_id, "attitude-at-ping",
                         [self._src("att", j) for j in range(len(self.attitude))],
                         att, att_fp, RULE_VERSION)

        bottom = self.pings[pi].get("bottom_track")
        bottom_depth_si = None
        if bottom is not None:
            bdepth = Quantity.from_json(bottom["depth_q"])
            bottom_depth_si = bdepth.si_value

        mount_mat = mount_rotation(
            float(self.mount["heading_offset_rad"]),
            float(self.mount["pitch_offset_rad"]),
            float(self.mount["roll_offset_rad"]),
        )
        att_mat = None
        if att.get("available"):
            heading_mag = math.radians(att["heading_deg"])
            heading_true = heading_mag + float(self.declination["angle_rad"])
            att_mat = attitude_rotation(
                heading_true, math.radians(att["pitch_deg"]),
                math.radians(att["roll_deg"]), self.transform_order)

        for ci in range(self.nbins):
            self._build_cell(pi, ci, geo_id, qpol_id, mt_id, dc_id, att_id,
                             mount_mat, att_mat, bottom_depth_si, att)

        if bottom is not None:
            self._build_bottom(pi, geo_id, qpol_id, mt_id, dc_id, att_id,
                               mount_mat, att_mat, att, bottom)

    def _build_cell(self, pi, ci, geo_id, qpol_id, mt_id, dc_id, att_id,
                    mount_mat, att_mat, bottom_depth_si, att) -> None:
        g = self.graph
        cell = self.pings[pi]["cells"][ci]
        depth_q = Quantity.from_json(cell["depth_q"])
        depth_si = depth_q.si_value

        # Invalid depth cell: kept in place with a diagnostic, never plotted away.
        depth_diag = None if depth_q.convertible else depth_q.diagnostic
        below_bottom = (bottom_depth_si is not None and depth_si is not None
                        and depth_si > bottom_depth_si)
        cell_excluded = depth_diag is not None or below_bottom

        # Beam verdicts -> instrument candidates (4-beam and every 3-beam).
        verdicts, velocities = [], []
        for bi in range(4):
            vel = self._cell_quantity(pi, ci, bi, "velocity_q")
            cor = self._cell_quantity(pi, ci, bi, "correlation_q")
            amp = self._cell_quantity(pi, ci, bi, "amplitude_q")
            bad = bool(cell["bad_beam"][bi])
            verdicts.append(judge_beam(vel, cor, amp, bad, self.policy))
            velocities.append(vel.si_value)

        from .quality import evaluate_cell
        solution = evaluate_cell(verdicts, velocities, self.geom, self.policy)
        inst_deps = ([geo_id, qpol_id] +
                     [self._src("beamvel", pi, ci, b) for b in range(4)] +
                     [self._src("beamcor", pi, ci, b) for b in range(4)] +
                     [self._src("beamamp", pi, ci, b) for b in range(4)] +
                     [self._src("beambad", pi, ci, b) for b in range(4)])

        candidates = []
        if solution["four"] is not None:
            candidates.append(("four", solution["four"]))
        for triple in solution["triples"]:
            candidates.append(("tri%d" % triple["missing"], triple))

        for label, cand in candidates:
            inst_payload = {
                "candidate": cand, "verdicts": verdicts,
                "excluded_beams": solution["excluded_beams"],
                "cell_status": solution["status"],
                "depth": qjson(depth_q),
                "excluded": cell_excluded,
                "depth_diagnostic": depth_diag,
                "below_bottom": below_bottom,
            }
            fp_inputs = {
                "beams": [raw_key(self._cell_quantity(pi, ci, b, "velocity_q"))
                          for b in range(4)],
                "correlations": [raw_key(self._cell_quantity(pi, ci, b, "correlation_q"))
                                 for b in range(4)],
                "amplitudes": [raw_key(self._cell_quantity(pi, ci, b, "amplitude_q"))
                               for b in range(4)],
                "bad": list(cell["bad_beam"]),
                "geometry": self.geom.to_json(),
                "policy": self.policy.to_json(),
            }
            inst_id = self._der("inst", pi, ci, label)
            g.upsert_derived(inst_id, "instrument-velocity", inst_deps,
                             inst_payload,
                             fingerprint("instrument", fp_inputs), RULE_VERSION)
            self._build_transforms(pi, ci, label, inst_id, cand, mt_id, dc_id,
                                   att_id, mount_mat, att_mat, att,
                                   cell_excluded, depth_q, depth_diag, below_bottom)

        # Cell-level evidence node (exclusion reasons + quality), always present.
        evidence_id = self._der("evidence", pi, ci)
        g.upsert_derived(
            evidence_id, "cell-evidence",
            inst_deps,
            {"verdicts": verdicts, "excluded_beams": solution["excluded_beams"],
             "depth": qjson(depth_q), "cell_status": solution["status"],
             "excluded": cell_excluded, "depth_diagnostic": depth_diag,
             "below_bottom": below_bottom,
             "bottom_depth_si": bottom_depth_si},
            fingerprint("evidence", {"depth": raw_key(depth_q),
                                     "bad": list(cell["bad_beam"]),
                                     "bottom_depth_si": bottom_depth_si}),
            RULE_VERSION)

    def _build_transforms(self, pi, ci, label, inst_id, cand, mt_id, dc_id,
                          att_id, mount_mat, att_mat, att,
                          cell_excluded, depth_q, depth_diag, below_bottom):
        g = self.graph
        vel = np.array(cand["velocity"], dtype=float)
        ship_vec = mount_mat @ vel
        ship_payload = {
            "kind": cand["kind"], "beams": list(cand["beams"]),
            "velocity": [float(x) for x in ship_vec],
            "frame": "SHIP", "unit": "m/s", "label": label,
            "mount_version": self.mount["mount_version"],
            "excluded": cell_excluded, "depth": qjson(depth_q),
        }
        ship_id = self._der("ship", pi, ci, label)
        g.upsert_derived(
            ship_id, "ship-velocity", [inst_id, mt_id], ship_payload,
            fingerprint("ship", {
                "inst_fingerprint": g.nodes[inst_id].fingerprint,
                "mount": self.mount,
            }), RULE_VERSION)

        earth_payload = {
            "kind": cand["kind"], "beams": list(cand["beams"]),
            "label": label, "excluded": cell_excluded, "depth": qjson(depth_q),
            "declination_version": self.declination["declination_version"],
            "transform_order": self.transform_order,
            "attitude_available": bool(att.get("available")),
        }
        earth_deps = [ship_id, dc_id, att_id]
        earth_fp = fingerprint("earth", {
            "ship_fingerprint": g.nodes[ship_id].fingerprint,
            "declination": self.declination,
            "attitude": att,
            "order": self.transform_order,
        })
        if att_mat is None:
            earth_payload.update({
                "frame": "NED", "unit": "m/s", "velocity": None,
                "status": "missing-attitude",
                "diagnostic": att.get("diagnostic", "attitude-unavailable"),
            })
        else:
            earth_vec = att_mat @ ship_vec
            earth_payload.update({
                "frame": "NED", "unit": "m/s",
                "velocity": [float(x) for x in earth_vec],
                "status": "ok",
            })
        earth_id = self._der("earth", pi, ci, label)
        g.upsert_derived(earth_id, "earth-velocity", earth_deps,
                         earth_payload, earth_fp, RULE_VERSION)

    def _build_bottom(self, pi, geo_id, qpol_id, mt_id, dc_id, att_id,
                      mount_mat, att_mat, att, bottom) -> None:
        g = self.graph
        from .quality import evaluate_cell
        verdicts, velocities = [], []
        for bi in range(4):
            vel = Quantity.from_json(bottom["velocity_q"][bi])
            cor = Quantity.from_json(bottom["correlation_q"][bi])
            amp = Quantity.from_json(bottom["amplitude_q"][bi])
            bad = bool(bottom["bad_beam"][bi])
            verdicts.append(judge_beam(vel, cor, amp, bad, self.policy))
            velocities.append(vel.si_value)
        solution = evaluate_cell(verdicts, velocities, self.geom, self.policy)
        cands = []
        if solution["four"] is not None:
            cands.append(("four", solution["four"]))
        for triple in solution["triples"]:
            cands.append(("tri%d" % triple["missing"], triple))
        deps = ([geo_id, qpol_id] +
                [self._src("btvel", pi, b) for b in range(4)] +
                [self._src("btcor", pi, b) for b in range(4)] +
                [self._src("btamp", pi, b) for b in range(4)] +
                [self._src("btbad", pi, b) for b in range(4)])
        for bi in range(4):
            for kind, key in (("btvel", "velocity_q"), ("btcor", "correlation_q"),
                              ("btamp", "amplitude_q")):
                payload = {"q": Quantity.from_json(bottom[key][bi]).to_json()}
                g.add_source(self._src(kind, pi, bi), kind, payload,
                             fingerprint("src", payload), RULE_VERSION)
            g.add_source(self._src("btbad", pi, bi), "btbad",
                         {"bad": bool(bottom["bad_beam"][bi])},
                         fingerprint("src", {"bad": bool(bottom["bad_beam"][bi])}),
                         RULE_VERSION)
        bdepth = Quantity.from_json(bottom["depth_q"])
        g.upsert_derived(
            self._der("bt", pi), "bottom-track",
            [self._src("btvel", pi, b) for b in range(4)] + [geo_id],
            {"depth": qjson(bdepth)},
            fingerprint("bottom-track", {"depth_raw": raw_key(bdepth)}),
            RULE_VERSION)
        fp_inputs = {
            "beams": [raw_key(Quantity.from_json(bottom["velocity_q"][b])) for b in range(4)],
            "correlations": [raw_key(Quantity.from_json(bottom["correlation_q"][b])) for b in range(4)],
            "amplitudes": [raw_key(Quantity.from_json(bottom["amplitude_q"][b])) for b in range(4)],
            "bad": list(bottom["bad_beam"]),
            "geometry": self.geom.to_json(),
            "policy": self.policy.to_json(),
            "bottom_track": True,
        }
        for label, cand in cands:
            inst_id = self._der("btinst", pi, label)
            g.upsert_derived(inst_id, "instrument-bottom-track", deps,
                             {"candidate": cand, "verdicts": verdicts,
                              "excluded_beams": solution["excluded_beams"],
                              "cell_status": solution["status"]},
                             fingerprint("instrument", fp_inputs), RULE_VERSION)
            ship_vec = mount_mat @ np.array(cand["velocity"], dtype=float)
            ship_id = self._der("btship", pi, label)
            g.upsert_derived(ship_id, "ship-bottom-track", [inst_id, mt_id],
                             {"kind": cand["kind"], "beams": list(cand["beams"]),
                              "velocity": [float(x) for x in ship_vec],
                              "frame": "SHIP", "unit": "m/s", "label": label},
                             fingerprint("ship", {
                                 "inst_fingerprint": g.nodes[inst_id].fingerprint,
                                 "mount": self.mount}), RULE_VERSION)
            earth_id = self._der("btearth", pi, label)
            payload = {"kind": cand["kind"], "beams": list(cand["beams"]),
                       "label": label, "frame": "NED", "unit": "m/s",
                       "declination_version": self.declination["declination_version"],
                       "transform_order": self.transform_order,
                       "attitude_available": bool(att.get("available"))}
            if att_mat is None:
                payload.update({"velocity": None, "status": "missing-attitude",
                                "diagnostic": att.get("diagnostic", "attitude-unavailable")})
            else:
                ev = att_mat @ ship_vec
                payload.update({"velocity": [float(x) for x in ev], "status": "ok"})
            g.upsert_derived(earth_id, "earth-bottom-track",
                             [ship_id, dc_id, att_id], payload,
                             fingerprint("earth", {
                                 "ship_fingerprint": g.nodes[ship_id].fingerprint,
                                 "declination": self.declination,
                                 "attitude": att, "order": self.transform_order}),
                             RULE_VERSION)


def build_computer(bundle: Dict[str, Any]) -> Computer:
    return Computer(bundle)
