"""Derivation engine: beam -> instrument -> ship -> earth (and bottom track).

The engine runs on a dependency graph and propagates only what changed.  It is
deliberately honest about uncertainty:

* 3-beam and 4-beam solutions are distinct node kinds with distinct slot
  identities and distinct content ids - they are never merged into one value;
* all legal 3-beam candidates are retained (exact inverse + zero error
  velocity);
* attitude is interpolated per ping but never across a gap;
* unconvertible raw values block the node with a diagnostic instead of being
  dropped;
* transform order, beam geometry and declination version enter the fingerprint.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import geometry as geom_mod
from . import transforms as tr
from . import units
from .dag import DependencyGraph, fingerprint, val_id

RULEBOOK_VERSION = "rb-2026.1"

SOURCE_KINDS = (
    "beam_velocity", "correlation", "amplitude", "bad_beam",
    "heading", "pitch", "roll", "declination", "mount", "bottom_track",
    "ping_meta", "cell_depth",
)


@dataclass
class PingInput:
    ping_id: str
    time_s: float
    cell_depth_m: List[float]
    beam_velocity: List[List[float]]      # [cell][beam] m/s
    beam_velocity_unit: str
    correlation: List[List[float]]        # percent
    amplitude: List[List[float]]          # dB
    bad_beam: List[List[bool]]            # explicit bad-beam flags
    bottom_track_range_m: Optional[List[float]] = None  # per beam, may be None
    bottom_track_velocity: Optional[List[float]] = None  # b1..b4
    bottom_available: bool = False


@dataclass
class Rules:
    geom: geom_mod.BeamGeometry
    geom_id: str
    mount: tr.MountAngles
    mount_id: str
    declination_deg: float
    declination_id: str
    transform_order: str
    quality_policy_id: str
    max_attitude_gap_s: float = 4.0
    correlation_min: float = 80.0
    exclude_below_bottom: bool = True

    def fp(self) -> str:
        return fingerprint(
            RULEBOOK_VERSION, self.transform_order, self.geom_id,
            self.declination_id, self.mount_id, self.quality_policy_id,
        )


def _norm(raw: float, raw_unit: str, canonical: str):
    return units.normalize(raw, raw_unit, canonical)


def build_rules(rule_map: Dict[str, dict]) -> Rules:
    """Assemble a Rules object from active store rules (with defaults)."""
    cfg = lambda t, k, d: (rule_map[t]["config"][k] if t in rule_map
                           and k in rule_map[t]["config"] else d)
    geom_cfg = rule_map.get("geometry", {}).get("config", {})
    g = geom_mod.BeamGeometry(
        angle_deg=geom_cfg.get("angle_deg", 20.0),
        azimuths_deg=tuple(geom_cfg.get("azimuths_deg",
                                        geom_mod.DEFAULT_AZIMUTHS)),
    )
    mount_cfg = rule_map.get("mount", {}).get("config", {})
    mount = tr.MountAngles(
        mount_cfg.get("pitch_deg", 0.0),
        mount_cfg.get("roll_deg", 0.0),
        mount_cfg.get("heading_deg", 0.0),
    )
    qp = rule_map.get("quality_policy", {}).get("config", {})
    dec_cfg = rule_map.get("declination", {}).get("config", {})
    return Rules(
        geom=g,
        geom_id=geom_cfg.get("config_id") or rule_map.get("geometry", {}).get("id") or g.config_id(),
        mount=mount,
        mount_id=mount_cfg.get("config_id") or rule_map.get("mount", {}).get("id") or mount.config_id(),
        declination_deg=dec_cfg.get("declination_deg", 0.0),
        declination_id=dec_cfg.get("model") or rule_map.get("declination", {}).get("id") or "decl-default",
        transform_order=cfg("transform_order", "order", tr.CANONICAL_ORDER),
        quality_policy_id=rule_map.get("quality_policy", {}).get("id", "qp-default"),
        max_attitude_gap_s=qp.get("max_attitude_gap_s", 4.0),
        correlation_min=qp.get("correlation_min", 80.0),
        exclude_below_bottom=qp.get("exclude_below_bottom", True),
    )

class Engine:
    """Computes run nodes from parsed ping inputs + attitude samples."""

    def __init__(self, pings: Sequence[PingInput],
                 attitude: Sequence[tr.AttitudeSample], rules: Rules,
                 source_refs: Optional[dict] = None) -> None:
        self.pings = list(pings)
        self.attitude = list(attitude)
        self.rules = rules
        self.source_refs = source_refs or {}
        self.graph = DependencyGraph()
        self.nodes: Dict[str, dict] = {}

    # ------------------------------------------------------------------ slots
    @staticmethod
    def s_inst(p: str, c: int) -> str:
        return "instrument/%s/c%02d" % (p, c)

    @staticmethod
    def s_three(p: str, c: int, cand: str) -> str:
        return "three_beam/%s/c%02d/%s" % (p, c, cand)

    @staticmethod
    def s_ship(p: str, c: int) -> str:
        return "ship/%s/c%02d" % (p, c)

    @staticmethod
    def s_earth(p: str, c: int) -> str:
        return "earth/%s/c%02d" % (p, c)

    @staticmethod
    def s_cell(p: str, c: int) -> str:
        return "cell_exclusion/%s/c%02d" % (p, c)

    @staticmethod
    def s_bottom(p: str) -> str:
        return "earth_bottom/%s" % p

    # ------------------------------------------------------------- compute
    def run(self) -> Dict[str, dict]:
        r = self.rules
        for ping in self.pings:
            att = tr.interpolate_attitude(
                self.attitude, ping.ping_id, ping.time_s, r.max_attitude_gap_s
            )
            bt_range = self._bottom_range(ping)
            for ci, depth in enumerate(ping.cell_depth_m):
                self._compute_cell(ping, ci, float(depth), att, bt_range)
            self._compute_bottom(ping, att)
        return self.nodes

    def _emit(self, slot: str, kind: str, deps: List[str], status: str,
              value=None, basis=None, unit=None, candidates=None,
              diagnostics=None, source_refs=None) -> dict:
        if slot in self.graph.nodes:
            raise ValueError("slot already computed: %s" % slot)
        self.graph.add(slot, kind, deps)
        refs = dict(self.source_refs)
        if source_refs:
            refs.update(source_refs)
        prov = {"fp": self.rules.fp(), "deps": deps, "refs": refs}
        node = {
            "slot": slot, "kind": kind, "status": status, "deps": deps,
            "value": value, "basis": basis, "unit": unit,
            "val_id": val_id(basis or kind, value if value is not None
                             else {"status": status,
                                   "diagnostics": diagnostics or []}, prov),
            "candidates": candidates,
            "diagnostics": diagnostics or [],
            "source_refs": refs,
        }
        self.nodes[slot] = node
        return node

    def _cell_bad_beams(self, ping: PingInput, ci: int) -> List[bool]:
        r = self.rules
        bad = [bool(x) for x in ping.bad_beam[ci]]
        # low correlation independently flags a beam bad for this cell
        for b, corr in enumerate(ping.correlation[ci]):
            if corr is not None and corr < r.correlation_min:
                bad[b] = True
        return bad

    def _compute_cell(self, ping: PingInput, ci: int, depth_m: float,
                      att: tr.AttitudeAtPing, bt_range: Optional[float]) -> None:
        r = self.rules
        p = ping.ping_id
        raws = ping.beam_velocity[ci]
        norm = [_norm(v, ping.beam_velocity_unit, "m/s") for v in raws]
        bvec = np.array(
            [n.normalized.value if n.normalized else np.nan for n in norm],
            dtype=float,
        )
        explicit_bad = self._cell_bad_beams(ping, ci)
        for b, n in enumerate(norm):
            if n.normalized is None:
                explicit_bad[b] = True
        n_valid = geom_mod.count_valid(bvec, [not x for x in explicit_bad])

        refs_cell = {"ping": p, "cell": ci,
                     "sources": self.source_refs.get("by_ping", {}).get(p, [])}

        # invalid depth cell evidence (below bottom track / implausible depth)
        exclusion_diags: List[str] = []
        if not math.isfinite(depth_m) or depth_m <= 0:
            exclusion_diags.append("invalid depth cell: depth=%r" % depth_m)
        if (r.exclude_below_bottom and bt_range is not None
                and math.isfinite(depth_m) and depth_m > bt_range):
            exclusion_diags.append(
                "cell depth %.2fm below bottom track %.2fm" % (depth_m, bt_range))
        self._emit(
            self.s_cell(p, ci), "cell_exclusion", [],
            "excluded" if exclusion_diags else "included",
            value={"depth_m": depth_m, "bottom_range_m": bt_range},
            basis="cell", unit="m", diagnostics=exclusion_diags,
            source_refs=refs_cell,
        )

        inst_slot = self.s_inst(p, ci)
        three_slots: List[str] = []
        if exclusion_diags:
            self._emit(inst_slot, "instrument", [], "blocked",
                       diagnostics=["cell excluded; velocity not derived"],
                       source_refs=refs_cell)
            for cand in ("exact_inverse", "zero_error_velocity"):
                three_slots.append(self.s_three(p, ci, cand))
                self._emit(three_slots[-1], "three_beam", [], "blocked",
                           diagnostics=["cell excluded"], source_refs=refs_cell)
            self._frame_chain_blocked(p, ci, inst_slot, three_slots, att, refs_cell)
            return

        if n_valid == 4:
            sol = geom_mod.solve_four(r.geom, bvec, [not x for x in explicit_bad])
            node = self._emit(
                inst_slot, "instrument", [], "ok",
                value=sol["v"], basis="instrument", unit="m/s",
                diagnostics=[] if abs(sol["error_velocity"]) < 0.2
                else ["high error velocity %.3f m/s" % sol["error_velocity"]],
                source_refs={**refs_cell, "method": "four_beam_lsq",
                             "error_velocity": sol["error_velocity"],
                             "used_beams": sol["used_beams"]},
            )
            node["value"] = sol["v"]
            for cand in ("exact_inverse", "zero_error_velocity"):
                three_slots.append(self.s_three(p, ci, cand))
                self._emit(three_slots[-1], "three_beam", [], "not_applicable",
                           diagnostics=["four beams valid; 3-beam candidate not used"],
                           source_refs=refs_cell)
            self._frame_chain(p, ci, sol["v"], inst_slot, three_slots, att, refs_cell)
        elif n_valid == 3:
            # Instrument slot stays indeterminate: record status, keep BOTH
            # legal 3-beam solutions as separate candidate nodes.
            cands = geom_mod.solve_three_candidates(
                r.geom, bvec, [not x for x in explicit_bad])
            self._emit(
                inst_slot, "instrument", [], "indeterminate",
                diagnostics=["%d valid beams; no unique instrument solution"
                             % n_valid],
                source_refs={**refs_cell, "bad_beams": explicit_bad},
            )
            for cand in cands:
                slot = self.s_three(p, ci, cand["assumption"])
                three_slots.append(slot)
                self._emit(
                    slot, "three_beam", [inst_slot], "candidate",
                    value=cand["v"], basis="instrument", unit="m/s",
                    candidates=[cand],
                    diagnostics=["3-beam candidate under assumption %r"
                                 % cand["assumption"]],
                    source_refs={**refs_cell, "assumption": cand["assumption"],
                                 "missing_beam": cand["missing_beam"],
                                 "used_beams": cand["used_beams"]},
                )
            self._frame_chain_blocked(p, ci, inst_slot, three_slots, att,
                                      refs_cell,
                                      reason="no unique 3-beam choice; both "
                                             "candidates retained, ship/earth "
                                             "frame not promoted")
        else:
            self._emit(
                inst_slot, "instrument", [], "blocked",
                diagnostics=["only %d valid beams" % n_valid],
                source_refs={**refs_cell, "bad_beams": explicit_bad},
            )
            for cand in ("exact_inverse", "zero_error_velocity"):
                three_slots.append(self.s_three(p, ci, cand))
                self._emit(three_slots[-1], "three_beam", [], "blocked",
                           diagnostics=["only %d valid beams" % n_valid],
                           source_refs=refs_cell)
            self._frame_chain_blocked(p, ci, inst_slot, three_slots, att, refs_cell)

    def _frame_chain(self, p: str, ci: int, v_inst: List[float],
                     inst_slot: str, three_slots: List[str],
                     att: tr.AttitudeAtPing, refs: dict) -> None:
        r = self.rules
        ship_slot = self.s_ship(p, ci)
        earth_slot = self.s_earth(p, ci)
        deps_ship = [inst_slot] + (att.bracket or [])
        if att.blocked:
            self._emit(ship_slot, "ship", deps_ship, "blocked",
                       diagnostics=[att.diagnostic or "attitude unavailable"],
                       source_refs={**refs, "attitude": att.to_dict()})
            self._emit(earth_slot, "earth", [ship_slot], "blocked",
                       diagnostics=["upstream ship velocity blocked"],
                       source_refs=refs)
            return
        v_ship = tr.ship_from_instrument(
            v_inst, att.pitch_deg, att.roll_deg, r.mount, r.transform_order)
        self._emit(
            ship_slot, "ship", deps_ship, "ok",
            value=[float(x) for x in v_ship], basis="ship", unit="m/s",
            source_refs={**refs, "attitude": att.to_dict(),
                         "mount_id": r.mount_id,
                         "transform_order": r.transform_order},
        )
        hdg = tr.true_heading(att.heading_mag_deg, r.declination_deg)
        v_earth = tr.earth_from_ship(v_ship, hdg)
        self._emit(
            earth_slot, "earth", [ship_slot], "ok",
            value=[float(x) for x in v_earth], basis="earth", unit="m/s",
            source_refs={**refs, "true_heading_deg": hdg,
                         "heading_mag_deg": att.heading_mag_deg,
                         "declination_deg": r.declination_deg,
                         "declination_id": r.declination_id},
        )

    def _frame_chain_blocked(self, p: str, ci: int, inst_slot: str,
                             three_slots: List[str], att: tr.AttitudeAtPing,
                             refs: dict, reason: str =
                             "upstream instrument velocity not uniquely defined"
                             ) -> None:
        ship_slot = self.s_ship(p, ci)
        earth_slot = self.s_earth(p, ci)
        self._emit(
            ship_slot, "ship", [inst_slot], "blocked",
            diagnostics=[reason] + ([att.diagnostic] if att.blocked
                                    and att.diagnostic else []),
            source_refs={**refs, "attitude": att.to_dict()},
        )
        self._emit(
            earth_slot, "earth", [ship_slot], "blocked",
            diagnostics=["upstream ship velocity blocked"], source_refs=refs,
        )

    # ---------------------------------------------------------- bottom track
    def _bottom_range(self, ping: PingInput) -> Optional[float]:
        if not ping.bottom_available or ping.bottom_track_range_m is None:
            return None
        vals = [x for x in ping.bottom_track_range_m
                if x is not None and math.isfinite(x)]
        return min(vals) if vals else None

    def _compute_bottom(self, ping: PingInput,
                        att: tr.AttitudeAtPing) -> None:
        r = self.rules
        p = ping.ping_id
        slot = self.s_bottom(p)
        bv = ping.bottom_track_velocity
        refs = {"ping": p, "sources": self.source_refs.get("by_ping", {}).get(p, [])}
        if not ping.bottom_available or bv is None:
            self._emit(slot, "earth_bottom", [], "unavailable",
                       diagnostics=["no bottom track for ping"], source_refs=refs)
            return
        mask = [x is not None and math.isfinite(x) for x in bv]
        sol = geom_mod.solve_four(r.geom, np.nan_to_num(np.asarray(bv, float),
                                                        nan=0.0), mask)
        if sol is None:
            self._emit(slot, "earth_bottom", [], "blocked",
                       diagnostics=["bottom track needs 4 beams"],
                       source_refs=refs)
            return
        if att.blocked:
            self._emit(slot, "earth_bottom", [], "blocked",
                       diagnostics=[att.diagnostic or "attitude gap"],
                       source_refs={**refs, "attitude": att.to_dict()})
            return
        v_ship = tr.ship_from_instrument(
            sol["v"], att.pitch_deg, att.roll_deg, r.mount, r.transform_order)
        hdg = tr.true_heading(att.heading_mag_deg, r.declination_deg)
        v_earth = tr.earth_from_ship(v_ship, hdg)
        self._emit(
            slot, "earth_bottom", [], "ok",
            value=[float(x) for x in v_earth], basis="earth", unit="m/s",
            diagnostics=[] if abs(sol["error_velocity"]) < 0.2
            else ["bottom-track error velocity %.3f" % sol["error_velocity"]],
            source_refs={**refs, "true_heading_deg": hdg,
                         "used_beams": sol["used_beams"],
                         "method": "bottom_track_four_beam"},
        )


# ---------------------------------------------------------- incremental run
def run_derivation(store, pings, attitude, rules, label: str, *,
                   branch: str = "main", rationale: str = "",
                   parent_run_id: Optional[str] = None,
                   trigger_event_seq: Optional[int] = None,
                   source_refs: Optional[dict] = None) -> str:
    """Compute a complete run and promote it atomically.

    The promotion either happens in full or not at all; callers needing the
    crash guarantee should kill the process before :meth:`Engine.complete_run`
    (see tests).
    """
    eng = Engine(pings, attitude, rules, source_refs=source_refs)
    nodes = list(eng.run().values())
    manifest = {
        "fp": rules.fp(),
        "n_pings": len(pings),
        "n_nodes": len(nodes),
        "slots": [n["slot"] for n in nodes],
    }
    rid = store.begin_run(label, parent_run_id, branch, manifest, rules.fp(),
                          rationale, trigger_event_seq)
    store.stage_nodes(rid, nodes)
    store.complete_run(rid)
    return rid
