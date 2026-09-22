from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .storage import Storage, canonical_json, digest, stable_id
from .transforms import (
    BEAMS,
    MAX_ATTITUDE_GAP_S,
    as_float,
    beam_candidates,
    beam_matrix,
    classify_beam,
    parse_geometry,
    transform_fingerprint,
    transform_vector,
    unit_float,
    unwrap_degrees,
    wrap_360,
)


class DerivationEngine:
    def __init__(self, storage: Storage):
        self.storage = storage

    def build(self, conn: sqlite3.Connection, version_id: str, graph_run_id: str) -> Dict[str, List[str]]:
        sources = [dict(row) for row in conn.execute("SELECT * FROM source_records WHERE version_id=?", (version_id,))]
        previous = self._previous_version(conn, version_id)
        old_nodes = self._old_nodes(conn, previous)
        old_derived = self._old_derived(conn, previous)
        groups = self._group_sources(sources)
        global_context = self._global_context(groups)
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        derived: List[Dict[str, Any]] = []
        summary = {"reused": [], "recomputed": [], "blocked": []}

        attitudes = self._attitude_series(groups)
        attitude_node = self._attitude_node(attitudes, global_context["fingerprint"])
        nodes.append(attitude_node)
        summary[attitude_node["status"].replace("reused", "reused")].append(attitude_node["node_id"])

        for ping_id, ping in sorted(groups["pings"].items()):
            water = self._solve_scope(
                ping_id,
                ping["time"],
                "water",
                [(cell_index, cell_beams) for cell_index, cell_beams in sorted(ping["water"]["cells"].items())],
                global_context,
                attitudes,
                groups["depth_cells"],
                old_nodes,
                old_derived,
                nodes,
                edges,
                derived,
            )
            self._solve_scope(
                ping_id,
                ping["time"],
                "bottom",
                [(None, ping["bottom"])],
                global_context,
                attitudes,
                groups["depth_cells"],
                old_nodes,
                old_derived,
                nodes,
                edges,
                derived,
                water_earth=water,
            )

        for node in nodes:
            if node["status"] == "recomputed":
                summary["recomputed"].append(node["node_id"])
            elif node["status"] == "reused":
                summary["reused"].append(node["node_id"])
            elif node["status"] == "blocked":
                summary["blocked"].append(node["node_id"])
        self.storage.insert_graph_outputs(
            conn,
            graph_run_id=graph_run_id,
            version_id=version_id,
            nodes=nodes,
            edges=edges,
            derived=derived,
        )
        return summary

    @staticmethod
    def _previous_version(conn: sqlite3.Connection, version_id: str) -> Optional[str]:
        row = conn.execute("SELECT parent_version_id FROM versions WHERE version_id=?", (version_id,)).fetchone()
        return row["parent_version_id"] if row else None

    @staticmethod
    def _old_nodes(conn: sqlite3.Connection, version_id: Optional[str]) -> Dict[str, Dict[str, Any]]:
        if not version_id:
            return {}
        run = conn.execute("SELECT graph_run_id FROM versions WHERE version_id=?", (version_id,)).fetchone()
        if not run:
            return {}
        return {
            row["node_id"]: {**dict(row), "reasons": json.loads(row["reasons_json"])}
            for row in conn.execute("SELECT * FROM graph_nodes WHERE graph_run_id=?", (run["graph_run_id"],))
        }

    @staticmethod
    def _old_derived(conn: sqlite3.Connection, version_id: Optional[str]) -> Dict[str, Dict[str, Any]]:
        if not version_id:
            return {}
        result = {}
        for row in conn.execute("SELECT * FROM derived_records WHERE version_id=?", (version_id,)):
            item = dict(row)
            result[f'{item["logical_id"]}|{item["candidate_id"] or ""}'] = item
        return result

    @staticmethod
    def _group_sources(sources: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        groups: Dict[str, Any] = {
            "pings": {},
            "global": {},
            "depth_cells": {},
            "attitude": {},
        }
        for record in sources:
            category = record["category"]
            logical = record["logical_id"]
            if category == "depth_cell":
                groups["depth_cells"][int(record["depth_cell"])] = record
            elif category in {"heading", "pitch", "roll"}:
                groups["attitude"].setdefault(record["ping_id"], {"time": record["ping_time_s"]})[category] = record
            elif category in {"magnetic_declination", "mount_heading", "mount_pitch", "mount_roll", "transducer_geometry", "transform_order"}:
                groups["global"][category] = record
            elif category.startswith(("water_", "bottom_")):
                scope, _, metric = category.partition("_")
                ping = groups["pings"].setdefault(
                    record["ping_id"],
                    {"time": record["ping_time_s"], "water": {"cells": {}}, "bottom": {}},
                )
                ping["time"] = record["ping_time_s"]
                if scope == "water":
                    cell = int(record["depth_cell"]) if record.get("depth_cell") is not None else 0
                    ping[scope]["cells"].setdefault(cell, {}).setdefault(int(record["beam"]), {})[metric] = record
                else:
                    ping[scope].setdefault(int(record["beam"]), {})[metric] = record
        return groups

    @staticmethod
    def _attitude_series(groups: Dict[str, Any]) -> List[Dict[str, Any]]:
        result = []
        for ping_id in sorted(groups["attitude"], key=lambda key: (groups["attitude"][key]["time"], key)):
            item = {"ping_id": ping_id, "time": groups["attitude"][ping_id]["time"]}
            for axis in ("heading", "pitch", "roll"):
                record = groups["attitude"][ping_id].get(axis)
                value, diagnostics = unit_float(record, si_unit="radian") if axis == "heading" else unit_float(record, si_unit="radian")
                if value is None and record and record.get("unit") in {"degree", "deg", "°"}:
                    value, diagnostics = None, []
                    value_float, bad = as_float(record)
                    if value_float is not None:
                        value = math.radians(value_float)
                    else:
                        diagnostics = bad
                item[axis] = value
                item[f"{axis}_record"] = record
                item[f"{axis}_diagnostics"] = diagnostics
            result.append(item)
        return result

    def _global_context(self, groups: Dict[str, Any]) -> Dict[str, Any]:
        global_records = groups["global"]
        geometry, geometry_id = parse_geometry(global_records.get("transducer_geometry"))
        mount = {}
        mount_values = {}
        for axis in ("heading", "pitch", "roll"):
            record = global_records.get(f"mount_{axis}")
            mount[axis] = record
            value, _ = unit_float(record, si_unit="radian")
            mount_values[axis] = value if value is not None else 0.0
        declination, _ = unit_float(global_records.get("magnetic_declination"), si_unit="radian")
        if declination is None:
            declination = 0.0
        order_record = global_records.get("transform_order")
        order = str(order_record["raw_repr"]) if order_record else "MOUNT_THEN_HPR"
        return {
            "geometry": geometry,
            "geometry_id": geometry_id,
            "matrix": beam_matrix(geometry),
            "mount_records": mount,
            "mount": mount_values,
            "declination": declination,
            "declination_record": global_records.get("magnetic_declination"),
            "order_record": order_record,
            "order": order,
            "fingerprint": transform_fingerprint(geometry_id, global_records.get("magnetic_declination"), mount, order_record),
        }

    def _attitude_node(self, attitudes: List[Dict[str, Any]], context_fingerprint: str) -> Dict[str, Any]:
        reasons: List[Dict[str, Any]] = []
        complete = [item for item in attitudes if all(item[axis] is not None for axis in ("heading", "pitch", "roll"))]
        for item in attitudes:
            for axis in ("heading", "pitch", "roll"):
                for diagnostic in item.get(f"{axis}_diagnostics", []):
                    reasons.append({**diagnostic, "ping_id": item["ping_id"], "axis": axis})
        gaps: List[Tuple[str, str, float]] = []
        for previous, current in zip(attitudes, attitudes[1:]):
            delta = float(current["time"]) - float(previous["time"])
            if delta > MAX_ATTITUDE_GAP_S:
                gaps.append((previous["ping_id"], current["ping_id"], delta))
                reasons.append(
                    {
                        "code": "attitude-gap-no-interpolation",
                        "message": "attitude is not interpolated across a time gap",
                        "from_ping": previous["ping_id"],
                        "to_ping": current["ping_id"],
                        "gap_s": round(delta, 6),
                    }
                )
        headings = [math.degrees(item["heading"]) for item in complete]
        unwrapped = unwrap_degrees(headings) if headings else []
        wraps = []
        for index, heading in enumerate(headings):
            if index > 0 and abs(unwrapped[index] - unwrapped[index - 1]) > 180.0:
                wraps.append(complete[index]["ping_id"])
        if wraps:
            reasons.append(
                {
                    "code": "heading-crossed-360-unwrapped",
                    "message": "heading continuity is computed after circular unwrap",
                    "pings": wraps,
                }
            )
        fingerprint = digest(
            [
                context_fingerprint,
                [
                    [
                        item["ping_id"],
                        item["time"],
                        None if item["heading"] is None else round(item["heading"], 12),
                        None if item["pitch"] is None else round(item["pitch"], 12),
                        None if item["roll"] is None else round(item["roll"], 12),
                    ]
                    for item in attitudes
                ],
            ]
        )
        return {
            "node_id": "attitude",
            "stage": "attitude",
            "status": "blocked" if gaps or any(item[axis] is None for item in attitudes for axis in ("heading", "pitch", "roll")) else "recomputed",
            "fingerprint": fingerprint,
            "reasons": reasons,
        }

    def _solve_scope(
        self,
        ping_id: str,
        ping_time: float,
        scope: str,
        cell_groups: Sequence[Tuple[Optional[int], Dict[int, Dict[str, Dict[str, Any]]]]],
        context: Dict[str, Any],
        attitudes: List[Dict[str, Any]],
        depth_cells: Dict[int, Dict[str, Any]],
        old_nodes: Dict[str, Dict[str, Any]],
        old_derived: Dict[str, Dict[str, Any]],
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        derived: List[Dict[str, Any]],
        water_earth: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> Optional[Dict[int, Dict[str, Any]]]:
        attitude = next((item for item in attitudes if item["ping_id"] == ping_id), None)
        attitude_ok = attitude is not None and all(attitude[axis] is not None for axis in ("heading", "pitch", "roll"))
        adjacent_gap = self._inside_gap(ping_id, ping_time, attitudes)
        ground_results: Dict[int, Dict[str, Any]] = {}
        for cell_index, beam_records in cell_groups:
            if scope == "water" and cell_index is not None:
                depth_record = depth_cells.get(cell_index)
                depth, depth_diagnostics = unit_float(depth_record, si_unit="m")
                if depth_record is None or depth is None or depth <= 0:
                    self._blocked_cell(
                        ping_id,
                        ping_time,
                        scope,
                        cell_index,
                        depth_record,
                        depth_diagnostics or [{"code": "missing-depth-cell", "message": "depth cell definition is absent"}],
                        context,
                        old_nodes,
                        nodes,
                        edges,
                        derived,
                    )
                    continue
            else:
                depth = None
                depth_record = None
                depth_diagnostics = []
            qualities = []
            beam_diagnostics = []
            for beam in BEAMS:
                records = beam_records.get(beam, {})
                velocity, velocity_diag = unit_float(records.get("velocity"), si_unit="m/s")
                correlation, corr_diag = unit_float(records.get("correlation"), si_unit="count")
                amplitude, amp_diag = unit_float(records.get("amplitude"), si_unit="dB")
                flag_value, flag_diag = as_float(records.get("bad_flag"))
                flag = str(flag_value) if flag_value is not None else None
                diagnostics = velocity_diag + corr_diag + amp_diag + flag_diag
                beam_diagnostics.extend(diagnostics)
                quality = classify_beam(beam, velocity, correlation, amplitude, flag)
                qualities.append(quality)
                self._quality_record(ping_id, ping_time, scope, cell_index, quality, records, derived)
            candidates, candidate_status, candidate_diag = beam_candidates(context["matrix"], qualities)
            source_fp = digest(
                [
                    context["fingerprint"],
                    scope,
                    ping_id,
                    cell_index,
                    [
                        [
                            beam,
                            (beam_records.get(beam, {}).get(metric) or {}).get("record_id"),
                            (beam_records.get(beam, {}).get(metric) or {}).get("raw_repr"),
                        ]
                        for beam in BEAMS
                        for metric in ("velocity", "correlation", "amplitude", "bad_flag")
                    ],
                    depth,
                ]
            )
            node_id = f"{scope}.beam.{ping_id}.{cell_index if cell_index is not None else 'bottom'}"
            reasons = [{"code": q.state, "beam": q.beam, "reasons": q.reasons} for q in qualities if q.state != "good"]
            reasons.extend(candidate_diag)
            if depth_diagnostics:
                reasons.append({"code": "invalid-depth-cell", "diagnostics": depth_diagnostics})
            old = old_nodes.get(node_id)
            status = "reused" if old and old.get("fingerprint") == source_fp and old.get("status") != "reused" else "recomputed"
            if not candidates:
                status = "blocked"
            node = {
                "node_id": node_id,
                "stage": f"{scope}_beam",
                "ping_id": ping_id,
                "depth_cell": cell_index,
                "status": status,
                "fingerprint": source_fp,
                "reasons": reasons,
            }
            nodes.append(node)
            edges.append({"from": "attitude", "to": node_id, "relation": "applies-to"})
            if status == "reused":
                self._reuse_node_records(node_id, old, old_derived, nodes, derived)
                continue
            earth_by_candidate: Dict[str, Any] = {}
            for candidate in candidates:
                logical_base = f"{scope}.velocity.{ping_id}.{cell_index if cell_index is not None else 'bottom'}"
                self._candidate_records(
                    logical_base,
                    ping_id,
                    ping_time,
                    cell_index,
                    candidate,
                    context,
                    attitude if attitude_ok and not adjacent_gap else None,
                    candidate_status,
                    derived,
                    earth_by_candidate,
                )
            if scope == "bottom":
                self._ground_records(ping_id, ping_time, water_earth or {}, earth_by_candidate, candidate_status, context, attitude_ok and not adjacent_gap, derived)
            else:
                ground_results[cell_index or 0] = {"status": candidate_status, "earth": earth_by_candidate}
        return ground_results if scope == "water" else None

    @staticmethod
    def _inside_gap(ping_id: str, ping_time: float, attitudes: List[Dict[str, Any]]) -> bool:
        ordered = sorted((item for item in attitudes), key=lambda item: (item["time"], item["ping_id"]))
        for previous, current in zip(ordered, ordered[1:]):
            if previous["time"] < ping_time < current["time"] and current["time"] - previous["time"] > MAX_ATTITUDE_GAP_S:
                return True
        return False

    @staticmethod
    def _quality_record(
        ping_id: str,
        ping_time: float,
        scope: str,
        cell_index: Optional[int],
        quality: Any,
        records: Dict[int, Dict[str, Any]],
        derived: List[Dict[str, Any]],
    ) -> None:
        derived.append(
            {
                "logical_id": f"{scope}.quality.{ping_id}.{cell_index if cell_index is not None else 'bottom'}.b{quality.beam}",
                "category": f"{scope}_beam_quality",
                "ping_id": ping_id,
                "ping_time_s": ping_time,
                "beam": quality.beam,
                "depth_cell": cell_index,
                "value": {
                    "state": quality.state,
                    "flag": quality.flag,
                    "correlation": quality.correlation,
                    "amplitude": quality.amplitude,
                    "velocity": quality.velocity,
                    "reasons": quality.reasons,
                },
                "unit": "mixed",
                "datum": "beam-frame-quality",
                "status": "excluded" if quality.state == "hard-invalid" else "marginal" if quality.state == "marginal" else "valid",
                "diagnostics": quality.reasons,
                "source_event_id": (records.get("bad_flag") or {}).get("record_id"),
            }
        )

    def _blocked_cell(
        self,
        ping_id: str,
        ping_time: float,
        scope: str,
        cell_index: int,
        depth_record: Optional[Dict[str, Any]],
        diagnostics: List[Dict[str, Any]],
        context: Dict[str, Any],
        old_nodes: Dict[str, Dict[str, Any]],
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        derived: List[Dict[str, Any]],
    ) -> None:
        logical = f"{scope}.excluded-cell.{ping_id}.{cell_index}"
        node_id = f"{scope}.cell.{ping_id}.{cell_index}"
        fingerprint = digest([logical, (depth_record or {}).get("record_id"), (depth_record or {}).get("raw_repr"), context["fingerprint"]])
        old = old_nodes.get(node_id)
        nodes.append(
            {
                "node_id": node_id,
                "stage": "depth_cell_validation",
                "ping_id": ping_id,
                "depth_cell": cell_index,
                "status": "reused" if old and old.get("fingerprint") == fingerprint and old.get("status") == "blocked" else "blocked",
                "fingerprint": fingerprint,
                "reasons": diagnostics,
            }
        )

    @staticmethod
    def _reuse_node_records(
        node_id: str,
        old_node: Dict[str, Any],
        old_derived: Dict[str, Dict[str, Any]],
        nodes: List[Dict[str, Any]],
        derived: List[Dict[str, Any]],
    ) -> None:
        scope, _, suffix = node_id.partition(".beam.")
        ping_id, _, cell_suffix = suffix.partition(".")
        cell_token = "bottom" if cell_suffix == "bottom" else cell_suffix
        prefixes = [
            f"{scope}.quality.{ping_id}.{cell_token}.",
            f"{scope}.velocity.{ping_id}.{cell_token}.",
            f"water.ground.{ping_id}.{cell_token}.",
        ]
        retained = []
        for item in old_derived.values():
            if any(item["logical_id"].startswith(prefix) for prefix in prefixes):
                copied = dict(item)
                copied["record_id"] = item["record_id"]
                copied["value"] = json.loads(item["value_json"])
                copied["diagnostics"] = json.loads(item["diagnostic_json"])
                derived.append(copied)
                retained.append(item["record_id"])
        old_node["retained_record_id"] = ",".join(retained)

    def _candidate_records(
        self,
        logical_base: str,
        ping_id: str,
        ping_time: float,
        cell_index: Optional[int],
        candidate: Dict[str, Any],
        context: Dict[str, Any],
        attitude: Optional[Dict[str, Any]],
        candidate_status: str,
        derived: List[Dict[str, Any]],
        earth_by_candidate: Optional[Dict[str, Any]],
    ) -> None:
        candidate_id = candidate["candidate_id"]
        common = {
            "ping_id": ping_id,
            "ping_time_s": ping_time,
            "depth_cell": cell_index,
            "candidate_id": candidate_id,
            "fingerprint": digest(
                [
                    context["fingerprint"],
                    logical_base,
                    candidate_id,
                    candidate["vector"],
                    context["order"],
                ]
            ),
        }
        instrument = candidate["vector"]
        self._append_velocity(
            derived,
            {
                **common,
                "logical_id": f"{logical_base}.instrument",
                "category": "instrument_velocity",
                "value": {
                    "vector": instrument,
                    "rank": candidate["rank"],
                    "omitted_beams": candidate["omitted_beams"],
                    "assumption": candidate["assumption"],
                },
                "unit": "m/s",
                "datum": "instrument-frame:x-east-ish,y-north-ish,z-up",
                "status": "candidate" if candidate_status == "ambiguous" else candidate_status,
                "diagnostics": [{"code": "candidate-retained", "candidate_id": candidate_id, "rank": candidate["rank"]}],
            },
        )
        if attitude is None:
            self._append_velocity(
                derived,
                {
                    **common,
                    "logical_id": f"{logical_base}.ship",
                    "category": "ship_velocity",
                    "value": {"vector": None, "blocked_stage": "attitude"},
                    "unit": "m/s",
                    "datum": "ship-frame:x-starboard,y-forward,z-up",
                    "status": "blocked",
                    "diagnostics": [{"code": "attitude-unavailable-or-gap", "message": "no interpolation across attitude gap"}],
                },
            )
            self._append_velocity(
                derived,
                {
                    **common,
                    "logical_id": f"{logical_base}.earth",
                    "category": "earth_velocity",
                    "value": {"vector": None, "blocked_stage": "attitude"},
                    "unit": "m/s",
                    "datum": "earth-frame:ENU;heading:true-north",
                    "status": "blocked",
                    "diagnostics": [{"code": "attitude-unavailable-or-gap", "message": "no interpolation across attitude gap"}],
                },
            )
            return
        v_instrument, v_ship, _, v_earth = transform_vector(
            instrument,
            heading=attitude["heading"],
            pitch=attitude["pitch"],
            roll=attitude["roll"],
            declination=context["declination"],
            mount_heading=context["mount"]["heading"],
            mount_pitch=context["mount"]["pitch"],
            mount_roll=context["mount"]["roll"],
            order=context["order"],
        )
        chain_value = {
            "instrument": v_instrument,
            "ship": v_ship,
            "earth": v_earth,
            "transform_order": context["order"],
            "geometry_fingerprint": context["geometry_id"],
            "declination_record": (context["declination_record"] or {}).get("logical_id"),
            "mount_records": {key: (record or {}).get("logical_id") for key, record in context["mount_records"].items()},
            "heading_deg": round(math.degrees(attitude["heading"]), 9),
            "true_heading_deg": wrap_360(math.degrees(attitude["heading"]) + math.degrees(context["declination"])),
        }
        self._append_velocity(
            derived,
            {
                **common,
                "logical_id": f"{logical_base}.ship",
                "category": "ship_velocity",
                "value": {"vector": v_ship, "chain": chain_value},
                "unit": "m/s",
                "datum": "ship-frame:x-starboard,y-forward,z-up",
                "status": "candidate" if candidate_status == "ambiguous" else candidate_status,
                "diagnostics": [],
            },
        )
        self._append_velocity(
            derived,
            {
                **common,
                "logical_id": f"{logical_base}.earth",
                "category": "earth_velocity",
                "value": {"vector": v_earth, "chain": chain_value},
                "unit": "m/s",
                "datum": "earth-frame:ENU;heading:true-north",
                "status": "candidate" if candidate_status == "ambiguous" else candidate_status,
                "diagnostics": [],
            },
        )
        if earth_by_candidate is not None:
            earth_by_candidate[candidate_id] = v_earth

    @staticmethod
    def _append_velocity(derived: List[Dict[str, Any]], record: Dict[str, Any]) -> None:
        derived.append(record)

    def _ground_records(
        self,
        ping_id: str,
        ping_time: float,
        water_by_cell: Dict[int, Dict[str, Any]],
        bottom_candidates: Dict[str, Any],
        bottom_status: str,
        context: Dict[str, Any],
        attitude_ok: bool,
        derived: List[Dict[str, Any]],
    ) -> None:
        if not attitude_ok:
            return
        for cell_index, water in water_by_cell.items():
            for candidate_id, water_earth in (water.get("earth") or {}).items():
                bottom_earth = bottom_candidates.get(candidate_id) or next(iter(bottom_candidates.values()), None)
                status = "blocked"
                vector = None
                diagnostics = [{"code": "bottom-track-candidate-missing", "message": "ground velocity needs matching bottom-track candidate"}]
                if bottom_earth is not None:
                    vector = [round(float(w) - float(b), 9) for w, b in zip(water_earth, bottom_earth)]
                    status = "candidate" if water.get("status") == "ambiguous" or bottom_status == "ambiguous" else "selected"
                    diagnostics = []
                logical = f"water.ground.{ping_id}.{cell_index}"
                derived.append(
                    {
                        "logical_id": logical,
                        "category": "ground_referenced_velocity",
                        "ping_id": ping_id,
                        "ping_time_s": ping_time,
                        "depth_cell": cell_index,
                        "candidate_id": candidate_id,
                        "value": {"vector": vector, "water_candidate": candidate_id, "bottom_candidate": candidate_id if candidate_id in bottom_candidates else None},
                        "unit": "m/s",
                        "datum": "earth-frame:ENU;bottom-track-referenced",
                        "status": status,
                        "diagnostics": diagnostics,
                        "fingerprint": digest([logical, candidate_id, vector, context["fingerprint"]]),
                    }
                )
        edges.append({"from": (depth_record or {}).get("logical_id", "depth_cell"), "to": node_id, "relation": "excluded-by"})
        derived.append(
            {
                "logical_id": logical,
                "category": "excluded_depth_cell",
                "ping_id": ping_id,
                "ping_time_s": ping_time,
                "depth_cell": cell_index,
                "candidate_id": None,
                "value": {"depth_raw": (depth_record or {}).get("raw_repr"), "depth_unit": (depth_record or {}).get("unit")},
                "unit": "m",
                "datum": "depth-cell",
                "status": "excluded",
                "diagnostics": diagnostics,
                "fingerprint": fingerprint,
                "source_event_id": (depth_record or {}).get("record_id"),
            }
        )
