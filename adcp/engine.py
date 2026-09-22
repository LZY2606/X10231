from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import (
    as_vector3,
    geometry_matrix,
    interpolate_heading,
    interpolate_linear,
    ordered_rotation,
    solve_radials,
)
from .normalization import integer_flag, scalar
from .storage import canonical_json, content_id

RULE_VERSION = "adcp-transform-v1"


class DerivationEngine:
    def __init__(
        self,
        sources: Dict[str, Dict[str, Any]],
        *,
        previous_nodes: Optional[Dict[str, Dict[str, Any]]] = None,
        max_attitude_gap_s: float = 1.5,
    ) -> None:
        self.sources = sources
        self.previous_nodes = previous_nodes or {}
        self.max_attitude_gap_s = max_attitude_gap_s
        self.by_kind: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for source in sources.values():
            self.by_kind[source["kind"]].append(source)
        self.nodes: List[Dict[str, Any]] = []
        self.node_ids: Dict[str, Dict[str, Any]] = {}
        self.propagation: List[Dict[str, Any]] = []

    def derive(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        config = self._one("transducer_config")
        declination = self._one("declination")
        installation = self._one("installation_angle")
        order = self._one("transform_order")
        attitudes = self._attitude_table()
        pings = self._ping_table()
        for ping in sorted(pings):
            self._attitude_node(ping, pings[ping], attitudes, declination, installation, order)
            self._derive_water_cell(ping, pings[ping], config, declination, installation, order)
            self._derive_bottom_track(ping, pings[ping], config, declination, installation, order)
        self._profile_nodes(pings, config, declination, installation, order)
        self._explain_propagation()
        summary = {
            "node_count": len(self.nodes),
            "status_counts": self._status_counts(),
            "rule_version": RULE_VERSION,
            "propagation": self.propagation,
        }
        return self.nodes, summary

    def _one(self, kind: str) -> Optional[Dict[str, Any]]:
        accepted = [s for s in self.by_kind[kind] if s["status"] == "accepted"]
        if not accepted:
            return None
        return sorted(accepted, key=lambda item: (item["stable_key"], item["revision_no"]))[-1]

    def _ping_table(self) -> Dict[str, Dict[str, Any]]:
        pings: Dict[str, Dict[str, Any]] = {}
        for kind in ("beam_velocity", "bottom_track_velocity"):
            for source in self.by_kind[kind]:
                value = source.get("normalized_value") or {}
                ping = str(value.get("ping"))
                if ping and ping not in pings:
                    pings[ping] = {"ping": ping, "time": float(value.get("time", 0.0))}
        for source in self.by_kind["heading_pitch_roll"]:
            value = source.get("normalized_value") or {}
            ping = str(value.get("ping"))
            if ping and ping not in pings:
                pings[ping] = {"ping": ping, "time": float(value.get("time", 0.0))}
        return dict(sorted(pings.items(), key=lambda item: item[1]["time"]))

    def _attitude_table(self) -> List[Dict[str, Any]]:
        rows = []
        for source in self.by_kind["heading_pitch_roll"]:
            value = source.get("normalized_value")
            if source["status"] != "accepted" or not value:
                continue
            rows.append(
                {
                    "time": float(value["time"]),
                    "heading": float(value["heading"]),
                    "pitch": float(value["pitch"]),
                    "roll": float(value["roll"]),
                    "revision_id": source["revision_id"],
                }
            )
        return sorted(rows, key=lambda row: row["time"])

    def _group_water(self, ping: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
        groups: Dict[Tuple[str, int], Dict[str, Any]] = defaultdict(dict)
        kinds = ("beam_velocity", "correlation", "amplitude", "bad_beam_flag", "depth_cell")
        for kind in kinds:
            for source in self.by_kind[kind]:
                value = source.get("normalized_value") or {}
                if str(value.get("ping")) != ping:
                    continue
                cell = str(value.get("cell"))
                beam = int(value.get("beam", 0))
                if kind == "depth_cell":
                    groups[(cell, 0)][kind] = source
                else:
                    groups[(cell, beam)][kind] = source
        return groups

    def _group_bottom(self, ping: str) -> Dict[int, Dict[str, Any]]:
        groups: Dict[int, Dict[str, Any]] = defaultdict(dict)
        for kind in ("bottom_track_velocity", "correlation", "amplitude", "bad_beam_flag"):
            for source in self.by_kind[kind]:
                value = source.get("normalized_value") or {}
                if str(value.get("ping")) != ping:
                    continue
                if not str(value.get("cell", "BT")).startswith("BT") and kind != "bottom_track_velocity":
                    continue
                beam = int(value.get("beam", 0))
                groups[beam][kind] = source
        return groups

    def _beam_quality(
        self,
        *,
        domain: str,
        ping: str,
        cell: str,
        beam: int,
        sources_by_kind: Dict[str, Dict[str, Any]],
        velocity_kind: str,
        config: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        vel = sources_by_kind.get(velocity_kind)
        if vel is None:
            return None
        diagnostics: List[str] = []
        radial = None
        unit = None
        raw = vel.get("raw_value")
        try:
            normalized = vel.get("normalized_value") or {}
            radial, unit = scalar(normalized.get("value"), normalized.get("unit"), "m/s")
        except Exception as exc:
            diagnostics.append("velocity_not_normalized:" + str(exc))
        corr_value = None
        corr_source = sources_by_kind.get("correlation")
        if corr_source:
            try:
                cv = corr_source["normalized_value"]
                corr_value, _ = scalar(cv.get("value"), cv.get("unit"), "percent")
            except Exception as exc:
                diagnostics.append("correlation_not_normalized:" + str(exc))
        amp_value = None
        amp_source = sources_by_kind.get("amplitude")
        if amp_source:
            try:
                av = amp_source["normalized_value"]
                amp_value, _ = scalar(av.get("value"), av.get("unit"), "counts")
            except Exception as exc:
                diagnostics.append("amplitude_not_normalized:" + str(exc))
        hard_bad = vel.get("status") != "accepted" or radial is None
        flag_source = sources_by_kind.get("bad_beam_flag")
        if flag_source:
            try:
                hard_bad = hard_bad or integer_flag(
                    (flag_source.get("normalized_value") or {}).get("bad")
                )
            except Exception as exc:
                diagnostics.append("bad_flag_invalid:" + str(exc))
                hard_bad = True
        policy = (config.get("normalized_value") or {}).get("quality_policy", {})
        hard_limit = float(policy.get("correlation_hard_percent", 60.0))
        warn_limit = float(policy.get("correlation_warn_percent", 80.0))
        suspect = corr_value is not None and hard_limit <= corr_value < warn_limit
        if corr_value is not None and corr_value < hard_limit:
            hard_bad = True
        if hard_bad:
            state = "hard_failed"
        elif suspect:
            state = "suspect"
        else:
            state = "accepted"
        value = {
            "domain": domain,
            "ping": ping,
            "cell": cell,
            "beam": beam,
            "raw_representation": raw,
            "radial_velocity_m_s": radial,
            "velocity_unit": "m/s",
            "correlation_percent": corr_value,
            "correlation_unit": "percent" if corr_value is not None else None,
            "amplitude_counts": amp_value,
            "amplitude_unit": "counts" if amp_value is not None else None,
            "quality_state": state,
            "hard_bad": hard_bad,
            "suspect": suspect,
            "thresholds_percent": {"hard": hard_limit, "warn": warn_limit},
        }
        deps = self._deps(sources_by_kind.values())
        return self._node(
            "beam_quality",
            "%s#%s#%s#b%d" % (domain, ping, cell, beam),
            value,
            "hard_failed" if hard_bad else ("suspect" if suspect else "accepted"),
            diagnostics,
            deps,
            extra_basis={"quality_policy": policy},
        )

    def _derive_water_cell(
        self,
        ping: str,
        ping_info: Dict[str, Any],
        config: Optional[Dict[str, Any]],
        declination: Optional[Dict[str, Any]],
        installation: Optional[Dict[str, Any]],
        order: Optional[Dict[str, Any]],
    ) -> None:
        groups = self._group_water(ping)
        cells = sorted({cell for cell, _ in groups if cell != "None"})
        for cell in cells:
            cell_source = groups.get((cell, 0), {}).get("depth_cell")
            cell_status = self._cell_status(ping, cell, cell_source)
            qualities = []
            for beam in range(1, 5):
                quality = self._beam_quality(
                    domain="water",
                    ping=ping,
                    cell=cell,
                    beam=beam,
                    sources_by_kind=groups.get((cell, beam), {}),
                    velocity_kind="beam_velocity",
                    config=config or {"normalized_value": {}},
                )
                if quality:
                    qualities.append(quality)
            if cell_status["status"] == "excluded":
                continue
            self._candidate_chain(
                domain="water",
                ping=ping,
                cell=cell,
                qualities=qualities,
                config=config,
                declination=declination,
                installation=installation,
                order=order,
                ping_info=ping_info,
            )

    def _cell_status(self, ping: str, cell: str, source: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        diagnostics = []
        valid = True
        depth = None
        unit = None
        if source is not None and source["status"] == "accepted":
            value = source.get("normalized_value") or {}
            try:
                valid = bool(value.get("valid", True))
                depth, unit = scalar(value.get("depth"), value.get("unit"), "m")
            except Exception as exc:
                valid = False
                diagnostics.append("depth_cell_not_normalized:" + str(exc))
        value = {
            "ping": ping,
            "cell": cell,
            "depth_m": depth,
            "depth_unit": unit,
            "valid": valid,
            "raw_representation": source.get("raw_value") if source else None,
        }
        node = self._node(
            "depth_cell",
            "%s#%s" % (ping, cell),
            value,
            "accepted" if valid else "excluded",
            diagnostics + ([] if valid else ["invalid_depth_cell"]),
            self._deps([source] if source else []),
        )
        return node

    def _derive_bottom_track(
        self,
        ping: str,
        ping_info: Dict[str, Any],
        config: Optional[Dict[str, Any]],
        declination: Optional[Dict[str, Any]],
        installation: Optional[Dict[str, Any]],
        order: Optional[Dict[str, Any]],
    ) -> None:
        groups = self._group_bottom(ping)
        if not groups:
            return
        qualities = []
        for beam in sorted(groups):
            quality = self._beam_quality(
                domain="bottom_track",
                ping=ping,
                cell="BT",
                beam=beam,
                sources_by_kind=groups[beam],
                velocity_kind="bottom_track_velocity",
                config=config or {"normalized_value": {}},
            )
            if quality:
                qualities.append(quality)
        self._candidate_chain(
            domain="bottom_track",
            ping=ping,
            cell="BT",
            qualities=qualities,
            config=config,
            declination=declination,
            installation=installation,
            order=order,
            ping_info=ping_info,
        )

    def _candidate_chain(
        self,
        *,
        domain: str,
        ping: str,
        cell: str,
        qualities: Sequence[Dict[str, Any]],
        config: Optional[Dict[str, Any]],
        declination: Optional[Dict[str, Any]],
        installation: Optional[Dict[str, Any]],
        order: Optional[Dict[str, Any]],
        ping_info: Dict[str, Any],
    ) -> None:
        subject = "%s#%s#%s" % (domain, ping, cell)
        if config is None:
            self._node(
                "candidate_set",
                subject,
                {"ping": ping, "cell": cell, "domain": domain, "candidates": []},
                "unavailable",
                ["missing_transducer_geometry"],
                [],
            )
            return
        beams = (config.get("normalized_value") or {}).get("beams")
        if not beams or len(beams) != 4:
            self._node(
                "candidate_set",
                subject,
                {"ping": ping, "cell": cell, "domain": domain, "candidates": []},
                "unavailable",
                ["invalid_transducer_geometry"],
                [{"id": config["revision_id"], "role": "transducer_geometry"}],
            )
            return
        beam_by_no = {int(beam["beam"]): beam for beam in beams}
        quality_by_beam = {int(q["value"]["beam"]): q for q in qualities}
        hard_failed = sorted(b for b in range(1, 5) if quality_by_beam.get(b, {}).get("status") == "hard_failed")
        suspects = sorted(b for b in range(1, 5) if quality_by_beam.get(b, {}).get("status") == "suspect")
        candidates = []
        if not hard_failed and len(quality_by_beam) == 4:
            candidate = self._solve_candidate(
                domain=domain,
                ping=ping,
                cell=cell,
                label="4beam",
                beam_numbers=[1, 2, 3, 4],
                qualities=[quality_by_beam[b] for b in range(1, 5)],
                beam_by_no=beam_by_no,
            )
            if candidate:
                candidates.append(candidate)
        # Three beam solutions are always independent candidates. They are
        # emitted whenever the three beams are available; the reviewer can see
        # that an all-beam 4beam solution is separate and compare disagreement.
        for omitted in range(1, 5):
            numbers = [b for b in range(1, 5) if b != omitted]
            if any(quality_by_beam.get(b, {}).get("status") == "hard_failed" for b in numbers):
                continue
            if len([b for b in numbers if b in quality_by_beam]) != 3:
                continue
            candidate = self._solve_candidate(
                domain=domain,
                ping=ping,
                cell=cell,
                label="3beam_omit_b%d" % omitted,
                beam_numbers=numbers,
                qualities=[quality_by_beam[b] for b in numbers],
                beam_by_no=beam_by_no,
            )
            if candidate:
                candidates.append(candidate)
        if hard_failed:
            status = "unique" if len(hard_failed) == 1 and len(candidates) == 1 else "unavailable"
            diagnostics = ["hard_failed_beams:" + ",".join(map(str, hard_failed))]
        elif suspects:
            status = "ambiguous" if len(candidates) > 1 else "unavailable"
            diagnostics = ["suspect_beams:" + ",".join(map(str, suspects))]
        else:
            status = "accepted" if len(candidates) == 1 and candidates[0]["value"]["solution_class"] == "4beam" else "ambiguous"
            diagnostics = []
        candidate_ids = [candidate["id"] for candidate in candidates]
        set_node = self._node(
            "candidate_set",
            subject,
            {
                "domain": domain,
                "ping": ping,
                "cell": cell,
                "hard_failed_beams": hard_failed,
                "suspect_beams": suspects,
                "candidate_ids": candidate_ids,
                "resolution_class": status,
                "note": "three_beam_and_four_beam_candidates_are_not_merged",
            },
            status,
            diagnostics,
            [{"id": config["revision_id"], "role": "transducer_geometry"}]
            + [{"id": candidate_id, "role": "instrument_candidate"} for candidate_id in candidate_ids],
            extra_basis={"rule_version": RULE_VERSION},
        )
        attitude_node = self.node_ids.get("attitude#%s" % ping)
        for candidate in candidates:
            self._coordinate_chain(
                candidate=candidate,
                set_node=set_node,
                attitude_node=attitude_node,
                declination=declination,
                installation=installation,
                order=order,
            )

    def _solve_candidate(
        self,
        *,
        domain: str,
        ping: str,
        cell: str,
        label: str,
        beam_numbers: Sequence[int],
        qualities: Sequence[Dict[str, Any]],
        beam_by_no: Dict[int, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        radials = []
        geometry_rows = []
        deps = []
        for beam_no, quality in zip(beam_numbers, qualities):
            value = quality["value"]
            if value["radial_velocity_m_s"] is None:
                return None
            radials.append(float(value["radial_velocity_m_s"]))
            geometry_rows.append(beam_by_no[beam_no])
            deps.append({"id": quality["id"], "role": "beam_quality"})
        try:
            mat = geometry_matrix(geometry_rows)
            instrument, residual = solve_radials(mat, np.asarray(radials))
        except Exception as exc:
            key = "%s#%s#%s#%s" % (domain, ping, cell, label)
            return self._node(
                "instrument_candidate",
                key,
                {"solution_class": "invalid", "beam_numbers": list(beam_numbers)},
                "unavailable",
                ["radial_solve_failed:" + str(exc)],
                deps,
            )
        solution_class = "4beam" if len(beam_numbers) == 4 else "3beam"
        value = {
            "domain": domain,
            "ping": ping,
            "cell": cell,
            "label": label,
            "solution_class": solution_class,
            "solution_is_three_beam": solution_class == "3beam",
            "solution_is_four_beam": solution_class == "4beam",
            "beam_numbers": list(beam_numbers),
            "radial_velocities_m_s": radials,
            "radial_unit": "m/s",
            "instrument_velocity_m_s": [float(x) for x in instrument],
            "coordinate_frame": "instrument:beam_geometry_revision_required",
            "least_squares_residual_m_s": residual,
            "geometry_rows": geometry_rows,
        }
        return self._node(
            "instrument_candidate",
            "%s#%s#%s#%s" % (domain, ping, cell, label),
            value,
            "accepted",
            [],
            deps,
            extra_basis={"geometry_rows": geometry_rows, "rule_version": RULE_VERSION},
        )

    def _coordinate_chain(
        self,
        *,
        candidate: Dict[str, Any],
        set_node: Dict[str, Any],
        attitude_node: Optional[Dict[str, Any]],
        declination: Optional[Dict[str, Any]],
        installation: Optional[Dict[str, Any]],
        order: Optional[Dict[str, Any]],
    ) -> None:
        value = candidate["value"]
        instrument_velocity = as_vector3(value["instrument_velocity_m_s"])
        key_base = "%s#%s#%s#%s" % (
            value["domain"],
            value["ping"],
            value["cell"],
            value["label"],
        )
        install_dep = [{"id": installation["revision_id"], "role": "installation_angle"}] if installation else []
        ship_angles = {
            "heading": 0.0,
            "pitch": float((installation.get("normalized_value") or {}).get("pitch_deg", 0.0)) if installation else 0.0,
            "roll": float((installation.get("normalized_value") or {}).get("roll_deg", 0.0)) if installation else 0.0,
        }
        install_order = (installation.get("normalized_value") or {}).get("order", "heading_pitch_roll") if installation else "heading_pitch_roll"
        ship_matrix = ordered_rotation(install_order, ship_angles)
        ship_velocity = ship_matrix @ instrument_velocity
        ship_node = self._node(
            "ship_velocity",
            key_base,
            {
                **self._coordinate_value_base(value),
                "coordinate_frame": "ship:transducer_installation_revision_required",
                "velocity_m_s": [float(x) for x in ship_velocity],
                "velocity_unit": "m/s",
                "rotation_matrix": ship_matrix.tolist(),
                "rotation_order": install_order,
            },
            "accepted" if installation else "unavailable",
            [] if installation else ["missing_installation_angles"],
            [{"id": candidate["id"], "role": "instrument_candidate"}] + install_dep,
            extra_basis={"installation": installation.get("normalized_value") if installation else None},
        )
        if attitude_node is None or attitude_node["status"] != "accepted":
            diagnostics = [] if attitude_node else ["missing_attitude"]
            if attitude_node and attitude_node["status"] != "accepted":
                diagnostics = list(attitude_node.get("diagnostics", []))
            self._node(
                "earth_velocity",
                key_base,
                {
                    **self._coordinate_value_base(value),
                    "coordinate_frame": "earth:unavailable",
                    "velocity_m_s": None,
                    "velocity_unit": "m/s",
                },
                "unavailable",
                diagnostics or ["attitude_unavailable"],
                [{"id": ship_node["id"], "role": "ship_velocity"}]
                + ([{"id": attitude_node["id"], "role": "attitude"}] if attitude_node else []),
            )
            return
        attitude_value = attitude_node["value"]
        order_name = (
            (order.get("normalized_value") or {}).get("order")
            if order
            else "heading_pitch_roll"
        )
        declination_deg = float((declination.get("normalized_value") or {}).get("declination_deg", 0.0)) if declination else 0.0
        angles = {
            "heading": float(attitude_value["heading_true_deg"]),
            "pitch": float(attitude_value["pitch_deg"]),
            "roll": float(attitude_value["roll_deg"]),
        }
        earth_matrix = ordered_rotation(order_name, angles)
        earth_velocity = earth_matrix @ ship_velocity
        deps = [
            {"id": ship_node["id"], "role": "ship_velocity"},
            {"id": attitude_node["id"], "role": "attitude"},
        ]
        if declination:
            deps.append({"id": declination["revision_id"], "role": "magnetic_declination"})
        if order:
            deps.append({"id": order["revision_id"], "role": "transform_order"})
        self._node(
            "earth_velocity",
            key_base,
            {
                **self._coordinate_value_base(value),
                "coordinate_frame": "earth:true_north:magnetic_declination_required",
                "velocity_m_s": [float(x) for x in earth_velocity],
                "velocity_unit": "m/s",
                "rotation_matrix": earth_matrix.tolist(),
                "rotation_order": order_name,
                "heading_basis": attitude_value["heading_basis"],
                "declination_deg": declination_deg,
                "declination_revision": declination["revision_id"] if declination else None,
            },
            "accepted",
            [],
            deps,
            extra_basis={
                "attitude_fingerprint": attitude_node["fingerprint"],
                "declination": declination.get("normalized_value") if declination else None,
                "transform_order": order.get("normalized_value") if order else order_name,
                "rule_version": RULE_VERSION,
            },
        )

    @staticmethod
    def _coordinate_value_base(value: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "domain": value["domain"],
            "ping": value["ping"],
            "cell": value["cell"],
            "label": value["label"],
            "solution_class": value["solution_class"],
            "solution_is_three_beam": value["solution_is_three_beam"],
            "solution_is_four_beam": value["solution_is_four_beam"],
            "beam_numbers": value["beam_numbers"],
        }

    def _profile_nodes(
        self,
        pings: Dict[str, Dict[str, Any]],
        config: Optional[Dict[str, Any]],
        declination: Optional[Dict[str, Any]],
        installation: Optional[Dict[str, Any]],
        order: Optional[Dict[str, Any]],
    ) -> None:
        for ping in sorted(pings):
            earth_nodes = [
                node
                for node in self.nodes
                if node["node_kind"] == "earth_velocity"
                and node["value"].get("domain") == "water"
                and node["value"].get("ping") == ping
            ]
            depth_nodes = [
                node
                for node in self.nodes
                if node["node_kind"] == "depth_cell" and node["value"].get("ping") == ping
            ]
            included_cells = sorted(
                {node["value"]["cell"] for node in earth_nodes if node["status"] == "accepted"}
            )
            excluded_cells = [
                {
                    "cell": node["value"]["cell"],
                    "depth_m": node["value"].get("depth_m"),
                    "depth_unit": node["value"].get("depth_unit"),
                    "reason": node.get("diagnostics", []),
                    "source_identity": node["id"],
                }
                for node in depth_nodes
                if node["status"] == "excluded"
            ]
            status = "accepted" if included_cells and not excluded_cells else (
                "partial" if included_cells else "excluded"
            )
            self._node(
                "vertical_profile",
                "water#" + ping,
                {
                    "domain": "water",
                    "ping": ping,
                    "included_cells": included_cells,
                    "excluded_cells": excluded_cells,
                    "earth_velocity_node_ids": [node["id"] for node in earth_nodes],
                    "depth_axis_unit": "m",
                    "velocity_unit": "m/s",
                },
                status,
                [] if status == "accepted" else ["profile_contains_excluded_or_missing_cells"],
                [{"id": node["id"], "role": "profile_member"} for node in earth_nodes]
                + [{"id": node["id"], "role": "depth_cell"} for node in depth_nodes],
            )

    def _attitude_node(
        self,
        ping: str,
        ping_info: Dict[str, Any],
        attitudes: Sequence[Dict[str, Any]],
        declination: Optional[Dict[str, Any]],
        installation: Optional[Dict[str, Any]],
        order: Optional[Dict[str, Any]],
    ) -> None:
        t = float(ping_info["time"])
        before = None
        after = None
        for row in attitudes:
            if row["time"] <= t:
                before = row
            if row["time"] >= t and after is None:
                after = row
        exact = next((row for row in attitudes if row["time"] == t), None)
        selected_left = selected_right = exact
        if exact is None:
            selected_left, selected_right = before, after
        diagnostics: List[str] = []
        interpolated = None
        status = "accepted"
        span = 0.0
        if selected_left is None or selected_right is None:
            status = "unavailable"
            diagnostics.append("attitude_gap:no_bracketing_sample")
        else:
            span = abs(selected_right["time"] - selected_left["time"])
            if selected_left is not selected_right and span > self.max_attitude_gap_s:
                status = "unavailable"
                diagnostics.append(
                    "attitude_gap:span_%.3fs_exceeds_%.3fs_interpolation_forbidden"
                    % (span, self.max_attitude_gap_s)
                )
            else:
                if selected_left is selected_right:
                    heading = selected_left["heading"] % 360.0
                    pitch = selected_left["pitch"]
                    roll = selected_left["roll"]
                    diagnostics.append("exact_attitude_sample")
                else:
                    heading = interpolate_heading(
                        selected_left["time"],
                        selected_left["heading"],
                        selected_right["time"],
                        selected_right["heading"],
                        t,
                    )
                    pitch = interpolate_linear(
                        selected_left["time"],
                        selected_left["pitch"],
                        selected_right["time"],
                        selected_right["pitch"],
                        t,
                    )
                    roll = interpolate_linear(
                        selected_left["time"],
                        selected_left["roll"],
                        selected_right["time"],
                        selected_right["roll"],
                        t,
                    )
                    raw_delta = (selected_right["heading"] - selected_left["heading"] + 180.0) % 360.0
                    if abs(raw_delta - 180.0) > 178.0:
                        diagnostics.append("heading_wrap_360_unwrapped_for_interpolation")
                    else:
                        diagnostics.append("attitude_interpolated_within_adjacent_samples")
                declination_deg = (
                    float((declination.get("normalized_value") or {}).get("declination_deg", 0.0))
                    if declination
                    else 0.0
                )
                interpolated = {
                    "ping": ping,
                    "time_s": t,
                    "time_unit": "s",
                    "heading_magnetic_deg": heading,
                    "heading_true_deg": (heading + declination_deg) % 360.0,
                    "heading_basis": "true_north" if declination else "magnetic_north",
                    "pitch_deg": pitch,
                    "roll_deg": roll,
                    "angle_unit": "degree",
                    "declination_deg": declination_deg,
                    "bracket_times_s": [selected_left["time"], selected_right["time"]],
                    "gap_span_s": span,
                }
        deps = []
        if selected_left is not None:
            deps.append({"id": selected_left["revision_id"], "role": "attitude_sample_left"})
        if selected_right is not None and selected_right is not selected_left:
            deps.append({"id": selected_right["revision_id"], "role": "attitude_sample_right"})
        if declination is not None:
            deps.append({"id": declination["revision_id"], "role": "magnetic_declination"})
        self._node(
            "attitude",
            ping,
            interpolated or {"ping": ping, "time_s": t, "time_unit": "s"},
            status,
            diagnostics,
            deps,
            extra_basis={
                "max_attitude_gap_s": self.max_attitude_gap_s,
            },
        )

    def _node(
        self,
        node_kind: str,
        stable_key: str,
        value: Dict[str, Any],
        status: str,
        diagnostics: Sequence[str],
        dependencies: Sequence[Dict[str, str]],
        *,
        extra_basis: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        node_id = content_id("node", {"kind": node_kind, "stable_key": stable_key})
        basis = {
            "kind": node_kind,
            "stable_key": stable_key,
            "value": value,
            "status": status,
            "diagnostics": list(diagnostics),
            "dependencies": sorted(
                ({"id": dep["id"], "role": dep["role"]} for dep in dependencies),
                key=lambda dep: (dep["role"], dep["id"]),
            ),
            "extra_basis": extra_basis or {},
            "rule_version": RULE_VERSION,
        }
        node = {
            "id": node_id,
            "node_kind": node_kind,
            "stable_key": stable_key,
            "fingerprint": content_id("fp", basis),
            "status": status,
            "value": value,
            "diagnostics": list(diagnostics),
            "dependencies": list(dependencies),
        }
        self.nodes.append(node)
        self.node_ids[node_kind + "#" + stable_key] = node
        return node

    @staticmethod
    def _deps(sources: Iterable[Optional[Dict[str, Any]]]) -> List[Dict[str, str]]:
        return [
            {"id": source["revision_id"], "role": "source:" + source["kind"]}
            for source in sources
            if source is not None
        ]

    def _status_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for node in self.nodes:
            counts[node["status"]] = counts.get(node["status"], 0) + 1
        return counts

    def _explain_propagation(self, changed_sources: Optional[Sequence[str]] = None) -> None:
        edges: Dict[str, List[str]] = defaultdict(list)
        for node in self.nodes:
            for dep in node.get("dependencies", []):
                edges[dep["id"]].append(node["id"])
        changed = set(changed_sources or [])
        for node in self.nodes:
            old = self.previous_nodes.get(node["id"])
            impacted_sources = sorted(
                dep_id for dep_id in changed if self._ancestor(node["id"], dep_id, edges)
            )
            if old is None:
                state, reason = "created", "new_required_node"
            elif old.get("fingerprint") == node["fingerprint"]:
                state, reason = "still_valid", "fingerprint_and_all_included_inputs_unchanged"
            elif impacted_sources:
                state, reason = "recomputed", "changed_dependency_reached_this_node"
            else:
                state, reason = "recomputed", "fingerprint_changed_without_marked_source_change"
            self.propagation.append(
                {
                    "node_id": node["id"],
                    "node_kind": node["node_kind"],
                    "stable_key": node["stable_key"],
                    "state": state,
                    "reason": reason,
                    "old_fingerprint": old.get("fingerprint") if old else None,
                    "new_fingerprint": node["fingerprint"],
                    "affected_by_source_revision": impacted_sources,
                }
            )

    @staticmethod
    def _ancestor(node_id: str, ancestor_id: str, edges: Dict[str, List[str]]) -> bool:
        seen = set()
        stack = [ancestor_id]
        while stack:
            current = stack.pop()
            if current == node_id:
                return True
            if current in seen:
                continue
            seen.add(current)
            stack.extend(edges.get(current, []))
        return False


def derive_from_sources(
    sources: Dict[str, Dict[str, Any]],
    *,
    previous_nodes: Optional[Dict[str, Dict[str, Any]]] = None,
    changed_sources: Optional[Sequence[str]] = None,
    max_attitude_gap_s: float = 1.5,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    engine = DerivationEngine(
        sources,
        previous_nodes=previous_nodes,
        max_attitude_gap_s=max_attitude_gap_s,
    )
    nodes, summary = engine.derive()
    if changed_sources is not None:
        engine.propagation = []
        engine._explain_propagation(changed_sources)
        summary["propagation"] = engine.propagation
    return nodes, summary
