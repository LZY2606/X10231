from __future__ import annotations

from itertools import combinations

from .attitude import interpolate_attitude
from .fingerprint import fingerprint, stable_json
from .geometry import (
    STANDARD_ORDER,
    beam_direction_matrix,
    solve_beams,
    transform_stages,
)
from .units import normalize_number


def derive_velocity_evidence(casepack: dict, previous_graph: dict | None = None) -> dict:
    rules = casepack["rules"]
    config = casepack["transducer_configuration"]
    geometry_matrix = beam_direction_matrix(
        float(config["beam_angle_deg"]),
        [float(value) for value in config["beam_azimuths_deg"]],
    )
    geometry_fingerprint = fingerprint(
        {
            "geometry_id": config["geometry_id"],
            "beam_angle_deg": config["beam_angle_deg"],
            "beam_azimuths_deg": config["beam_azimuths_deg"],
        }
    )
    transform_order = tuple(config.get("transform_order", STANDARD_ORDER))
    mount = _effective_record(casepack["installation_alignment"], 0.0)
    declination = _effective_record(casepack["magnetic_declination"], 0.0)
    rule_version = fingerprint(
        {
            "rules": rules,
            "geometry_fingerprint": geometry_fingerprint,
            "transform_order": list(transform_order),
            "installation_angle_id": mount["id"],
            "installation_angle_deg": mount["raw_installation_angle_z_deg"],
            "declination_id": declination["id"],
            "declination_version": declination["datum_version"],
            "declination_angle_deg": declination["raw_angle_deg"],
        }
    )
    pings = {ping["id"]: ping for ping in casepack["pings"]}
    source_index = _index_sources(casepack)
    nodes: dict[str, dict] = {}
    candidates: list[dict] = []
    events: list[dict] = []
    diagnostics: list[dict] = []
    bottom_track: list[dict] = []
    events.append(
        {
            "id": "EV-DERIVATION-START",
            "type": "DERIVATION_STARTED",
            "severity": "info",
            "rationale": {"rule_fingerprint": rule_version},
            "subject_id": casepack["casepack_id"],
        }
    )
    for ping in casepack["pings"]:
        ping_id = ping["id"]
        attitude = interpolate_attitude(
            float(ping["time_epoch_s"]),
            casepack["attitude"],
            float(rules["max_attitude_gap_s"]),
        )
        for depth_cell in source_index["depth_by_ping"][ping_id]:
            depth = normalize_number(depth_cell["raw_depth"], depth_cell["raw_depth_unit"], "m")
            depth_diagnostic = None
            if depth.normalized is None:
                depth_diagnostic = {
                    "id": f"DIAG-{depth_cell['id']}-INVALID-DEPTH",
                    "ping_id": ping_id,
                    "depth_cell_id": depth_cell["id"],
                    "code": "INVALID_DEPTH_CELL",
                    "status": "excluded",
                    "raw_value": depth.raw,
                    "raw_unit": depth.unit,
                    "diagnostic": depth.diagnostic,
                    "source_chain": [depth_cell["id"]],
                }
                diagnostics.append(depth_diagnostic)
                events.append(
                    _event(
                        f"EV-{depth_cell['id']}-EXCLUDED",
                        "CELL_EXCLUDED",
                        "warning",
                        depth_cell["id"],
                        {
                            "reason": "INVALID_DEPTH_CELL",
                            "raw_value": depth.raw,
                            "unit": depth.unit,
                            "source_chain": [depth_cell["id"]],
                        },
                        ping_id,
                    )
                )
            quality = _beam_quality(ping_id, depth_cell["id"], source_index, rules)
            radial = {}
            for item in quality["beams"]:
                normalized = normalize_number(item["velocity_raw"], item["velocity_unit"], "m/s")
                item["velocity_normalized_m_s"] = normalized.normalized
                if normalized.normalized is not None and item["state"] != "bad" and depth_diagnostic is None:
                    radial[item["beam_number"]] = normalized.normalized
                else:
                    if depth_diagnostic is not None:
                        item["state"] = "excluded"
                        item["state_reason"] = "INVALID_DEPTH_CELL"
            ping_candidates, ping_events, ping_diagnostics = _solve_ping(
                ping_id=ping_id,
                time_epoch_s=float(ping["time_epoch_s"]),
                depth_cell=depth_cell,
                quality=quality,
                radial=radial,
                depth_excluded=depth_diagnostic is not None,
                geometry_matrix=geometry_matrix,
                geometry_fingerprint=geometry_fingerprint,
                attitude=attitude,
                mount=mount,
                declination=declination,
                transform_order=transform_order,
                rule_version=rule_version,
            )
            candidates.extend(ping_candidates)
            events.extend(ping_events)
            diagnostics.extend(ping_diagnostics)
        for target in ("velocity", "bottom_track"):
            collection = bottom_track if target == "bottom_track" else None
            if collection is None:
                continue
        _solve_bottom_track(
            bottom_track,
            events,
            diagnostics,
            ping_id,
            float(ping["time_epoch_s"]),
            attitude,
            source_index["bottom_by_ping"][ping_id],
            geometry_matrix,
            geometry_fingerprint,
            mount,
            declination,
            transform_order,
            rule_version,
        )
    _build_nodes(nodes, casepack, candidates, diagnostics, events, rule_version, pings)
    _mark_propagation(nodes, previous_graph)
    resolution_summary = _resolution_summary(candidates)
    events.append(
        {
            "id": "EV-DERIVATION-COMPLETE",
            "type": "DERIVATION_COMPLETED",
            "severity": "info",
            "rationale": {
                "rule_fingerprint": rule_version,
                "resolution": resolution_summary,
            },
            "subject_id": casepack["casepack_id"],
        }
    )
    return {
        "casepack_id": casepack["casepack_id"],
        "rule_fingerprint": rule_version,
        "geometry_fingerprint": geometry_fingerprint,
        "pings": _ping_view(casepack, candidates, diagnostics),
        "candidates": candidates,
        "bottom_track": bottom_track,
        "diagnostics": diagnostics,
        "events": events,
        "dependency_graph": {
            "nodes": nodes,
            "edges": _edges(),
        },
        "resolution": resolution_summary,
        "sources_fingerprint": fingerprint(_source_fingerprint_payload(casepack)),
    }


def _effective_record(records: list[dict], time_epoch_s: float) -> dict:
    eligible = [
        record
        for record in records
        if float(record.get("effective_from_time_epoch_s", 0.0)) <= time_epoch_s
    ]
    return sorted(eligible, key=lambda record: (record["id"],))[-1]


def _index_sources(casepack: dict) -> dict:
    indexed = {}
    for kind in ("beam_velocity", "correlation", "amplitude", "bad_beam_flags", "bottom_track"):
        records = {}
        by_ping: dict[str, list[dict]] = {}
        for record in casepack[kind]:
            records[(record["ping_id"], record["beam_number"])] = record
            by_ping.setdefault(record["ping_id"], []).append(record)
        indexed[kind] = records
        indexed[
            kind.replace("beam_velocity", "velocity_by_ping")
            .replace("correlation", "correlation_by_ping")
            .replace("amplitude", "amplitude_by_ping")
            .replace("bad_beam_flags", "flag_by_ping")
            .replace("bottom_track", "bottom_by_ping")
        ] = by_ping
    depth_by_ping: dict[str, list[dict]] = {}
    for record in casepack["depth_cells"]:
        depth_by_ping.setdefault(record["ping_id"], []).append(record)
    indexed["depth_by_ping"] = depth_by_ping
    return indexed


def _beam_quality(ping_id: str, depth_cell_id: str, source_index: dict, rules: dict) -> dict:
    beams = []
    warnings = []
    for beam_number in range(1, 5):
        velocity = source_index["beam_velocity"][(ping_id, beam_number)]
        correlation_record = source_index["correlation"][(ping_id, beam_number)]
        amplitude_record = source_index["amplitude"][(ping_id, beam_number)]
        flag_record = source_index["bad_beam_flags"][(ping_id, beam_number)]
        low_correlation = int(correlation_record["raw_value"]) < int(rules["correlation_threshold"])
        low_amplitude = int(amplitude_record["raw_value"]) < int(rules["amplitude_threshold"])
        flag = str(flag_record["raw_value"])
        state = flag
        reasons = []
        if flag == "bad":
            reasons.append("EXPLICIT_BAD_BEAM_FLAG")
        else:
            if low_correlation:
                reasons.append("LOW_CORRELATION")
            if low_amplitude:
                reasons.append("LOW_AMPLITUDE")
            if reasons and flag != "suspect":
                state = "suspect"
        state = "bad" if flag == "bad" else state
        if state != "ok":
            warnings.append(
                {
                    "beam_number": beam_number,
                    "state": state,
                    "reasons": reasons,
                    "correlation": correlation_record["raw_value"],
                    "amplitude": amplitude_record["raw_value"],
                }
            )
        beams.append(
            {
                "beam_number": beam_number,
                "state": state,
                "state_reason": ";".join(reasons) if reasons else None,
                "velocity_id": velocity["id"],
                "velocity_raw": velocity["raw_value"],
                "velocity_unit": velocity["raw_unit"],
                "correlation_id": correlation_record["id"],
                "correlation_raw": correlation_record["raw_value"],
                "correlation_unit": correlation_record["raw_unit"],
                "amplitude_id": amplitude_record["id"],
                "amplitude_raw": amplitude_record["raw_value"],
                "amplitude_unit": amplitude_record["raw_unit"],
                "flag_id": flag_record["id"],
                "flag_raw": flag_record["raw_value"],
                "flag_unit": flag_record["raw_unit"],
                "depth_cell_id": depth_cell_id,
            }
        )
    return {"ping_id": ping_id, "depth_cell_id": depth_cell_id, "beams": beams, "warnings": warnings}


def _candidate_specs(beams: list[dict]) -> list[tuple[str, tuple[int, ...], str]]:
    bad = {item["beam_number"] for item in beams if item["state"] == "bad"}
    suspect = {item["beam_number"] for item in beams if item["state"] == "suspect"}
    all_beams = set(range(1, 5))
    if not bad and not suspect:
        return [("FOUR_BEAM", (1, 2, 3, 4), "unique")]
    if len(bad) == 1 and not suspect:
        selected = tuple(sorted(all_beams - bad))
        return [("THREE_BEAM", selected, "unique_hard_failure")]
    if bad:
        available = tuple(sorted(all_beams - bad))
        if len(available) == 3:
            return [("THREE_BEAM", available, "unique_hard_failure")]
        return []
    specs: list[tuple[str, tuple[int, ...], str]] = [("FOUR_BEAM", (1, 2, 3, 4), "ambiguous_four_beam")]
    for omitted in sorted(suspect):
        specs.append(("THREE_BEAM", tuple(sorted(all_beams - {omitted})), f"ambiguous_omit_b{omitted}"))
    return specs


def _vector_payload(vector) -> dict:
    return {
        "x": float(vector[0]),
        "y": float(vector[1]),
        "z": float(vector[2]),
        "unit": "m/s",
        "frame": None,
    }


def _vector(vector, frame: str) -> dict:
    payload = _vector_payload(vector)
    payload["frame"] = frame
    return payload


def _event(
    event_id: str,
    event_type: str,
    severity: str,
    subject_id: str,
    rationale: dict,
    ping_id: str | None = None,
) -> dict:
    return {
        "id": event_id,
        "type": event_type,
        "severity": severity,
        "subject_id": subject_id,
        "ping_id": ping_id,
        "rationale": rationale,
    }


def _source_chain(quality: dict, beam_numbers: tuple[int, ...], depth_cell: dict) -> list[str]:
    beam_by_number = {item["beam_number"]: item for item in quality["beams"]}
    chain = [depth_cell["id"]]
    for beam_number in beam_numbers:
        item = beam_by_number[beam_number]
        chain.extend(
            [
                item["velocity_id"],
                item["correlation_id"],
                item["amplitude_id"],
                item["flag_id"],
            ]
        )
    return chain


def _solve_ping(
    ping_id: str,
    time_epoch_s: float,
    depth_cell: dict,
    quality: dict,
    radial: dict,
    depth_excluded: bool,
    geometry_matrix,
    geometry_fingerprint: str,
    attitude: dict,
    mount: dict,
    declination: dict,
    transform_order: tuple[str, ...],
    rule_version: str,
) -> tuple[list[dict], list[dict], list[dict]]:
    candidates = []
    events = []
    diagnostics = []
    if depth_excluded:
        return candidates, events, diagnostics
    specs = _candidate_specs(quality["beams"])
    ambiguity = any(spec[2].startswith("ambiguous") for spec in specs)
    if ambiguity:
        events.append(
            _event(
                f"EV-{ping_id}-AMBIGUITY",
                "MULTIPLE_LEGAL_CANDIDATES",
                "warning",
                ping_id,
                {
                    "reason": "THREE_BEAM_AND_FOUR_BEAM_CANDIDATES_REMAIN_DISTINCT",
                    "candidate_count": len(specs),
                    "suspect_beams": [item["beam_number"] for item in quality["beams"] if item["state"] == "suspect"],
                },
                ping_id,
            )
        )
    for candidate_type, beam_numbers, basis in specs:
        values = {number: radial[number] for number in beam_numbers if number in radial}
        if len(values) != len(beam_numbers):
            diagnostics.append(
                {
                    "id": f"DIAG-{ping_id}-{candidate_type}-{'-'.join(str(number) for number in beam_numbers)}",
                    "ping_id": ping_id,
                    "code": "INSUFFICIENT_VALID_BEAMS",
                    "status": "unresolved",
                    "source_chain": _source_chain(quality, beam_numbers, depth_cell),
                }
            )
            continue
        instrument_vector = solve_beams(values, geometry_matrix, {number: number - 1 for number in beam_numbers})
        source_chain = _source_chain(quality, beam_numbers, depth_cell)
        input_fingerprint = fingerprint(
            {
                "radial_m_s": {str(number): values[number] for number in beam_numbers},
                "beam_numbers": beam_numbers,
                "geometry": geometry_fingerprint,
                "candidate_type": candidate_type,
                "basis": basis,
            }
        )
        attitude_status = attitude.get("status")
        vectors = {
            "beam": None,
            "instrument": _vector(instrument_vector, "instrument:x-forward-y-starboard-z-down"),
            "ship": None,
            "earth": None,
        }
        candidate_status = "instrument_only"
        attitude_fingerprint = None
        if attitude_status in ("EXACT", "INTERPOLATED"):
            transformed = transform_stages(
                instrument_vector,
                float(attitude["heading_deg"]),
                float(attitude["pitch_deg"]),
                float(attitude["roll_deg"]),
                float(mount["raw_installation_angle_z_deg"]),
                float(declination["raw_angle_deg"]),
                transform_order,
            )
            vectors["ship"] = _vector(transformed["ship"], "ship:x-forward-y-starboard-z-down")
            vectors["earth"] = _vector(transformed["earth"], "earth:east-north-up")
            candidate_status = "unresolved" if ambiguity else "adopted"
            attitude_fingerprint = fingerprint(
                {
                    "status": attitude_status,
                    "heading_deg": attitude["heading_deg"],
                    "pitch_deg": attitude["pitch_deg"],
                    "roll_deg": attitude["roll_deg"],
                    "left_id": attitude.get("left_id"),
                    "right_id": attitude.get("right_id"),
                    "fraction": attitude.get("fraction"),
                }
            )
            source_chain.extend([str(attitude.get("left_id")), str(attitude.get("right_id"))])
        else:
            diagnostics.append(
                {
                    "id": f"DIAG-{ping_id}-ATTITUDE-{candidate_type}-{'-'.join(str(number) for number in beam_numbers)}",
                    "ping_id": ping_id,
                    "code": attitude.get("diagnostic", "ATTITUDE_UNAVAILABLE"),
                    "status": "unresolved",
                    "left_id": attitude.get("left_id"),
                    "right_id": attitude.get("right_id"),
                    "gap_s": attitude.get("gap_s"),
                    "source_chain": source_chain,
                }
            )
        selected_beams = "-".join(str(number) for number in beam_numbers)
        candidate_id = f"CAND-{ping_id}-{depth_cell['bin_number']:02d}-{candidate_type.lower()}-B{selected_beams}"
        candidate_fingerprint = fingerprint(
            {
                "input": input_fingerprint,
                "attitude": attitude_fingerprint,
                "rules": rule_version,
                "transform_order": list(transform_order),
            }
        )
        adopted = candidate_status == "adopted"
        candidate = {
            "id": candidate_id,
            "stable_identity": candidate_id,
            "ping_id": ping_id,
            "time_epoch_s": time_epoch_s,
            "time_unit": "s",
            "depth_cell_id": depth_cell["id"],
            "solution_type": candidate_type,
            "basis": basis,
            "beam_numbers": list(beam_numbers),
            "status": candidate_status,
            "adopted": adopted,
            "quality": quality,
            "vectors": vectors,
            "attitude": attitude,
            "installation_alignment_id": mount["id"],
            "magnetic_declination_id": declination["id"],
            "magnetic_datum_version": declination["datum_version"],
            "transform_order": list(transform_order),
            "geometry_id": "JANUS-4-CONVEX-20DEG-v1",
            "source_chain": source_chain,
            "input_fingerprint": input_fingerprint,
            "attitude_fingerprint": attitude_fingerprint,
            "rule_fingerprint": rule_version,
            "candidate_fingerprint": candidate_fingerprint,
        }
        candidates.append(candidate)
        events.append(
            _event(
                f"EV-{candidate_id}-DERIVED",
                "VELOCITY_CANDIDATE_DERIVED",
                "info" if candidate_status == "adopted" else "warning",
                candidate_id,
                {
                    "solution_type": candidate_type,
                    "beam_numbers": list(beam_numbers),
                    "status": candidate_status,
                    "candidate_fingerprint": candidate_fingerprint,
                    "three_beam_not_mixed_with_four_beam": True,
                },
                ping_id,
            )
        )
    return candidates, events, diagnostics


def _solve_bottom_track(
    bottom_track: list[dict],
    events: list[dict],
    diagnostics: list[dict],
    ping_id: str,
    time_epoch_s: float,
    attitude: dict,
    records: list[dict],
    geometry_matrix,
    geometry_fingerprint: str,
    mount: dict,
    declination: dict,
    transform_order: tuple[str, ...],
    rule_version: str,
) -> None:
    radial = {}
    source_chain = []
    for record in sorted(records, key=lambda item: item["beam_number"]):
        normalized = normalize_number(record["raw_value"], record["raw_unit"], "m/s")
        if normalized.normalized is not None:
            radial[record["beam_number"]] = normalized.normalized
        source_chain.append(record["id"])
    if len(radial) == 4:
        beam_numbers = (1, 2, 3, 4)
        solution_type = "FOUR_BEAM"
    elif len(radial) == 3:
        beam_numbers = tuple(sorted(radial))
        solution_type = "THREE_BEAM"
    else:
        diagnostics.append(
            {
                "id": f"DIAG-{ping_id}-BOTTOM-TRACK",
                "ping_id": ping_id,
                "code": "BOTTOM_TRACK_INSUFFICIENT_BEAMS",
                "status": "unresolved",
                "source_chain": source_chain,
            }
        )
        return
    instrument_vector = solve_beams(radial, geometry_matrix, {number: number - 1 for number in beam_numbers})
    vectors = {
        "instrument": _vector(instrument_vector, "instrument:x-forward-y-starboard-z-down"),
        "ship": None,
        "earth": None,
    }
    status = "instrument_only"
    attitude_fingerprint = None
    if attitude.get("status") in ("EXACT", "INTERPOLATED"):
        transformed = transform_stages(
            instrument_vector,
            float(attitude["heading_deg"]),
            float(attitude["pitch_deg"]),
            float(attitude["roll_deg"]),
            float(mount["raw_installation_angle_z_deg"]),
            float(declination["raw_angle_deg"]),
            transform_order,
        )
        vectors["ship"] = _vector(transformed["ship"], "ship:x-forward-y-starboard-z-down")
        vectors["earth"] = _vector(transformed["earth"], "earth:east-north-up")
        status = "adopted"
        attitude_fingerprint = fingerprint({"attitude": attitude, "geometry": geometry_fingerprint})
        source_chain.extend([str(attitude.get("left_id")), str(attitude.get("right_id"))])
    candidate_id = f"CAND-{ping_id}-BOTTOM-{solution_type.lower()}-B{'-'.join(str(number) for number in beam_numbers)}"
    candidate_fingerprint = fingerprint(
        {
            "source_chain": source_chain,
            "rule_fingerprint": rule_version,
            "attitude_fingerprint": attitude_fingerprint,
            "kind": "bottom_track",
        }
    )
    bottom_track.append(
        {
            "id": f"BT-{ping_id}",
            "stable_identity": f"BT-{ping_id}",
            "ping_id": ping_id,
            "time_epoch_s": time_epoch_s,
            "time_unit": "s",
            "solution_type": solution_type,
            "beam_numbers": list(beam_numbers),
            "status": status,
            "adopted": status == "adopted",
            "vectors": vectors,
            "attitude": attitude,
            "source_chain": source_chain,
            "candidate_fingerprint": candidate_fingerprint,
            "rule_fingerprint": rule_version,
            "installation_alignment_id": mount["id"],
            "magnetic_declination_id": declination["id"],
            "magnetic_datum_version": declination["datum_version"],
        }
    )
