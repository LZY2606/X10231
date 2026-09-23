"""Counter-example archive scenarios.

Each scenario forks the baseline and changes exactly ONE direct source record
(one raw measurement, one attitude record or one configuration version).  The
scenario metadata declares the affected object and the machine-checked
expectations, so replay can prove the blast radius is correctly limited.
"""
from __future__ import annotations

import copy


def _fork(baseline, scenario_id, title, changed_source, description):
    raw = copy.deepcopy(baseline)
    raw["case_id"] = "flowbench-" + scenario_id
    return {
        "id": scenario_id,
        "title": title,
        "changed_source": changed_source,
        "description": description,
        "raw": raw,
        "expect": {},
    }


def all_scenarios(baseline):
    sc = []

    # 1. Single beam failure (hard bad flag) -> exact 3-beam solution.
    s = _fork(baseline, "s1-single-beam", "单 beam 失效",
              "bad_beam_flag p002 c02 beam3",
              "Only the bad-beam flag of ping p002 / cell c02 / beam 3 flips.")
    s["raw"]["pings"][2]["cells"]["c02"]["bad_beam_flag"][2] = True
    s["expect"] = {
        "node": "beam-solution:p002:c02", "status": "three_beam",
        "candidate_count": 1, "instrument_vector_present": True,
        "error_velocity_present": False,
        "unaffected_node": "earth:p000:c00",
    }
    sc.append(s)

    # 2. Heading crosses 360 deg at the interpolation bracket.
    s = _fork(baseline, "s2-heading-360", "姿态跨越三百六十度",
              "attitude sample p001 heading_deg 2.4 -> 1.2",
              "Only the magnetic heading record of ping p001 changes.")
    s["raw"]["attitude_samples"][1]["heading_deg"] = 1.2
    s["expect"] = {"unwrap_applied_at": "bt-earth:p001",
                   "interpolated_heading_deg": 359.9,
                   "naive_linear_would_be": 179.9}
    sc.append(s)

    # 3. Time gap in the attitude series: interpolation must not cross it.
    s = _fork(baseline, "s3-time-gap", "时间缺口",
              "attitude sample p005 time 5 -> 34",
              "Only the timestamp of the last attitude sample changes.")
    s["raw"]["attitude_samples"][5]["time"] = 34.0
    s["expect"] = {"blocked_status": "blocked-gap", "blocked_bt": "bt-earth:p004"}
    sc.append(s)

    # 4. All four beams pass quality but mutually disagree -> four legal
    #    3-beam candidates; they must never collapse into one value.
    s = _fork(baseline, "s4-threebeam-multisol", "三波束多解",
              "beam velocity p001 c01 beam2",
              "Only beam 2 velocity of ping p001 / cell c01 is perturbed.")
    cell = s["raw"]["pings"][1]["cells"]["c01"]
    cell["beam_velocity"][1]["raw"] = round(cell["beam_velocity"][1]["raw"] + 30.0, 4)
    s["expect"] = {"node": "beam-solution:p001:c01", "status": "ambiguous",
                   "candidate_count": 4, "instrument_vector_present": False}
    sc.append(s)

    # 5. Transducer alignment revision (mount heading offset).
    s = _fork(baseline, "s5-mount-revision", "安装角修订",
              "config mount_heading_deg 0 -> 2.5",
              "Only the mounting alignment version changes.")
    s["raw"]["config"]["mount_heading_deg"] = 2.5
    s["expect"] = {"changed_resource": "resource:mount",
                   "earth_fingerprint_changes": True, "quality_unchanged": True}
    sc.append(s)

    # 6. Coordinate transform order revision.
    s = _fork(baseline, "s6-transform-order", "坐标变换顺序",
              "config transform_order hpr-v1 -> rph-v1",
              "Only the rotation composition order version changes.")
    s["raw"]["config"]["transform_order"] = "rph-v1"
    s["expect"] = {"changed_resource": "resource:order",
                   "earth_fingerprint_changes": True,
                   "earth_vector_differs_from_baseline": True}
    sc.append(s)

    # 7. Explicitly invalid depth cell.
    s = _fork(baseline, "s7-invalid-cell", "无效深度 cell（显式标记）",
              "cell validity p001 c04",
              "Only the validity flag of ping p001 / cell c04 flips.")
    c = s["raw"]["pings"][1]["cells"]["c04"]
    c["valid"] = False
    c["reason"] = "below side-lobe interference threshold"
    s["expect"] = {"node": "cell-exclusion:p001:c04", "status": "excluded",
                   "no_earth_node_for_cell": True}
    sc.append(s)

    # 8. Depth that cannot be converted stays in place with a diagnostic.
    s = _fork(baseline, "s8-unconvertible-depth", "无法转换的深度留在原位置",
              "cell depth unit p000 c03 m -> fathom",
              "Only the raw depth representation of p000 / c03 changes.")
    c = s["raw"]["pings"][0]["cells"]["c03"]
    c["depth_m"] = {"raw": 3.3, "raw_unit": "fathom"}
    s["expect"] = {"node": "cell-exclusion:p000:c03", "status": "excluded",
                   "raw_preserved": {"raw": 3.3, "raw_unit": "fathom"}}
    sc.append(s)

    # 9. Magnetic declination revision (new model year/version).
    s = _fork(baseline, "s9-declination", "磁偏角版本修订",
              "config declination -7.5/wmm-2024 -> -7.0/wmm-2025",
              "Only the magnetic declination version changes.")
    s["raw"]["config"]["declination_deg"] = -7.0
    s["raw"]["config"]["declination_version"] = "wmm-2025-v1"
    s["expect"] = {"changed_resource": "resource:declination",
                   "earth_fingerprint_changes": True}
    sc.append(s)

    return sc
