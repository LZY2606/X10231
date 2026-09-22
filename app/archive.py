"""Offline ADCP case archive (the "beam-conversion bench").

Not a generic table: beam velocities, correlation, amplitude, bad-beam flags,
heading/pitch/roll, magnetic declination and mount angles are seeded and stored
as *separate* sources. Baseline beam velocities are forward-modelled from an
analytic current through the documented beam geometry, so reviewers can verify
the instrument/ship/earth reconstruction round-trips.

Each replay step mutates exactly one direct source and re-derives; accepted and
rejected results are both kept. Scenario set:
  s1 single beam failure
  s2 heading crossing 360
  s3 attitude time gap
  s4 three-beam ambiguity (both legal candidates retained)
  s5 mount-angle revision
  s6 coordinate transform order variant
  s7 invalid depth cell / beyond bottom track
  s8 conflicting human decision (resolvable)
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np

from . import geometry as geom
from .pipeline import derive_set

HERE = os.path.dirname(__file__)
CASE_DIR = os.path.join(HERE, "case_pack")

N_PINGS = 12
N_BINS = 6
PING_S = 1.0
BASE_DECLINATION = 7.5
MOUNT0 = {"mount_heading_deg": 0.0, "mount_pitch_deg": 0.0,
          "mount_roll_deg": 0.0}
MOUNT1 = {"mount_heading_deg": 2.0, "mount_pitch_deg": 0.0,
          "mount_roll_deg": 0.0}


def true_current(t: float, bin_index: int) -> np.ndarray:
    """Analytic earth-frame current (mm/s): shear + slow time evolution."""
    depth_speed = 600.0 - 35.0 * bin_index
    dir_deg = 25.0 + 8.0 * math.sin(0.13 * t)
    d = math.radians(dir_deg)
    east = depth_speed * math.cos(d) + 40.0 * math.sin(0.2 * t + bin_index)
    north = depth_speed * math.sin(d)
    up = -12.0 + 3.0 * math.sin(0.31 * t + 0.5 * bin_index)
    return np.array([east, north, up])


def attitude_true(t: float) -> Dict[str, float]:
    return {"heading_deg": 300.0 + 5.5 * t,
            "pitch_deg": 2.0 * math.sin(0.25 * t),
            "roll_deg": 1.5 * math.cos(0.21 * t)}


def seed_baseline(store, branch: str = "main") -> Dict[str, Any]:
    """Seed the complete baseline; return a description dict."""
    cfg = geom.TransducerConfig()
    B = geom.beam_matrix(cfg)
    inv = np.linalg.pinv(B)
    from . import transforms as tf

    pings = ["p%02d" % i for i in range(N_PINGS)]
    store.put_source("index", "pings", {"pings": pings, "ping_period_s": PING_S},
                     unit="seconds", datum="ordered list", branch=branch)
    store.put_source("config", "transducer", cfg.to_dict(),
                     unit="mixed", datum="transducer geometry", branch=branch)
    store.put_source("declination", "site",
                     {"declination_deg": BASE_DECLINATION,
                      "convention": "signed east-positive",
                      "version": "IGRF-2025-demo"},
                     unit="deg", datum="magnetic->true", branch=branch)
    store.put_source("mount", "angles", dict(MOUNT0, revision="v0"),
                     unit="deg", datum="instrument->ship", branch=branch)
    store.put_source("config", "transform_order", {"order": "RDI_HRP"},
                     datum="Rz H * Rx roll * Ry pitch", branch=branch)

    for i, pid in enumerate(pings):
        t = i * PING_S
        att = attitude_true(t)
        h, p, r = att["heading_deg"], att["pitch_deg"], att["roll_deg"]
        Me = tf.earth_matrix(h, p, r, BASE_DECLINATION, "RDI_HRP")
        radials, corrs, amps, flags, depths = [], [], [], [], []
        bottom_range = 14.5 + 0.2 * math.sin(0.4 * t)
        for b in range(N_BINS):
            v_earth = true_current(t, b)
            v_inst = np.linalg.solve(Me, v_earth)  # mount = identity
            rvec = B @ v_inst
            radials.append([float(x) for x in rvec])
            corrs.append([200, 198, 205, 190])
            amps.append([120, 118, 122, 115])
            flags.append([False, False, False, False])
            depths.append(2.5 + 1.5 * b)
        store.put_source("beam", pid, {"t": t, "radials_mm_s": radials,
                                       "unit_original": "mm/s"},
                         unit="mm/s", datum="beam radial, +away", branch=branch)
        store.put_source("correlation", pid,
                         {"t": t, "correlation_counts": corrs},
                         unit="counts(0-255)", branch=branch)
        store.put_source("amplitude", pid,
                         {"t": t, "amplitude_counts": amps},
                         unit="counts", branch=branch)
        store.put_source("badflag", pid, {"t": t, "bad_beam": flags},
                         datum="boolean per beam", branch=branch)
        # two attitude samples per ping; baseline gap is small (1s)
        samples = [dict(t=t - 0.5, **attitude_true(t - 0.5)),
                   dict(t=t + 0.5, **attitude_true(t + 0.5))]
        store.put_source("attitude", pid, {"t": t, "samples": samples},
                         unit="deg", datum="heading/pitch/roll", branch=branch)
        store.put_source("bottomtrack", pid,
                         {"t": t, "bottom_range_m": bottom_range,
                          "bt_velocity_mm_s": [0.0, 0.0, 0.0]},
                         unit="m,mm/s", branch=branch)
        store.put_source("bins", pid,
                         {"t": t, "bin_depth_m": depths,
                          "cell_valid": [True] * N_BINS},
                         unit="m", branch=branch)

    versions = store.current_versions()
    set_id, nodes, summary = derive_set(store, versions, tag="baseline",
                                        branch=branch)
    store.add_decision("baseline", None, "accept:baseline",
                       {"note": "clean 4-beam baseline accepted"},
                       branch=branch, set_id=set_id)
    return {"set_id": set_id, "pings": pings, "summary": summary}


def _expected_version(store, kind, key):
    cur = store.get_source_current(kind, key)
    return cur["version_id"] if cur else None


def _rederive(store, tag, branch="main", order="RDI_HRP"):
    parent = store.current_set_id(branch)
    return derive_set(store, store.current_versions(), tag=tag, branch=branch,
                      parent_set_id=parent, transform_order=order)


def step_single_beam_failure(store, branch="main"):
    pid = "p03"
    cur = store.get_source_current("badflag", pid)
    flags = [row[:] for row in cur["payload"]["bad_beam"]]
    flags[2][1] = True
    store.put_source("badflag", pid, {"t": cur["payload"]["t"], "bad_beam": flags},
                     expected_version=cur["version_id"], datum="boolean per beam",
                     author="scenario:s1", branch=branch)
    set_id, nodes, summary = _rederive(store, "s1-single-beam-failure", branch)
    store.append_event("scenario", {"step": "s1", "name": "single_beam_failure",
                                    "set_id": set_id, "affected": [pid],
                                    "expectation": "3-beam nodes carry candidate"
                                                   " ids 3bA:pair(1,4) and 3bB,"
                                                   " never the 4-beam value",
                                    "summary": summary}, branch=branch)
    return {"set_id": set_id, "summary": summary, "affected": [pid]}


def step_heading_wrap(store, branch="main"):
    pid = "p05"
    cur = store.get_source_current("attitude", pid)
    samples = json.loads(json.dumps(cur["payload"]["samples"]))
    # push attitude across the 0/360 seam; circular interp must stay smooth
    for s in samples:
        s["heading_deg"] = 358.0 + (s["t"] - samples[0]["t"]) * 4.0
    samples[0]["heading_deg"] = 358.0
    samples[1]["heading_deg"] = 2.0
    store.put_source("attitude", pid, {"t": cur["payload"]["t"],
                                       "samples": samples},
                     expected_version=cur["version_id"], unit="deg",
                     datum="heading/pitch/roll", author="scenario:s2",
                     branch=branch)
    set_id, nodes, summary = _rederive(store, "s2-heading-crossing-360", branch)
    store.append_event("scenario", {"step": "s2",
                                    "name": "heading_crossing_360",
                                    "set_id": set_id, "affected": [pid],
                                    "expectation": "interpolated heading near 0,"
                                                   " no ~180 deg jump"},
                       branch=branch)
    return {"set_id": set_id, "summary": summary}


def step_attitude_gap(store, branch="main"):
    pid = "p07"
    cur = store.get_source_current("attitude", pid)
    samples = json.loads(json.dumps(cur["payload"]["samples"]))
    samples[0]["t"] = cur["payload"]["t"] - 5.0
    samples[1]["t"] = cur["payload"]["t"] + 5.0
    store.put_source("attitude", pid, {"t": cur["payload"]["t"],
                                       "samples": samples},
                     expected_version=cur["version_id"], unit="deg",
                     datum="heading/pitch/roll", author="scenario:s3",
                     branch=branch)
    set_id, nodes, summary = _rederive(store, "s3-attitude-gap", branch)
    store.append_event("scenario", {"step": "s3", "name": "attitude_gap",
                                    "set_id": set_id, "affected": [pid],
                                    "expectation": "earth velocity unresolved;"
                                                   " no interpolation across gap"},
                       branch=branch)
    return {"set_id": set_id, "summary": summary}


def step_three_beam_ambiguity(store, branch="main"):
    """Inject an inhomogeneous (sheared) perturbation so 3bA != 3bB."""
    pid = "p09"
    beam = store.get_source_current("beam", pid)
    radials = json.loads(json.dumps(beam["payload"]["radials_mm_s"]))
    for b in range(len(radials)):
        shear = 60.0 * math.exp(-0.5 * (b - 2) ** 2)
        radials[b][2] += shear
        radials[b][0] -= 0.5 * shear
    flag = store.get_source_current("badflag", pid)
    flags = [row[:] for row in flag["payload"]["bad_beam"]]
    for b in range(len(flags)):
        flags[b][3] = True
    store.put_source("beam", pid, {"t": beam["payload"]["t"],
                                   "radials_mm_s": radials,
                                   "unit_original": "mm/s"},
                     expected_version=beam["version_id"], unit="mm/s",
                     datum="beam radial, +away", author="scenario:s4",
                     branch=branch)
    store.put_source("badflag", pid, {"t": flag["payload"]["t"],
                                      "bad_beam": flags},
                     expected_version=flag["version_id"],
                     datum="boolean per beam", author="scenario:s4",
                     branch=branch)
    set_id, nodes, summary = _rederive(store, "s4-three-beam-multisolution",
                                       branch)
    store.append_event("scenario", {"step": "s4",
                                    "name": "three_beam_multisolution",
                                    "set_id": set_id, "affected": [pid],
                                    "expectation": "both 3bA:pair(1,2) and 3bB"
                                                   " retained with distinct ids"
                                                   " and values"},
                       branch=branch)
    return {"set_id": set_id, "summary": summary}


def step_mount_revision(store, branch="main"):
    cur = store.get_source_current("mount", "angles")
    store.put_source("mount", "angles", dict(MOUNT1, revision="v1"),
                     expected_version=cur["version_id"], unit="deg",
                     datum="instrument->ship", author="scenario:s5",
                     branch=branch)
    set_id, nodes, summary = _rederive(store, "s5-mount-revision", branch)
    store.append_event("scenario", {"step": "s5", "name": "mount_revision",
                                    "set_id": set_id, "affected": "all pings",
                                    "expectation": "only ship/earth nodes change"
                                                   " fingerprint; beam nodes keep"
                                                   " theirs"},
                       branch=branch)
    return {"set_id": set_id, "summary": summary}


def step_transform_order(store, branch="main"):
    cur = store.get_source_current("config", "transform_order")
    store.put_source("config", "transform_order", {"order": "ALT_PRH"},
                     expected_version=cur["version_id"],
                     datum="Ry pitch * Rx roll * Rz H", author="scenario:s6",
                     branch=branch)
    set_id, nodes, summary = _rederive(store, "s6-transform-order", branch,
                                       order="ALT_PRH")
    store.append_event("scenario", {"step": "s6", "name": "transform_order",
                                    "set_id": set_id,
                                    "expectation": "ALT_PRH fingerprints differ"
                                                   " from RDI_HRP; variant kept"},
                       branch=branch)
    return {"set_id": set_id, "summary": summary}


def step_invalid_depth_cell(store, branch="main"):
    pid = "p11"
    cur = store.get_source_current("bins", pid)
    payload = json.loads(json.dumps(cur["payload"]))
    payload["cell_valid"][5] = False
    payload.setdefault("diagnostics", []).append(
        {"bin": 5, "code": "BAD_CELL",
         "message": "cell flagged invalid by acquisition QC"})
    store.put_source("bins", pid, payload, expected_version=cur["version_id"],
                     unit="m", author="scenario:s7", branch=branch)
    set_id, nodes, summary = _rederive(store, "s7-invalid-depth-cell", branch)
    store.append_event("scenario", {"step": "s7",
                                    "name": "invalid_depth_cell",
                                    "set_id": set_id, "affected": [pid],
                                    "expectation": "bin 5 node excluded,"
                                                   " value retained but not"
                                                   " plotted as valid"},
                       branch=branch)
    return {"set_id": set_id, "summary": summary}


def step_conflicting_decision(store, branch="main"):
    """Two reviewers disagree on which 3-beam candidate to continue.

    Both decisions are recorded; resolution is an explicit, non-destructive
    event, and the rejected candidate remains in the archive.
    """
    set_id = store.current_set_id(branch)
    store.add_decision("3bA:pair(1,2)", "3bB",
                       "three-beam:p09",
                       {"reviewer": "hydrologist-A",
                        "rationale": "opposite-pair mean is the field SOP"},
                       branch=branch, set_id=set_id)
    conflicting = store.add_decision(
        "3bB", "3bA:pair(1,2)", "three-beam:p09",
        {"reviewer": "hydrologist-B",
         "rationale": "zero-error completion matches 4-beam convention"},
        branch=branch, set_id=set_id)
    store.append_event("conflict_recorded",
                       {"conflict_key": "three-beam:p09",
                        "positions": ["3bA:pair(1,2)", "3bB"],
                        "resolvable": True,
                        "note": "resolution creates a new event; no overwrite"},
                       branch=branch)
    return {"conflict_event": conflicting}


STEPS = [
    ("s1", step_single_beam_failure),
    ("s2", step_heading_wrap),
    ("s3", step_attitude_gap),
    ("s4", step_three_beam_ambiguity),
    ("s5", step_mount_revision),
    ("s6", step_transform_order),
    ("s7", step_invalid_depth_cell),
    ("s8", step_conflicting_decision),
]


def replay_all(store, branch="main"):
    baseline = seed_baseline(store, branch)
    results = [("baseline", baseline)]
    for sid, fn in STEPS:
        results.append((sid, fn(store, branch)))
    return results


def write_case_pack(store, path: str = CASE_DIR) -> str:
    """Persist the domain case pack as deterministic offline JSON fixtures."""
    os.makedirs(path, exist_ok=True)
    bundle = store.export_bundle()
    with open(os.path.join(path, "case_bundle.json"), "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2, sort_keys=True)
    manifest = {
        "title": "流束换算台：ADCP 波束速度、姿态旋转与质量层级",
        "pings": N_PINGS, "bins": N_BINS, "ping_period_s": PING_S,
        "beam_angle_deg": 20.0,
        "azimuths_deg": list(geom.DEFAULT_AZIMUTHS_DEG),
        "declination_deg": BASE_DECLINATION,
        "scenarios": [s for s, _ in STEPS],
        "semantics": [
            "three-beam and four-beam solutions never share one value",
            "attitude never interpolated across a gap",
            "transform order, beam geometry and declination version are"
            " fingerprinted",
            "all legal candidates retained when not uniquely determined",
        ],
        "bundle_id": bundle["bundle_id"],
    }
    with open(os.path.join(path, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
    return path
