"""派生依赖图:beam_quality -> attitude -> instrument/ship/earth -> profile。

- 每个节点有独立指纹;输入指纹不变则复用旧结果(仍有效),变化才重算。
- 三波束/四波束解作为不同 kind 的候选贯穿全部层级,不混为一值。
- 无法转换的值留在原位置并附诊断,不丢数据。
"""
from __future__ import annotations

import json
from collections import defaultdict

from .transform import (
    BeamGeometry,
    derivation_fingerprint,
    instrument_to_ship,
    ship_to_earth,
    solve_four_beam,
    solve_three_beam,
    stable_hash,
    interp_attitude,
    unwrap_headings,
)

NODES = [
    "beam_quality",
    "attitude_series",
    "attitude_interp",
    "instrument_vel",
    "ship_vel",
    "earth_vel",
    "profile",
]

DEPS = {
    "beam_quality": ["src:beam_velocity", "src:correlation", "src:amplitude", "src:beam_flag"],
    "attitude_series": ["src:attitude"],
    "attitude_interp": ["attitude_series"],
    "instrument_vel": ["beam_quality", "cfg:beam_geometry"],
    "ship_vel": ["instrument_vel", "src:install_angle"],
    "earth_vel": ["ship_vel", "attitude_interp", "src:declination", "cfg:transform_order"],
    "profile": ["earth_vel", "beam_quality"],
}

UNIT_TO_MS = {"m/s": 1.0, "cm/s": 0.01, "mm/s": 0.001}


def source_view(raw_sources):
    """raw_sources: {table: [{skey, revision, payload}]};按 skey 取最高版本,跳过 deleted。"""
    view = {}
    revisions = {}
    for table, rows in raw_sources.items():
        latest = {}
        for row in rows:
            key = row["skey"]
            if key not in latest or row["revision"] > latest[key]["revision"]:
                latest[key] = row
        payloads = {}
        rev = 0
        for key, row in latest.items():
            payload = row["payload"]
            rev = max(rev, row["revision"])
            if isinstance(payload, str):
                payload = json.loads(payload)
            if payload.get("deleted"):
                continue
            payloads[key] = payload
        view[table] = payloads
        revisions[table] = rev
    return view, revisions


def source_fingerprints(revisions):
    return {table: f"{table}@r{rev}" for table, rev in revisions.items()}


def _index_pbc(rows):
    out = {}
    for p in rows.values():
        out[(p["ping"], p["beam"], p["cell"])] = p
    return out


def _beam_quality(sources, config, states):
    vel = _index_pbc(sources["beam_velocity"])
    corr = _index_pbc(sources["correlation"])
    amp = _index_pbc(sources["amplitude"])
    flag = _index_pbc(sources["beam_flag"])
    corr_min = config["corr_min"]
    amp_lo, amp_hi = config["amp_min"], config["amp_max"]
    pings = sorted({k[0] for k in set(vel) | set(corr) | set(amp) | set(flag)})
    cells = sorted({k[2] for k in set(vel) | set(corr) | set(amp) | set(flag)})
    out = {}
    for ping in pings:
        cells_out = {}
        for cell in cells:
            beams_out = {}
            usable = []
            for beam in range(4):
                v = vel.get((ping, beam, cell))
                co = corr.get((ping, beam, cell))
                am = amp.get((ping, beam, cell))
                fl = flag.get((ping, beam, cell))
                reasons = []
                flag_value = fl["flag"] if fl else "good"
                corr_value = co["value"] if co else None
                amp_value = am["value"] if am else None
                vel_ms = None
                vel_id = v["id"] if v else None
                raw = v.get("raw") if v else None
                unit = v.get("unit") if v else None
                if v is not None:
                    factor = UNIT_TO_MS.get(v.get("unit"))
                    if factor is None:
                        reasons.append("unit_unconvertible")
                    else:
                        vel_ms = v["value"] * factor
                if flag_value == "bad":
                    reasons.append("bad_beam_flag")
                if corr_value is not None and corr_value < corr_min:
                    reasons.append("low_correlation")
                if amp_value is not None and not (amp_lo <= amp_value <= amp_hi):
                    reasons.append("amplitude_out_of_range")
                if flag_value == "suspect":
                    reasons.append("suspect_beam")
                ok = not reasons or reasons == ["suspect_beam"]
                if ok:
                    usable.append(beam)
                beams_out[str(beam)] = {
                    "id": vel_id,
                    "flag_id": fl["id"] if fl else None,
                    "raw": raw,
                    "unit": unit,
                    "velocity_ms": vel_ms,
                    "correlation": corr_value,
                    "correlation_id": co["id"] if co else None,
                    "amplitude": amp_value,
                    "amplitude_id": am["id"] if am else None,
                    "flag": flag_value,
                    "ok": ok,
                    "reasons": reasons,
                }
            excluded = len(usable) < 3
            cells_out[str(cell)] = {"beams": beams_out, "usable": usable, "excluded": excluded}
        out[str(ping)] = {"cells": cells_out}
    return {"pings": out}


def _attitude_series(sources, config, states):
    rows = list(sources["attitude"].values())
    valid = [r for r in rows if r.get("heading") is not None]
    valid.sort(key=lambda r: r["t"])
    headings = unwrap_headings([r["heading"] for r in valid])
    samples = [
        {"t": r["t"], "heading": h, "pitch": r["pitch"], "roll": r["roll"], "id": r["id"]}
        for r, h in zip(valid, headings)
    ]
    max_gap = config["max_attitude_gap_s"]
    gaps = []
    for a, b in zip(samples, samples[1:]):
        if b["t"] - a["t"] > max_gap:
            gaps.append([a["t"], b["t"]])
    return {"samples": samples, "gaps": gaps}


def _attitude_interp(sources, config, states):
    series = states["attitude_series"]["payload"]
    samples = series["samples"]
    max_gap = config["max_attitude_gap_s"]
    t0, dt, n_pings = config["ping_t0"], config["ping_dt"], config["n_pings"]
    out = {}
    for ping in range(n_pings):
        t = t0 + dt * ping
        att = interp_attitude(samples, t, max_gap)
        if att is None:
            out[str(ping)] = {
                "t": t,
                "status": "no_attitude",
                "diag": "姿态样本缺口或超出样本范围,不跨缺口插值",
            }
        else:
            out[str(ping)] = {"t": t, "status": "ok", **att}
    return {"pings": out}


def _instrument_vel(sources, config, states):
    geom = BeamGeometry(config["beam_angle_deg"], tuple(config["azimuths_deg"]))
    quality = states["beam_quality"]["payload"]["pings"]
    out = {}
    for ping_str, pdata in quality.items():
        cells_out = {}
        for cell_str, cell in pdata["cells"].items():
            beams = cell["beams"]
            usable = [b for b in range(4) if beams[str(b)]["ok"]]
            suspects = [b for b in usable if beams[str(b)]["flag"] == "suspect"]
            bad = [b for b in range(4) if b not in usable]
            solutions = []
            status = "ok"
            diag = ""
            if len(usable) < 3:
                status, diag = "excluded", f"可用波束 {len(usable)} < 3:{bad}"
            elif len(usable) == 3:
                drop = bad[0]
                bvals = [beams[str(b)]["velocity_ms"] for b in usable]
                sol, err = solve_three_beam(geom, usable, bvals)
                entry = {"kind": "three_beam", "candidate": 0, "dropped": drop}
                if sol is None:
                    entry.update(status="undetermined", diag=err)
                    status, diag = "undetermined", err
                else:
                    entry.update(vec=[sol["u"], sol["v"], sol["w"]], status="ok")
                    if suspects:
                        status = "ambiguous"
                        diag = "存在可疑 beam,但不足以构造额外候选,保留单一候选并标记"
                solutions.append(entry)
            else:
                bvals = [beams[str(b)]["velocity_ms"] for b in range(4)]
                four = solve_four_beam(geom, bvals)
                solutions.append(
                    {
                        "kind": "four_beam",
                        "candidate": 0,
                        "vec": [four["u"], four["v"], four["w"]],
                        "err": four["err"],
                        "status": "ok",
                    }
                )
                if suspects:
                    candidates = [
                        {
                            "kind": "three_beam",
                            "candidate": i + 1,
                            "dropped": s,
                        }
                        for i, s in enumerate(suspects)
                    ]
                    for cand in candidates:
                        used = [b for b in range(4) if b != cand["dropped"]]
                        sol, err = solve_three_beam(
                            geom, used, [beams[str(b)]["velocity_ms"] for b in used]
                        )
                        if sol is None:
                            cand.update(status="undetermined", diag=err)
                        else:
                            cand.update(vec=[sol["u"], sol["v"], sol["w"]], status="ok")
                    solutions.extend(candidates)
                    status = "ambiguous"
                    diag = f"可疑 beam {suspects}:四波束与多个三波束候选全部保留,不合并"
            cells_out[cell_str] = {"status": status, "diag": diag, "solutions": solutions}
        out[ping_str] = {"cells": cells_out}
    return {"pings": out}


def _ship_vel(sources, config, states):
    install_rows = sources["install_angle"]
    install = list(install_rows.values())[0]
    inst = states["instrument_vel"]["payload"]["pings"]
    out = {}
    for ping_str, pdata in inst.items():
        cells_out = {}
        for cell_str, cell in pdata["cells"].items():
            sols = []
            for sol in cell["solutions"]:
                if sol.get("status") != "ok":
                    sols.append(sol)
                    continue
                vec = instrument_to_ship(sol["vec"], install)
                sols.append({**sol, "vec": [float(vec[0]), float(vec[1]), float(vec[2])]})
            cells_out[cell_str] = {"status": cell["status"], "diag": cell["diag"], "solutions": sols}
        out[ping_str] = {"cells": cells_out}
    return {"pings": out}


def _earth_vel(sources, config, states):
    decl = list(sources["declination"].values())[0]
    order = config["transform_order"]
    ship = states["ship_vel"]["payload"]["pings"]
    atts = states["attitude_interp"]["payload"]["pings"]
    out = {}
    for ping_str, pdata in ship.items():
        att = atts[ping_str]
        cells_out = {}
        for cell_str, cell in pdata["cells"].items():
            if att["status"] != "ok":
                cells_out[cell_str] = {
                    "status": "no_attitude",
                    "diag": att["diag"],
                    "solutions": [],
                    "ship_solutions_held": len(cell["solutions"]),
                }
                continue
            att_vec = {"heading": att["heading"], "pitch": att["pitch"], "roll": att["roll"]}
            sols = []
            for sol in cell["solutions"]:
                if sol.get("status") != "ok":
                    sols.append(sol)
                    continue
                vec = ship_to_earth(sol["vec"], att_vec, order, decl["degrees"])
                sols.append({**sol, "vec": [float(vec[0]), float(vec[1]), float(vec[2])]})
            cells_out[cell_str] = {
                "status": cell["status"],
                "diag": cell["diag"],
                "solutions": sols,
                "datum": "ENU/true",
                "declination_version": decl["version"],
                "transform_order": order,
            }
        out[ping_str] = {"cells": cells_out}
    return {"pings": out}


def _profile(sources, config, states):
    earth = states["earth_vel"]["payload"]["pings"]
    quality = states["beam_quality"]["payload"]["pings"]
    blank, size, n_cells = config["blank_m"], config["cell_size_m"], config["n_cells"]
    cells_out = {}
    for cell in range(n_cells):
        cell_str = str(cell)
        sums = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
        excluded = []
        for ping_str in sorted(earth, key=int):
            node = earth[ping_str]["cells"].get(cell_str)
            q = quality[ping_str]["cells"][cell_str]
            if q["excluded"]:
                reasons = sorted(
                    {r for b in q["beams"].values() for r in b["reasons"] if r != "suspect_beam"}
                )
                excluded.append({"ping": int(ping_str), "reasons": reasons})
                continue
            if node is None or node["status"] == "no_attitude":
                excluded.append({"ping": int(ping_str), "reasons": ["no_attitude"]})
                continue
            for sol in node["solutions"]:
                if sol.get("status") != "ok":
                    continue
                key = f"{sol['kind']}#{sol.get('candidate', 0)}"
                acc = sums[key]
                acc[0] += sol["vec"][0]
                acc[1] += sol["vec"][1]
                acc[2] += sol["vec"][2]
                acc[3] += 1
        by_kind = {}
        for key, (su, sv, sw, n) in sorted(sums.items()):
            by_kind[key] = {"u": su / n, "v": sv / n, "w": sw / n, "n": n}
        cells_out[cell_str] = {
            "depth_m": blank + size * (cell + 1),
            "by_kind": by_kind,
            "excluded": excluded,
        }
    return {"cells": cells_out}


COMPUTE = {
    "beam_quality": _beam_quality,
    "attitude_series": _attitude_series,
    "attitude_interp": _attitude_interp,
    "instrument_vel": _instrument_vel,
    "ship_vel": _ship_vel,
    "earth_vel": _earth_vel,
    "profile": _profile,
}


def compute_all(raw_sources, config, base_states=None):
    """运行依赖图。返回 {node: {fingerprint, inputs, action, why, payload}}。"""
    sources, revisions = source_view(raw_sources)
    src_fps = source_fingerprints(revisions)
    geom = BeamGeometry(config["beam_angle_deg"], tuple(config["azimuths_deg"]))
    cfg_fps = {"beam_geometry": geom.fingerprint(), "transform_order": config["transform_order"]}
    decl = list(sources["declination"].values())[0]
    install = list(sources["install_angle"].values())[0]
    states = {}
    fps = {}
    for node in NODES:
        comps = {}
        for dep in DEPS[node]:
            if dep.startswith("src:"):
                comps[dep] = src_fps[dep[4:]]
            elif dep.startswith("cfg:"):
                comps[dep] = cfg_fps[dep[4:]]
            else:
                comps[dep] = fps[dep]
        if node == "earth_vel":
            comps["declination_version"] = decl["version"]
        fp = stable_hash({"node": node, "inputs": comps})
        fps[node] = fp
        base = (base_states or {}).get(node)
        if base and base["fingerprint"] == fp:
            states[node] = {
                "fingerprint": fp,
                "inputs": comps,
                "action": "reused",
                "why": "输入指纹未变化,结果仍然有效",
                "payload": base["payload"],
            }
            continue
        changed = []
        if base:
            for key, value in comps.items():
                if base.get("inputs", {}).get(key) != value:
                    changed.append(key)
        else:
            changed = ["<baseline>"]
        payload = COMPUTE[node](sources, config, states)
        states[node] = {
            "fingerprint": fp,
            "inputs": comps,
            "action": "recomputed",
            "why": f"输入变化需要重算: {', '.join(changed)}",
            "payload": payload,
        }
    derivation_fp = derivation_fingerprint(
        transform_order=config["transform_order"],
        geom=geom,
        declination=decl,
        install={"heading_deg": install["heading_deg"], "pitch_deg": install["pitch_deg"], "roll_deg": install["roll_deg"]},
        source_fps=src_fps,
    )
    return states, derivation_fp
