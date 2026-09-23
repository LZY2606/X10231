"""依赖图重算引擎:按 ping/cell 逐级派生,指纹驱动增量复用。

节点链(每 ping、每 depth cell 独立):
  config / attitude
    -> pingN:att
    -> pingN:cellC:beam -> :instr -> :ship -> :earth
    -> pingN:profile

指纹逐级累积:
  instr  包含波束几何版本;
  ship   包含安装角值及其来源 id;
  earth  包含磁偏角值+模型版本、坐标转换顺序、姿态指纹。
所以安装角修订只使 ship 及下游失效;波束几何/磁偏角版本/转换顺序任一
改变都会让对应派生指纹变化。多候选(三波束/四波束、多解)平行保留,
不被压成单一值。
"""
from __future__ import annotations

import json

from .fingerprint import fp
from .transforms import (
    beam_matrix,
    beam_to_instrument,
    bracket_wraps,
    candidate_id,
    instrument_to_ship,
    interp_series,
    magnetic_to_true,
    ship_to_earth,
    unwrap_headings,
)

SINGLETON_KINDS = (
    "transducer_config",
    "mounting_angle",
    "magnetic_declination",
    "transform_order",
    "quality_thresholds",
)


def assemble(sources):
    """把分源记录重组成配置、姿态序列与按 ping/cell 寻址的测量结构。"""
    cfg = {}
    headings, pitchs, rolls = [], [], []
    pings = {}

    def ping_cell(v, kind):
        p = pings.setdefault(v["ping"], {"cells": {}})
        c = p["cells"].setdefault(v["cell"], {})
        c.setdefault(kind, {})[v["beam"]] = v

    for s in sources:
        if s.get("superseded"):
            continue
        v = json.loads(s["value"]) if s["value"] else None
        k = s["kind"]
        if k in SINGLETON_KINDS:
            cfg[k] = dict(v or {}, __id__=s["id"])
        elif k == "heading_sample":
            headings.append((v["t"], v["value"], s["id"]))
        elif k == "pitch_sample":
            pitchs.append((v["t"], v["value"], s["id"]))
        elif k == "roll_sample":
            rolls.append((v["t"], v["value"], s["id"]))
        elif k == "ping":
            p = pings.setdefault(v["n"], {"cells": {}})
            p["t"], p["src"] = v["t"], s["id"]
        elif k in ("beam_velocity", "correlation", "amplitude"):
            rec = dict(v, __id__=s["id"], __raw__=s["raw"],
                       __diagnostic__=s.get("diagnostic"))
            ping_cell(rec, k)
        elif k == "bad_beam_flag":
            p = pings.setdefault(v["ping"], {"cells": {}})
            p.setdefault("bad_beams", {})[v["beam"]] = {
                "id": s["id"], "reason": v.get("reason", "")}
        elif k == "bottom_track":
            p = pings.setdefault(v["ping"], {"cells": {}})
            p["bt"] = {"id": s["id"], "raw": s["raw"],
                       "vec": [v["u"], v["v"], v["w"]]}

    headings.sort()
    pitchs.sort()
    rolls.sort()
    attitude = {
        "t": [x[0] for x in headings],
        "heading": [x[1] for x in headings],
        "pitch": [x[1] for x in pitchs],
        "roll": [x[1] for x in rolls],
        "ids": [x[2] for x in headings],
    }
    return cfg, attitude, pings


def _quality(measure, table, beam, key, default=None):
    rec = measure.get(table, {}).get(beam)
    return rec.get(key, default) if rec else default


def build_nodes(sources, overrides=(), previous=None):
    """重算全部节点;previous 为上一版本节点时,按指纹决定复用/重算。"""
    cfg, att, pings = assemble(sources)
    tc = cfg["transducer_config"]
    decl = cfg["magnetic_declination"]
    mount = cfg["mounting_angle"]
    order = cfg["transform_order"]["order"]
    qt = cfg["quality_thresholds"]
    gap_thr = qt["gap_threshold_s"]
    marginal_band = qt.get("marginal_band", 16)
    geom = beam_matrix(tc["beam_angle_deg"], tc["beam_azimuths_deg"])

    nodes, events = {}, []

    def emit(kind, **payload):
        events.append({"kind": kind, "payload": payload})

    geom_fp = fp({"geom": tc["geometry_version"],
                  "angle": tc["beam_angle_deg"],
                  "az": tc["beam_azimuths_deg"]})
    cfg_fp = fp({
        "geom": geom_fp, "order": order,
        "decl": [decl["value_deg"], decl["model"], decl["__id__"]],
        "mount": [mount["angle_deg"], mount["__id__"]],
        "thr": [qt["correlation_min"], qt["amplitude_min"], gap_thr],
    })
    nodes["config"] = {
        "fingerprint": cfg_fp, "status": "ok",
        "deps": [tc["__id__"], decl["__id__"], mount["__id__"]],
        "payload": {
            "geometry_version": tc["geometry_version"],
            "beam_angle_deg": tc["beam_angle_deg"],
            "beam_azimuths_deg": tc["beam_azimuths_deg"],
            "transform_order": order,
            "declination": {"value_deg": decl["value_deg"],
                            "model": decl["model"], "source": decl["__id__"]},
            "mounting_angle_deg": mount["angle_deg"],
            "mounting_source": mount["__id__"],
        },
    }

    unwrapped = unwrap_headings(att["heading"])
    att_ids = att["ids"] + [s for s in []]
    att_fp = fp({"ids": att_ids, "raw": list(zip(att["t"], att["heading"]))})
    nodes["attitude"] = {
        "fingerprint": att_fp, "status": "ok", "deps": att_ids,
        "payload": {"t": att["t"], "heading_mag": att["heading"],
                    "heading_unwrapped": unwrapped,
                    "pitch": att["pitch"], "roll": att["roll"]},
    }

    override_map = {(o["ping"], o["cell"]): o for o in (overrides or [])}

    for n in sorted(pings):
        p = pings[n]
        t = p["t"]
        akey = f"ping{n}:att"
        wrap = bracket_wraps(att["t"], att["heading"], t, gap_thr)
        hw, eh = interp_series(att["t"], unwrapped, t, gap_thr)
        pw, _ = interp_series(att["t"], att["pitch"], t, gap_thr)
        rw, _ = interp_series(att["t"], att["roll"], t, gap_thr)
        if hw is None:
            afp = fp({"att": att_fp, "t": t, "status": "attitude_gap"})
            nodes[akey] = {"fingerprint": afp, "status": "attitude_gap",
                           "deps": [att_fp], "payload": {"t": t, "diagnostic": eh}}
            emit("attitude_gap_no_interpolation", ping=n, t=t,
                 gap_threshold_s=gap_thr,
                 judgment="姿态插值不跨缺口,缺口内 ping 不产生 earth 值")
        else:
            if wrap:
                emit("heading_unwrapped_360", ping=n, t=t, bracket=wrap,
                     unwrapped_heading=hw,
                     judgment="先解卷再插值,航向不被 0/360 跳变带偏")
            heading_true = magnetic_to_true(hw % 360.0, decl["value_deg"])
            afp = fp({"att": att_fp, "t": round(t, 6),
                      "h": round(heading_true, 6), "p": round(pw, 6),
                      "r": round(rw, 6),
                      "decl": [decl["value_deg"], decl["model"]]})
            nodes[akey] = {
                "fingerprint": afp, "status": "ok", "deps": [att_fp],
                "payload": {"t": t, "heading_mag_unwrapped": hw,
                            "heading_true": heading_true, "pitch": pw,
                            "roll": rw,
                            "declination": {"value_deg": decl["value_deg"],
                                            "model": decl["model"]}}}
        _build_cells(nodes, emit, n, p, t, akey, geom, geom_fp, tc, mount, decl,
                     order, qt, marginal_band, override_map)

    recomputed, reused = [], []
    if previous:
        prev_fp = {k: v["fingerprint"] for k, v in previous.items()}
        for key, node in nodes.items():
            if prev_fp.get(key) == node["fingerprint"]:
                reused.append(key)
            else:
                recomputed.append(key)
    emit("recompute_summary",
         judgment="依赖图按指纹传播:指纹相同说明输入与规则未变,结果仍有效;"
                  "不同说明直接或间接依赖已修订,必须重算",
         recomputed=sorted(recomputed), reused=sorted(reused))
    return nodes, events


def _build_cells(nodes, emit, n, p, t, akey, geom, geom_fp, tc, mount, decl,
                 order, qt, marginal_band, override_map):
    bad_flags = p.get("bad_beams", {})
    an = nodes[akey]
    cell_earth_keys = []

    for c in sorted(p["cells"]):
        cell = p["cells"][c]
        bkey = f"ping{n}:cell{c}:beam"
        ikey = f"ping{n}:cell{c}:instr"
        skey = f"ping{n}:cell{c}:ship"
        ekey = f"ping{n}:cell{c}:earth"

        vals, raws, src_ids, diags = [], {}, [], {}
        bad = set()
        for b in range(4):
            bv = cell.get("beam_velocity", {}).get(b)
            if bv is None:
                vals.append(None)
                raws[b] = None
                continue
            src_ids.append(bv["__id__"])
            raws[b] = bv["__raw__"]
            if bv.get("value") is None:
                vals.append(None)
                diags[b] = bv.get("__diagnostic__") or "unconvertible"
                emit("unconvertible_value_kept", ping=n, cell=c, beam=b,
                     source=bv["__id__"], raw=bv["__raw__"], diagnostic=diags[b],
                     judgment="无法转换的值留在原位置并附诊断,不为画图丢弃")
            else:
                vals.append(bv["value"])
            if b in bad_flags:
                bad.add(b)

        corr = [_quality(cell, "correlation", b, "value") for b in range(4)]
        amp = [_quality(cell, "amplitude", b, "value") for b in range(4)]
        corr_ids = [cell.get("correlation", {}).get(b, {}).get("__id__")
                    for b in range(4) if cell.get("correlation", {}).get(b)]
        amp_ids = [cell.get("amplitude", {}).get(b, {}).get("__id__")
                   for b in range(4) if cell.get("amplitude", {}).get(b)]
        reasons = []
        if any(x is not None and x < qt["correlation_min"] for x in corr):
            reasons.append("low_correlation")
        if any(x is not None and x < qt["amplitude_min"] for x in amp):
            reasons.append("low_amplitude")
        excluded = bool(reasons)
        bfp = fp({"src": sorted(src_ids), "raw": raws, "bad": sorted(bad),
                  "corr": corr, "amp": amp})
        nodes[bkey] = {
            "fingerprint": bfp,
            "status": "excluded" if excluded else "ok",
            "deps": sorted(src_ids + [bad_flags[b]["id"] for b in bad]),
            "payload": {
                "beam_vel_m_s": vals, "raw_mm_s": raws,
                "bad_beams": sorted(bad),
                "bad_beam_sources": {str(b): bad_flags[b]["id"] for b in bad},
                "correlation_counts": corr, "correlation_sources": corr_ids,
                "amplitude_db": amp, "amplitude_sources": amp_ids,
                "excluded": excluded, "reasons": reasons,
                "unconvertible": diags,
            },
        }
        if excluded:
            emit("cell_excluded", ping=n, cell=c, reasons=reasons,
                 correlation=corr, amplitude=amp,
                 judgment="无效 depth cell 留在剖面位置并标证据,不参与拟合")

        cands = _instrument_candidates(vals, bad, corr, qt, marginal_band,
                                       geom, n, c, emit)
        for cand in cands:
            cand["id"] = candidate_id(cand)
        if excluded:
            istatus = "excluded"
        elif len(cands) > 1:
            istatus = "ambiguous"
        elif len(cands) == 1:
            istatus = "three_beam" if cands[0]["kind"] == "three_beam" else "ok"
        else:
            istatus = "insufficient_beams"
        ifp = fp({"beam": bfp, "geom": geom_fp})
        nodes[ikey] = {"fingerprint": ifp, "status": istatus, "deps": [bfp],
                       "payload": {"candidates": cands,
                                   "geometry_version": tc["geometry_version"]}}

        ship_cands = []
        for cand in cands:
            ship_cands.append({
                "id": cand["id"],
                "vec": instrument_to_ship(cand["vec"], mount["angle_deg"]),
            })
        sfp = fp({"instr": ifp,
                  "mount": [mount["angle_deg"], mount["__id__"]]})
        nodes[skey] = {
            "fingerprint": sfp, "status": istatus, "deps": [ifp],
            "payload": {"candidates": ship_cands,
                        "mounting_angle_deg": mount["angle_deg"],
                        "mounting_source": mount["__id__"]}}

        earth_cands = []
        if an["status"] == "ok":
            a = an["payload"]
            for sc in ship_cands:
                earth_cands.append({
                    "id": sc["id"],
                    "vec": ship_to_earth(sc["vec"], a["heading_true"],
                                         a["pitch"], a["roll"], order),
                })
            estatus = istatus
            ediag = None
        else:
            estatus = "attitude_gap"
            ediag = an["payload"].get("diagnostic")
        efp = fp({"ship": sfp, "att": an["fingerprint"],
                  "order": order,
                  "decl": [decl["value_deg"], decl["model"], decl["__id__"]]})
        nodes[ekey] = {
            "fingerprint": efp, "status": estatus, "deps": [sfp, an["fingerprint"]],
            "payload": {"candidates": earth_cands, "transform_order": order,
                        "declination": {"value_deg": decl["value_deg"],
                                        "model": decl["model"]},
                        "diagnostic": ediag}}
        cell_earth_keys.append((c, ekey, bkey))

    _profile(nodes, emit, n, t, p, cell_earth_keys, tc, override_map,
             bad_flags)


def _instrument_candidates(vals, bad, corr, qt, marginal_band, geom,
                           n, c, emit):
    cands = []
    missing = [b for b in range(4) if vals[b] is None]
    flagged = sorted(bad)
    suspects = sorted(set(flagged) | set(missing))
    marginal = [b for b in range(4)
                if b not in suspects and corr[b] is not None
                and qt["correlation_min"] <= corr[b]
                < qt["correlation_min"] + marginal_band]

    if not suspects and not marginal:
        cand = beam_to_instrument(vals, geom)
        if cand:
            cands.append(cand)
    elif len(suspects) == 1 and not marginal:
        cand = beam_to_instrument(vals, geom, drop=suspects)
        if cand:
            cands.append(cand)
            emit("three_beam_not_merged", ping=n, cell=c,
                 dropped=suspects, four_beam_available=False,
                 candidate=candidate_id(cand),
                 judgment="单 beam 失效:三波束解独立成候选,"
                          "不与四波束解混为一值")
    else:
        # 可疑(标志/缺失/临界相关)波束有多个 => 逐一丢弃形成全部合法候选,
        # 无法唯一决定时保留全部,不吞掉多解。
        tried = set()
        for b in suspects + marginal:
            if b in tried:
                continue
            tried.add(b)
            cand = beam_to_instrument(vals, geom, drop=[b])
            if cand:
                cands.append(cand)
        if len(cands) > 1:
            emit("ambiguous_candidates_kept", ping=n, cell=c,
                 suspects=suspects, marginal=marginal,
                 candidates=[candidate_id(x) for x in cands],
                 judgment="三波束多解:全部合法候选保留,剖面不强行选值")
    return cands


def _profile(nodes, emit, n, t, p, cell_earth_keys, tc, override_map,
             bad_flags):
    pkey = f"ping{n}:profile"
    out_cells = []
    deps = []
    for c, ekey, bkey in cell_earth_keys:
        en = nodes[ekey]
        bn = nodes[bkey]
        depth = tc["blanking_m"] + (c + 0.5) * tc["cell_size_m"]
        cand_ids = [cd["id"] for cd in en["payload"]["candidates"]]
        adopted, vec, st = None, None, en["status"]
        ov = override_map.get((n, c))

        if bn["payload"]["excluded"]:
            st = "excluded"
        elif en["payload"]["candidates"]:
            default = "four_beam" if "four_beam" in cand_ids else (
                cand_ids[0] if len(cand_ids) == 1 else None)
            if default and len(cand_ids) == 1 and default != "four_beam":
                st = "three_beam"
            if len(cand_ids) > 1:
                st = "ambiguous"
            if ov and ov.get("action") == "adopt" and ov.get("candidate") in cand_ids:
                adopted = ov["candidate"]
                st = f"adopted:{adopted}"
            elif ov and ov.get("action") == "reject" and ov.get("candidate") == default:
                adopted, st = None, "rejected"
                emit("manual_reject", ping=n, cell=c,
                     candidate=ov["candidate"], actor=ov.get("actor"))
            else:
                adopted = default
            if adopted:
                vec = next(cd["vec"] for cd in en["payload"]["candidates"]
                           if cd["id"] == adopted)
        out_cells.append({
            "cell": c, "depth_m": depth, "status": st, "adopted": adopted,
            "vec_m_s": vec, "candidates": cand_ids,
            "reasons": bn["payload"]["reasons"],
            "earth_fingerprint": en["fingerprint"],
        })
        deps.append(en["fingerprint"])

    ov_fp = sorted(
        f"{o.get('actor')}:{o.get('action')}:{o.get('candidate')}"
        for (pp, _cc), o in override_map.items() if pp == n)
    pfp = fp({"cells": deps, "t": round(t, 6), "overrides": ov_fp})
    nodes[pkey] = {
        "fingerprint": pfp, "status": "ok", "deps": deps,
        "payload": {"ping": n, "t": t, "cells": out_cells,
                    "bottom_track": p.get("bt")},
    }
