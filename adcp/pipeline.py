"""依赖图组装：从存储层当前快照读取分开记录的源流，构建 ping 上下文并调用规则层。

每个派生结果带 basis（所依赖的流版本/几何版本/磁偏角版本/安装角版本/顺序）与指纹。
重算只输出受输入影响的部分；指纹不变则上层可判定“仍然有效、无需重算”。
"""
from __future__ import annotations

import json
from typing import Any

from .attitude import interpolate_attitude
from .beam import BeamGeometry
from .fingerprint import derive_fingerprint
from importlib.metadata import version as _pkg_version

import adcp
from . import transforms as T

RULES_VERSION = f"adcp-rules-{adcp.__version__}"


def _one(records: list[dict], kind: str) -> dict | None:
    rows = [r for r in records if r["kind"] == kind]
    return rows[0] if rows else None


def build_geometry(state: dict[str, Any]) -> tuple[BeamGeometry, str]:
    rec = _one(state["records"], "geom")
    payload = rec["payload"]
    geom = BeamGeometry(
        beam_angle_deg=float(payload["beam_angle_deg"]),
        azimuths_deg=tuple(float(a) for a in payload["azimuths_deg"]),
        n_beams=int(payload.get("n_beams", 4)),
    )
    return geom, f'{rec["stream_id"]}@v{rec["version"]}:{geom.version_id()}'


def _streams_version_map(state: dict[str, Any]) -> dict[str, int]:
    return {r["stream_id"]: r["version"] for r in state["records"]}


def recompute(state: dict[str, Any]) -> list[dict[str, Any]]:
    """读取当前状态，返回全部派生对象（cell_flow / profile / bottom_track_flow）。"""
    geom, geom_ref = build_geometry(state)
    grid = _one(state["records"], "cellgrid")["payload"]
    decl = _one(state["records"], "declination")
    mount = _one(state["records"], "mount")
    decl_deg = float(decl["payload"]["angle_deg"])
    mount_angles = (float(mount["payload"]["heading_deg"]),
                    float(mount["payload"]["pitch_deg"]),
                    float(mount["payload"]["roll_deg"]))
    order = grid.get("transform_order", "rdi_xyz")

    att_samples = []
    for r in state["records"]:
        if r["kind"] == "attitude":
            p = r["payload"]
            att_samples.append({"id": r["stream_id"].split("stm-")[-1], "time_s": float(p["time_s"]),
                                "heading_deg": float(p["heading_raw"]),
                                "pitch_deg": float(p["pitch_raw"]),
                                "roll_deg": float(p["roll_raw"])})
    att_samples.sort(key=lambda s: s["time_s"])

    # 按 ping 分组
    ping_ids = sorted({r["payload"]["ping_id"] for r in state["records"]
                       if r["kind"] in ("beam_vel", "bottom_track", "bottom_depth")})
    by: dict[tuple[str, str, str], dict] = {}
    for r in state["records"]:
        if r["kind"] in ("beam_vel", "correlation", "amplitude", "badflag"):
            by[(r["payload"]["ping_id"], r["payload"]["cell_id"], r["kind"])] = r
    bt_by: dict[str, dict] = {r["payload"]["ping_id"]: r for r in state["records"] if r["kind"] == "bottom_track"}
    depth_by: dict[str, dict] = {r["payload"]["ping_id"]: r for r in state["records"] if r["kind"] == "bottom_depth"}
    vmap = _streams_version_map(state)
    decl_ref = f'{decl["stream_id"]}@v{decl["version"]}'
    mount_ref = f'{mount["stream_id"]}@v{mount["version"]}'

    derivations: list[dict[str, Any]] = []
    depths = [float(d) for d in grid["depths_m"]]
    for ping_id in ping_ids:
        t = float(ping_id.split("-")[-1])
        att = interpolate_attitude(ping_id, t, att_samples, gap_threshold_s=float(grid["gap_threshold_s"]))
        bdepth = float(depth_by[ping_id]["payload"]["depth_m"]) if ping_id in depth_by else None
        ctx = {
            "ping_id": ping_id, "time_s": t, "geometry": geom, "geom_version": geom_ref,
            "attitude": att, "attitude_events": att.events,
            "declination_deg": decl_deg, "decl_version": decl_ref,
            "mount": mount_angles, "mount_version": mount_ref,
            "transform_order": order,
            "blanking_m": float(grid["blanking_m"]), "min_depth_m": float(grid["min_depth_m"]),
            "corr_threshold_pct": float(grid["corr_threshold_pct"]),
            "errvel_threshold_mps": float(grid["errvel_threshold_mps"]),
            "bottom_depth_m": bdepth, "depth_frame": grid.get("depth_frame", "transducer_down"),
            "cells": [],
        }
        for depth in depths:
            cell_id = f"{ping_id}-cell{int(round(depth*10)):03d}"
            cell = {"cell_id": cell_id, "depth_m": depth}
            for kind, field_name in (("beam_vel", "beam_vel"), ("correlation", "correlation"),
                                     ("amplitude", "amplitude"), ("badflag", "bad_flags")):
                rec = by.get((ping_id, cell_id, kind))
                p = rec["payload"] if rec else {}
                if kind == "beam_vel":
                    cell["beam_vel"] = list(p.get("beams", [None] * 4))
                    cell["beam_vel_unit"] = p.get("unit", "m/s")
                elif kind == "correlation":
                    cell["correlation"] = list(p.get("values", [None] * 4))
                    cell["correlation_unit"] = p.get("unit", "pct")
                elif kind == "amplitude":
                    cell["amplitude"] = list(p.get("values", [None] * 4))
                    cell["amplitude_unit"] = p.get("unit", "count")
                else:
                    cell["bad_flags"] = list(p.get("flags", [None] * 4))
            ctx["cells"].append(cell)

        profile = T.compute_ping_cells(ctx)
        cell_versions = sorted({f"{rec['stream_id']}@v{rec['version']}"
                                for rec in state["records"]
                                if rec["payload"].get("ping_id") == ping_id
                                and rec["kind"] in ("beam_vel", "correlation", "amplitude", "badflag")})
        att_versions = sorted(
            f'{r["stream_id"]}@v{r["version"]}' for r in state["records"]
            if r["kind"] == "attitude"
            and r["stream_id"].split("stm-")[-1] in (att.bracketed_by or ()))
        basis = {**profile["basis"],
                 "rules_version": RULES_VERSION,
                 "used_attitude_versions": att_versions,
                 "grid_version": f'{_one(state["records"], "cellgrid")["stream_id"]}@v{_one(state["records"], "cellgrid")["version"]}',
                 "cell_versions": cell_versions}
        # 每个 cell 一条独立派生
        for cell in profile["cells"]:
            fp = derive_fingerprint("cell_flow", {"result": _stable_cell(cell), "basis": basis})
            derivations.append({"derivation_id": cell["cell_id"], "kind": "cell_flow",
                                "ping_id": ping_id, "cell_id": cell["cell_id"],
                                "payload": cell, "fingerprint": fp, "basis": basis})
        derivations.append({"derivation_id": f"{ping_id}-profile", "kind": "profile",
                            "ping_id": ping_id, "cell_id": None,
                            "payload": {"ping_id": ping_id, "attitude_status": profile["attitude_status"],
                                        "attitude": profile["attitude"], "events": profile["events"]},
                            "fingerprint": derive_fingerprint("profile",
                                                              {"cells": [c["cell_id"] for c in profile["cells"]],
                                                               "basis": basis, "att": profile["attitude_status"]}),
                            "basis": basis})
        if ping_id in bt_by:
            btr = bt_by[ping_id]["payload"]
            bctx = dict(ctx)
            bctx["bottom_track"] = {
                "bt_id": btr.get("bt_id", f"{ping_id}-bt"),
                "beam_vel": list(btr.get("beams", [None] * 4)),
                "beam_vel_unit": btr.get("unit", "m/s"),
                "correlation": list(btr.get("correlation", [None] * 4)),
                "correlation_unit": btr.get("correlation_unit", "pct"),
                "amplitude": list(btr.get("amplitude", [None] * 4)),
                "amplitude_unit": btr.get("amplitude_unit", "count"),
                "bad_flags": list(btr.get("bad_flags", [0, 0, 0, 0])),
                "range_m": btr.get("range_m"),
            }
            bt_result = T.compute_bottom_track(bctx)
            bt_basis = {**bt_result["basis"], "rules_version": RULES_VERSION,
                        "bt_version": f'{bt_by[ping_id]["stream_id"]}@v{bt_by[ping_id]["version"]}'}
            derivations.append({"derivation_id": f"{ping_id}-bt", "kind": "bottom_track_flow",
                                "ping_id": ping_id, "cell_id": None,
                                "payload": bt_result, "fingerprint": derive_fingerprint("bt", {"result": bt_result, "basis": bt_basis}),
                                "basis": bt_basis})
    return derivations


def _stable_cell(cell: dict) -> dict:
    # 指纹只看数值结论与其候选，不看展示层
    return {"cell_id": cell["cell_id"], "depth_m": cell["depth_m"],
            "candidates": cell["candidates"], "excluded": cell["excluded"],
            "exclude_reasons": cell["exclude_reasons"], "ambiguous": cell["ambiguous"]}


def impacted_stream_kinds(changes: list[dict]) -> set[str]:
    return {c["kind"] for c in changes}


def explain_basis(basis: dict, current_versions: dict[str, str]) -> list[dict]:
    """对每条 basis 引用给出 still_valid / stale 判定。"""
    out = []
    for key, ref in basis.items():
        if key in ("rules_version", "transform_order", "errvel_threshold_mps", "corr_threshold_pct"):
            out.append({"input": key, "ref": ref, "state": "fixed" if key != "rules_version" else "code"})
            continue
        refs = ref if isinstance(ref, list) else [ref]
        for one_ref in refs:
            out.append({"input": key, "ref": one_ref,
                        "state": "still_valid" if current_versions.get(one_ref) is not None else "stale"})
    return out
