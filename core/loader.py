"""把案例包 JSON 载入为 ProfileSources；每类来源单独给出内容版本号。

版本号是该来源内容的规范化哈希，因此“只改动一个直接来源”时只有该来源版本变化，
依赖图传播范围也随之被限制在正确的下游集合。
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

from .fingerprint import fingerprint, short
from .geometry import TransducerConfig
from .model import PingCellRaw, PingRaw, ProfileSources, validate_snapshot
from .quality import QualityRules
from .rotation import MountingAngles


def source_version(kind: str, payload: Any) -> str:
    return f"{kind}-{short(fingerprint({'kind': kind, 'payload': payload}), 16)}"


def load_case(data: dict[str, Any], *, version_label: str | None = None) -> ProfileSources:
    diags = validate_snapshot(data)
    if diags:
        raise ValueError("invalid case pack: " + "; ".join(diags))
    pings = []
    for p in data["pings"]:
        cells = [PingCellRaw(depth_raw=c["depth_raw"], depth_unit=c.get("depth_unit", "m"),
                             invalid=bool(c.get("invalid", False))) for c in p["cells"]]
        pings.append(PingRaw(
            ping=int(p["ping"]), time_s=float(p["time_s"]),
            beam_velocity_raw=p["beam_velocity_raw"],
            velocity_unit=p.get("velocity_unit", "mm/s"),
            correlation=p["correlation"], amplitude=p["amplitude"],
            bad_beam=p["bad_beam"], cells=cells,
            bottom_track_raw=p.get("bottom_track_raw"),
            bottom_track_unit=p.get("bottom_track_unit", "mm/s"),
        ))
    attitude = {
        "times": data["attitude"]["times"],
        "heading_deg": data["attitude"]["heading_deg"],
        "pitch_deg": data["attitude"]["pitch_deg"],
        "roll_deg": data["attitude"]["roll_deg"],
    }
    geometry = TransducerConfig.from_dict(data["geometry"])
    mounting = MountingAngles.from_dict(data["mounting"])
    rules = QualityRules(**{k: v for k, v in data["quality_rules"].items()})

    versions = {
        "beam_velocity": source_version("beam_velocity",
                                        [p.beam_velocity_raw for p in pings]),
        "correlation": source_version("correlation", [p.correlation for p in pings]),
        "amplitude": source_version("amplitude", [p.amplitude for p in pings]),
        "bad_beam": source_version("bad_beam", [p.bad_beam for p in pings]),
        "depth_cell": source_version("depth_cell",
                                     [[(c.depth_raw, c.depth_unit, c.invalid) for c in p.cells]
                                      for p in pings]),
        "attitude": source_version("attitude", attitude),
        "declination": source_version("declination", data["declination"]),
        "mounting": source_version("mounting", data["mounting"]),
        "geometry": source_version("geometry", data["geometry"]),
        "bottom_track": source_version("bottom_track",
                                       [(p.bottom_track_raw, p.bottom_track_unit) for p in pings]),
    }
    return ProfileSources(
        pings=pings, attitude=attitude,
        declination={"value_deg": data["declination"]["value_deg"],
                     "unit": data["declination"].get("unit", "deg"),
                     "version": versions["declination"]},
        mounting=mounting, mounting_version=versions["mounting"],
        geometry=geometry, geometry_version=versions["geometry"],
        rules=rules, rules_version=source_version("quality_rules", data["quality_rules"]),
        transform_order=data.get("transform_order", "HPR"),
        transform_version=source_version("transform", data.get("transform_order", "HPR")),
        attitude_max_gap_s=float(data.get("attitude_max_gap_s", 2.0)),
        source_versions=versions,
    )


def load_case_file(path: str | pathlib.Path) -> tuple[dict[str, Any], ProfileSources]:
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return data, load_case(data)
