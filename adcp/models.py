"""Typed in-memory view over separately-recorded source documents.

Every physical source is a distinct document with its own version:
  geometry | beam_velocity | correlation | amplitude | bad_beam |
  attitude | declination | mounting | bottom_track
Raw representations are retained (inside quantity dicts); normalisation to
canonical units happens here, in the derivation layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .units import parse_quantity

DOC_TYPES = (
    "geometry", "beam_velocity", "correlation", "amplitude", "bad_beam",
    "attitude", "declination", "mounting", "bottom_track",
)


@dataclass
class Document:
    doc_id: str
    doc_type: str
    version: int
    source: str
    content: dict[str, Any]
    fingerprint: str
    unit_system: str = "raw"

    @property
    def version_id(self) -> str:
        return f"{self.doc_id}@v{self.version}"


@dataclass
class Ruleset:
    rule_id: str
    rule_release: str
    transform_order: str
    earth_frame: str
    geometry_version: str
    declination_version: str
    mounting_version: str
    correlation_threshold: float
    amplitude_threshold: float
    max_attitude_gap_s: float
    valid_cell_depth_m: tuple[float, float]
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_release": self.rule_release,
            "transform_order": self.transform_order,
            "earth_frame": self.earth_frame,
            "geometry_version": self.geometry_version,
            "declination_version": self.declination_version,
            "mounting_version": self.mounting_version,
            "correlation_threshold": self.correlation_threshold,
            "amplitude_threshold": self.amplitude_threshold,
            "max_attitude_gap_s": self.max_attitude_gap_s,
            "valid_cell_depth_m": list(self.valid_cell_depth_m),
        }


@dataclass
class CellVelocity:
    ping: int
    cell: int
    time_s: float
    depth_m: float
    beams: dict[int, dict[str, Any]] = field(default_factory=dict)  # qty dicts


@dataclass
class Dataset:
    documents: dict[str, Document]
    geometry: dict[str, Any] = field(default_factory=dict)
    velocities: dict[tuple[int, int], CellVelocity] = field(default_factory=dict)
    correlation: dict[tuple[int, int], dict[int, Any]] = field(default_factory=dict)
    amplitude: dict[tuple[int, int], dict[int, Any]] = field(default_factory=dict)
    bad: dict[tuple[int, int], dict[int, bool]] = field(default_factory=dict)
    attitude: list[dict[str, Any]] = field(default_factory=list)
    declination: dict[str, Any] = field(default_factory=dict)
    mounting: dict[str, Any] = field(default_factory=dict)
    bottom: dict[int, dict[str, Any]] = field(default_factory=dict)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    @property
    def doc_versions(self) -> dict[str, str]:
        return {dt: self.documents[dt].version_id
                for dt in DOC_TYPES if dt in self.documents}

    @property
    def doc_hashes(self) -> dict[str, str]:
        return {dt: self.documents[dt].fingerprint
                for dt in DOC_TYPES if dt in self.documents}


def _qty(raw: Any, unit: str, datum: str | None = None) -> dict[str, Any]:
    return parse_quantity(raw, unit, datum).as_dict()


def assemble(documents: dict[str, Document]) -> Dataset:
    ds = Dataset(documents=dict(documents))
    if "geometry" in documents:
        ds.geometry = documents["geometry"].content

    bv = documents.get("beam_velocity")
    if bv:
        for e in bv.content["entries"]:
            cv = CellVelocity(e["ping"], e["cell"], e["time_s"], e["depth_m"])
            for b, raw in e["beams"].items():
                cv.beams[int(b)] = _qty(raw, bv.content.get("unit", "m/s"))
            ds.velocities[(cv.ping, cv.cell)] = cv

    for dtype, target, unit in (
        ("correlation", ds.correlation, "count"),
        ("amplitude", ds.amplitude, "count"),
    ):
        doc = documents.get(dtype)
        if doc:
            for e in doc.content["entries"]:
                target[(e["ping"], e["cell"])] = {
                    int(b): _qty(v, doc.content.get("unit", unit))
                    for b, v in e["beams"].items()}

    bad = documents.get("bad_beam")
    if bad:
        for e in bad.content["entries"]:
            ds.bad[(e["ping"], e["cell"])] = {
                int(b): bool(v) for b, v in e["beams"].items()}

    att = documents.get("attitude")
    if att:
        hu = att.content.get("heading_unit", "deg")
        for e in att.content["entries"]:
            ds.attitude.append({
                "time_s": e["time_s"],
                "ping": e.get("ping"),
                "heading": _qty(e["heading"], hu, att.content.get("heading_datum",
                                                                  "MAG")),
                "pitch": _qty(e["pitch"], att.content.get("pitch_unit", "deg")),
                "roll": _qty(e["roll"], att.content.get("roll_unit", "deg")),
            })

    dec = documents.get("declination")
    if dec:
        c = dec.content
        ds.declination = {
            "declination": _qty(c["declination_deg"], "deg",
                                c.get("datum", "MAG->TRUE")),
            "source": c.get("source"),
        }

    mt = documents.get("mounting")
    if mt:
        c = mt.content
        ds.mounting = {
            "alignment": _qty(c["alignment_deg"], "deg", "SHIP"),
            "pitch_offset": _qty(c.get("pitch_offset_deg", 0.0), "deg"),
            "roll_offset": _qty(c.get("roll_offset_deg", 0.0), "deg"),
        }

    bt = documents.get("bottom_track")
    if bt:
        for e in bt.content["entries"]:
            rec = {"ping": e["ping"], "time_s": e["time_s"],
                   "beams": {}, "ranges": {}}
            for b, raw in e.get("beams", {}).items():
                rec["beams"][int(b)] = _qty(raw, bt.content.get("unit", "m/s"))
            for b, raw in e.get("ranges", {}).items():
                rec["ranges"][int(b)] = _qty(raw, bt.content.get("range_unit", "m"))
            ds.bottom[e["ping"]] = rec
    return ds
