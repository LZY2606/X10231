"""Materialised ADCP dataset model and the raw-preserving importer.

A *package* is the domain exchange object (used by the offline case pack):
beam velocities, correlation, amplitude, bad-beam flags, attitude, magnetic
declination, transducer mounting and bottom track each appear as separate
named sections.  Every measured number is a RawValue with a unit/datum.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .angles import Attitude, Declination, Mounting
from .geometry import (
    BeamGeometry,
    DEFAULT_AZIMUTHS_DEG,
    DEFAULT_BEAM_ANGLE_DEG,
)
from .quality import QualityThresholds
from .units import RawValue, normalise, convert


@dataclass
class BeamSample:
    ping: int
    cell: int
    beam: int
    raw: Dict[str, Any]
    value_mps: Optional[float]
    diagnostic: Optional[Dict[str, Any]] = None
    source_kind: str = "water"


@dataclass
class ScalarSample:
    ping: int
    cell: int
    beam: int
    raw: Dict[str, Any]
    value: Optional[float]
    diagnostic: Optional[Dict[str, Any]] = None


@dataclass
class AttitudeSample:
    ping: Optional[int]
    t: float
    raw: Dict[str, Any]
    attitude: Optional[Attitude]
    diagnostic: Optional[Dict[str, Any]] = None


@dataclass
class BottomRange:
    ping: int
    beam: int
    raw: Dict[str, Any]
    range_m: Optional[float]
    diagnostic: Optional[Dict[str, Any]] = None


@dataclass
class DatasetModel:
    dataset_id: str
    name: str
    beams: List[BeamSample] = field(default_factory=list)
    correlations: List[ScalarSample] = field(default_factory=list)
    amplitudes: List[ScalarSample] = field(default_factory=list)
    bad_flags: List[ScalarSample] = field(default_factory=list)
    attitudes: List[AttitudeSample] = field(default_factory=list)
    bottoms: List[BottomRange] = field(default_factory=list)
    declination: Optional[Declination] = None
    declination_raw: Optional[Dict[str, Any]] = None
    mounting: Mounting = field(default_factory=Mounting)
    mounting_raw: Optional[Dict[str, Any]] = None
    geometry: BeamGeometry = field(
        default_factory=lambda: BeamGeometry(
            azimuths_deg=DEFAULT_AZIMUTHS_DEG,
            beam_angle_deg=DEFAULT_BEAM_ANGLE_DEG,
        )
    )
    thresholds: QualityThresholds = field(default_factory=QualityThresholds)
    cell_depths_m: List[float] = field(default_factory=list)
    time_step_s: float = 1.0
    max_attitude_gap_s: float = 2.0
    declared_time_gaps: List[Tuple[float, float]] = field(default_factory=list)
    sidelobe_fraction: float = 0.06
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    transform_order: str = "HPR"
    earth_datum: str = "ENU"

    @property
    def pings(self) -> List[int]:
        return sorted({b.ping for b in self.beams})

    @property
    def cells(self) -> List[int]:
        return sorted({b.cell for b in self.beams})

    def beam_value(self, ping: int, cell: int, beam: int) -> Optional[float]:
        for b in self.beams:
            if b.ping == ping and b.cell == cell and b.beam == beam:
                return b.value_mps
        return None

    def _scalar(self, store: List[ScalarSample], ping, cell, beam) -> Optional[float]:
        for s in store:
            if s.ping == ping and s.cell == cell and s.beam == beam:
                return s.value
        return None

    def correlation(self, ping, cell, beam):
        v = self._scalar(self.correlations, ping, cell, beam)
        return None if v is None else int(v)

    def amplitude(self, ping, cell, beam):
        v = self._scalar(self.amplitudes, ping, cell, beam)
        return None if v is None else int(v)

    def bad_flag(self, ping, cell, beam) -> bool:
        v = self._scalar(self.bad_flags, ping, cell, beam)
        return bool(v) if v is not None else False

    def attitude_at(self, t: float):
        from .angles import interpolate_attitude
        samples = [a.attitude for a in self.attitudes if a.attitude is not None]
        return interpolate_attitude(
            samples, t, self.max_attitude_gap_s, self.declared_time_gaps
        )

    def bottom_depth(self, ping: int) -> Optional[float]:
        vals = [b.range_m for b in self.bottoms if b.ping == ping and b.range_m is not None]
        return sum(vals) / len(vals) if vals else None


def _raw_from(obj: Any) -> RawValue:
    if isinstance(obj, RawValue):
        return obj
    if isinstance(obj, dict):
        return RawValue(
            value=obj.get("value"),
            unit=obj.get("unit"),
            datum=obj.get("datum"),
            note=obj.get("note"),
        )
    return RawValue(value=obj, unit=None)


def _diag_to_dict(diag):
    return diag.to_dict() if diag is not None else None


def import_package(pkg: Dict[str, Any]) -> DatasetModel:
    """Convert an exchange package into a materialised model.

    Raw representations are retained on every sample.  Values that cannot be
    normalised stay at their position (``value_mps is None``) with diagnostics;
    the pipeline excludes rather than fabricates them.
    """
    ds_id = pkg["dataset_id"]
    geom_cfg = pkg.get("geometry", {})
    geometry = BeamGeometry(
        azimuths_deg=tuple(geom_cfg.get("azimuths_deg", DEFAULT_AZIMUTHS_DEG)),
        beam_angle_deg=geom_cfg.get("beam_angle_deg", DEFAULT_BEAM_ANGLE_DEG),
        radial_sign=geom_cfg.get("radial_sign", "away"),
        look_direction=geom_cfg.get("look_direction", "down"),
        convention_version=geom_cfg.get(
            "convention_version", "beam-geometry/1"
        ),
    )
    thr_cfg = pkg.get("thresholds", {})
    thresholds = QualityThresholds(
        correlation_min=thr_cfg.get("correlation_min", 64),
        amplitude_min=thr_cfg.get("amplitude_min", 30),
        amplitude_max=thr_cfg.get("amplitude_max", 250),
        correlation_marginal=thr_cfg.get("correlation_marginal", 80),
    )
    mounting_cfg = pkg.get("mounting", {})
    mounting = Mounting(
        heading_offset_deg=mounting_cfg.get("heading_offset_deg", 0.0),
        pitch_offset_deg=mounting_cfg.get("pitch_offset_deg", 0.0),
        roll_offset_deg=mounting_cfg.get("roll_offset_deg", 0.0),
        order=mounting_cfg.get("order", "HPR"),
        revision=mounting_cfg.get("revision", "rev-0"),
    )
    model = DatasetModel(
        dataset_id=ds_id,
        name=pkg.get("name", ds_id),
        geometry=geometry,
        thresholds=thresholds,
        mounting=mounting,
        mounting_raw=dict(mounting_cfg) if mounting_cfg else None,
        cell_depths_m=list(pkg.get("cell_depths_m", [])),
        time_step_s=pkg.get("time_step_s", 1.0),
        max_attitude_gap_s=pkg.get("max_attitude_gap_s", 2.0),
        declared_time_gaps=[tuple(g) for g in pkg.get("declared_time_gaps", [])],
        sidelobe_fraction=pkg.get("sidelobe_fraction", 0.06),
        transform_order=pkg.get("transform_order", "HPR"),
        earth_datum=pkg.get("earth_datum", "ENU"),
    )

    for row in pkg.get("beam_velocities", []):
        raw = _raw_from(row["raw"])
        qty, diag = normalise(raw, "m/s")
        model.beams.append(BeamSample(
            ping=int(row["ping"]), cell=int(row["cell"]), beam=int(row["beam"]),
            raw=raw.to_dict(), value_mps=qty.value if qty else None,
            diagnostic=_diag_to_dict(diag),
            source_kind=row.get("kind", "water"),
        ))
        if diag:
            model.diagnostics.append({"family": "beam", **diag.to_dict()})

    def _scalars(section, canonical_unit, target):
        for row in pkg.get(section, []):
            raw = _raw_from(row["raw"])
            qty, diag = normalise(raw, canonical_unit)
            target.append(ScalarSample(
                ping=int(row["ping"]), cell=int(row["cell"]), beam=int(row["beam"]),
                raw=raw.to_dict(), value=qty.value if qty else None,
                diagnostic=_diag_to_dict(diag),
            ))
            if diag:
                model.diagnostics.append({"family": section, **diag.to_dict()})

    # correlation/amplitude are raw instrument counts; unit "count" passes
    # through undeclared-style? counts ARE dimensionless counts: accept both
    # bare numbers and {"value": n, "unit": "count"}.
    def _counts(section, target):
        for row in pkg.get(section, []):
            raw = _raw_from(row["raw"])
            if raw.unit in (None, "count"):
                try:
                    val = int(float(raw.value))
                    diag = None
                except (TypeError, ValueError):
                    val, diag = None, None
                    _, d2 = convert(raw)
                    diag = d2
                target.append(ScalarSample(
                    ping=int(row["ping"]), cell=int(row["cell"]),
                    beam=int(row["beam"]), raw=raw.to_dict(),
                    value=val, diagnostic=_diag_to_dict(diag),
                ))
            else:
                qty, diag = normalise(raw, "m/s")
                target.append(ScalarSample(
                    ping=int(row["ping"]), cell=int(row["cell"]),
                    beam=int(row["beam"]), raw=raw.to_dict(),
                    value=qty.value if qty else None,
                    diagnostic=_diag_to_dict(diag),
                ))
            if diag:
                model.diagnostics.append({"family": section, **diag.to_dict()})

    _counts("correlations", model.correlations)
    _counts("amplitudes", model.amplitudes)

    for row in pkg.get("bad_beam_flags", []):
        raw = _raw_from(row["raw"])
        try:
            val = 1 if int(float(raw.value)) != 0 else 0
            diag = None
        except (TypeError, ValueError):
            val, d2 = None, convert(raw)[1]
            diag = d2
        model.bad_flags.append(ScalarSample(
            ping=int(row["ping"]), cell=int(row["cell"]), beam=int(row["beam"]),
            raw=raw.to_dict(), value=val, diagnostic=_diag_to_dict(diag),
        ))

    for row in pkg.get("attitudes", []):
        raw_block = row["raw"]
        t_raw = _raw_from(raw_block["time"])
        t_qty, t_diag = normalise(t_raw, "s")
        h_raw = _raw_from(raw_block["heading"])
        p_raw = _raw_from(raw_block["pitch"])
        r_raw = _raw_from(raw_block["roll"])
        h_qty, h_diag = normalise(h_raw, "deg")
        p_qty, p_diag = normalise(p_raw, "deg")
        r_qty, r_diag = normalise(r_raw, "deg")
        att = None
        if t_qty and h_qty and p_qty and r_qty:
            att = Attitude(
                t=t_qty.value,
                heading_deg=h_qty.value % 360.0,
                heading_datum=h_raw.datum or "magnetic",
                pitch_deg=p_qty.value,
                roll_deg=r_qty.value,
            )
        diag = next((d for d in (t_diag, h_diag, p_diag, r_diag) if d), None)
        model.attitudes.append(AttitudeSample(
            ping=row.get("ping"),
            t=t_qty.value if t_qty else float("nan"),
            raw=raw_block,
            attitude=att,
            diagnostic=_diag_to_dict(diag),
        ))
        if diag:
            model.diagnostics.append({"family": "attitude", **diag.to_dict()})

    decl = pkg.get("declination")
    if decl:
        raw_decl = _raw_from(decl["raw"]) if "raw" in decl else None
        model.declination_raw = dict(decl)
        model.declination = Declination(
            value_deg=float(decl["value_deg"]),
            model=decl.get("model", "WMM"),
            model_version=decl.get("model_version", "unknown"),
            epoch=decl.get("epoch"),
        )

    for row in pkg.get("bottom_tracks", []):
        raw = _raw_from(row["raw"])
        qty, diag = normalise(raw, "m")
        model.bottoms.append(BottomRange(
            ping=int(row["ping"]), beam=int(row["beam"]), raw=raw.to_dict(),
            range_m=qty.value if qty else None, diagnostic=_diag_to_dict(diag),
        ))
        if diag:
            model.diagnostics.append({"family": "bottom", **diag.to_dict()})

    return model
