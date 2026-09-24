"""Beam quality hierarchy and candidate generation.

Levels, from raw to decided:

1. raw flag        instrument-reported bad-beam mark (kept verbatim);
2. measured        correlation / amplitude thresholds (versioned rule set);
3. availability    bad + measured -> GOOD / SUSPECT / BAD per beam;
4. candidates      four-beam solution and every legal three-beam triplet;
5. decision        an accepted/rejected human or automatic verdict.

Invalid depth cells (blanking / bottom / marked bad bin) are excluded as
whole cells: they remain visible with their reason instead of disappearing
from the profile.  Three-beam and four-beam solutions are never merged into
one value; when uniqueness fails, all legal candidates survive.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .geometry import solve_four_beam, solve_three_beam_candidates, BeamGeometry

GOOD, SUSPECT, BAD = "good", "suspect", "bad"


@dataclass(frozen=True)
class QualityRules:
    rules_id: str
    correlation_min: float = 80.0
    amplitude_min: float = 40.0
    correlation_suspect_band: float = 15.0
    amplitude_suspect_band: float = 8.0
    max_attitude_gap_s: float = 2.0

    def content(self) -> dict:
        return {
            "rules_id": self.rules_id,
            "correlation_min": self.correlation_min,
            "amplitude_min": self.amplitude_min,
            "correlation_suspect_band": self.correlation_suspect_band,
            "amplitude_suspect_band": self.amplitude_suspect_band,
            "max_attitude_gap_s": self.max_attitude_gap_s,
        }


@dataclass
class CellQuality:
    bin_index: int
    status: str
    per_beam: list[dict] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)


@dataclass
class CandidateSet:
    bin_index: int | None
    candidates: list[dict]
    diagnostics: list[str]
    cell_status: str


def classify_beam(
    beam_index: int,
    raw_bad: bool,
    correlation: float | None,
    amplitude: float | None,
    rules: QualityRules,
) -> dict:
    diagnostics: list[str] = []
    if raw_bad:
        diagnostics.append(f"beam{beam_index}:raw-bad-flag")
        return {"beam": beam_index, "status": BAD, "diagnostics": diagnostics,
                "correlation": correlation, "amplitude": amplitude}
    if correlation is None or amplitude is None:
        diagnostics.append(f"beam{beam_index}:missing-quality-measurement")
        return {"beam": beam_index, "status": BAD, "diagnostics": diagnostics,
                "correlation": correlation, "amplitude": amplitude}

    status = GOOD
    if correlation < rules.correlation_min or amplitude < rules.amplitude_min:
        status = BAD
        if correlation < rules.correlation_min:
            diagnostics.append(f"beam{beam_index}:correlation-low")
        if amplitude < rules.amplitude_min:
            diagnostics.append(f"beam{beam_index}:amplitude-low")
    elif (correlation < rules.correlation_min + rules.correlation_suspect_band
          or amplitude < rules.amplitude_min + rules.amplitude_suspect_band):
        status = SUSPECT
        diagnostics.append(f"beam{beam_index}:marginal-quality")
    return {"beam": beam_index, "status": status, "diagnostics": diagnostics,
            "correlation": correlation, "amplitude": amplitude}


def evaluate_cell(
    bin_index: int | None,
    radial: list[float | None],
    raw_bad: list[bool],
    correlation: list[float | None],
    amplitude: list[float | None],
    invalid_cell: bool,
    invalid_reason: str | None,
    geometry: BeamGeometry,
    rules: QualityRules,
) -> CandidateSet:
    """Produce the full quality record and all legal velocity candidates."""
    per_beam = [
        classify_beam(i, raw_bad[i], correlation[i], amplitude[i], rules)
        for i in range(geometry.n_beams)
    ]
    diagnostics: list[str] = []
    if invalid_cell:
        reason = invalid_reason or "invalid-depth-cell"
        diagnostics.append(f"cell:{reason}")
        return CandidateSet(bin_index, [], diagnostics, "excluded")

    good_beams = [b["beam"] for b in per_beam if b["status"] == GOOD]
    suspect_beams = [b["beam"] for b in per_beam if b["status"] == SUSPECT]
    usable = sorted(good_beams + suspect_beams)
    complete = [b for b in usable if radial[b] is not None]
    candidates: list[dict] = []

    if len(complete) == geometry.n_beams:
        solution = solve_four_beam(geometry, np_radial(radial))
        solution["assumptions"] = []
        candidates.append(solution)
        if suspect_beams:
            diagnostics.append("candidates:ambiguous-suspect-beams")
    elif len(complete) >= 3:
        diagnostics.append("candidates:no-four-beam-solution")

    three_beam = solve_three_beam_candidates(
        geometry, np_radial(radial, fill=0.0), complete
    )
    # When all four beams are good the triplets add no legal ambiguity.
    if not suspect_beams and len(complete) != geometry.n_beams:
        candidates.extend(three_beam)
        if len(three_beam) > 1:
            diagnostics.append("candidates:multiple-three-beam-solutions")
    elif suspect_beams:
        # Ambiguous case: keep every triplet that omits at least one suspect.
        for candidate in three_beam:
            if any(b not in candidate["used_beams"] for b in suspect_beams):
                candidates.append(candidate)
        if len(candidates) > 1:
            diagnostics.append("candidates:ambiguous-suspect-beams")

    if len(complete) < 3:
        diagnostics.append("candidates:insufficient-beams")
        return CandidateSet(bin_index, [], diagnostics, "no-solution")

    if len(candidates) == 1:
        return CandidateSet(bin_index, candidates, diagnostics, "unique")
    return CandidateSet(bin_index, candidates, diagnostics, "ambiguous")


def np_radial(radial: list[float | None], fill: float = 0.0):
    import numpy as np

    return np.array([fill if value is None else value for value in radial], dtype=float)
