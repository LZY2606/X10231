"""Beam quality hierarchy and multi-candidate construction.

A cell is never collapsed to one number just because a number would be
convenient:

* 4 valid beams  -> one 4-beam least-squares value (with error velocity)
                    plus four distinct 3-beam values kept as references.
* error velocity exceeds policy -> the 4-beam value is "suspect" and every
                    3-beam solution is a legal, non-unique candidate.
* 1 invalid beam -> one legal 3-beam value; the 4-beam value using the bad
                    beam is excluded with a diagnostic (never averaged in).
* <= 2 valid beams -> undetermined, the cell carries no velocity.

Bad-beam flags, correlation and amplitude are evaluated independently and
all reasons are retained.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from .geometry import TransducerConfig, solve_radial

BAD_FLAG = "bad"
LOW_CORR = "low_correlation"
LOW_AMP = "low_amplitude"
UNCONVERTIBLE = "unconvertible_amplitude"
ADOPTED = "adopted"
REFERENCE = "reference"
SUSPECT = "suspect"
EXCLUDED = "excluded"
UNDETERMINED = "undetermined"


@dataclass(frozen=True)
class QualityPolicy:
    corr_min_counts: int = 80      # 0..255 scale
    amp_min_counts: int = 60
    err_velocity_max: float = 0.05 # m/s


@dataclass
class BeamObservation:
    beam: int
    radial_si: float | None        # beam velocity normalised to m/s
    bad: bool
    corr_counts: float | None
    amp_counts: float | None
    amp_raw: str | None = None
    amp_unit: str | None = None
    reasons: list[str] = field(default_factory=list)


def evaluate_beams(observations: list[BeamObservation],
                   policy: QualityPolicy) -> list[BeamObservation]:
    for obs in observations:
        if obs.bad:
            obs.reasons.append(BAD_FLAG)
        if obs.radial_si is None:
            if BAD_FLAG not in obs.reasons:
                obs.reasons.append(BAD_FLAG)
        if obs.corr_counts is None:
            obs.reasons.append("missing_correlation")
        elif obs.corr_counts < policy.corr_min_counts:
            obs.reasons.append(LOW_CORR)
        if obs.amp_counts is None:
            # Unconvertible raw value is retained in place, not dropped.
            obs.reasons.append(UNCONVERTIBLE if obs.amp_raw not in (None, "") else "missing_amplitude")
        elif obs.amp_counts < policy.amp_min_counts:
            obs.reasons.append(LOW_AMP)
    return observations


def _valid_beams(observations: list[BeamObservation]) -> tuple[int, ...]:
    return tuple(obs.beam for obs in observations
                 if not obs.reasons or all(r == UNCONVERTIBLE for r in obs.reasons))


def build_cell_candidates(cfg: TransducerConfig,
                          observations: list[BeamObservation],
                          policy: QualityPolicy) -> dict:
    """Return {candidates, primary_candidate_id, status, diagnostics}."""
    observations = evaluate_beams(observations, policy)
    valid = _valid_beams(observations)
    diag: list[str] = []
    candidates: list[dict] = []

    if len(valid) <= 2:
        return {"candidates": [], "primary_candidate_id": None,
                "status": UNDETERMINED,
                "diagnostics": [f"only {len(valid)} valid beams; velocity undetermined",
                                *(f"beam {o.beam}: {','.join(o.reasons)}"
                                  for o in observations if o.reasons)]}

    radial = [o.radial_si for o in observations]
    if len(valid) == 3:
        sol = solve_radial(cfg, [radial[i] for i in valid], valid)
        cid = "c3_" + "".join(str(b + 1) for b in valid)
        candidates.append({
            "candidate_id": cid, "beams": list(valid), "kind": "3beam",
            "v_instrument": sol["velocity"], "error_velocity": None,
            "status": ADOPTED,
        })
        excluded_beams = [o.beam for o in observations if o.beam not in valid]
        diag.append(f"3-beam solution {cid} using beams "
                    f"{[b + 1 for b in valid]}; never mixed with a 4-beam value")
        for b in excluded_beams:
            reasons = next(o.reasons for o in observations if o.beam == b)
            diag.append(f"4-beam solution excluded: beam {b + 1} invalid ({','.join(reasons)})")
        return {"candidates": candidates, "primary_candidate_id": cid,
                "status": ADOPTED, "diagnostics": diag}

    # All 4 beams valid: the 4-beam least-squares value plus four references.
    sol4 = solve_radial(cfg, radial, tuple(range(cfg.n_beams)))
    err = sol4["error_velocity"] or 0.0
    suspicious = abs(err) > policy.err_velocity_max
    c4 = {"candidate_id": "c4", "beams": list(range(cfg.n_beams)), "kind": "4beam",
          "v_instrument": sol4["velocity"], "error_velocity": sol4["error_velocity"],
          "status": SUSPECT if suspicious else ADOPTED}
    candidates.append(c4)
    for triple in combinations(range(cfg.n_beams), 3):
        sol = solve_radial(cfg, [radial[i] for i in triple], triple)
        cid = "c3_" + "".join(str(b + 1) for b in triple)
        candidates.append({
            "candidate_id": cid, "beams": list(triple), "kind": "3beam",
            "v_instrument": sol["velocity"], "error_velocity": None,
            "status": SUSPECT if suspicious else REFERENCE,
        })
    if suspicious:
        diag.append(
            f"4-beam error velocity {err:.4f} m/s exceeds "
            f"{policy.err_velocity_max} m/s; c4 not adopted; all four 3-beam "
            "solutions retained as legal candidates (not merged)")
        return {"candidates": candidates, "primary_candidate_id": None,
                "status": SUSPECT, "diagnostics": diag}
    diag.append(f"4-beam c4 adopted; error velocity {err:.4f} m/s; "
                "3-beam solutions retained as references, never averaged")
    return {"candidates": candidates, "primary_candidate_id": "c4",
            "status": ADOPTED, "diagnostics": diag}


def cell_depth_ok(cell_depth_m: float, bottom_range_m: float | None,
                  blank_m: float = 0.0) -> tuple[bool, str]:
    if cell_depth_m <= 0:
        return False, f"invalid depth cell {cell_depth_m:g} m (non-positive)"
    if cell_depth_m < blank_m:
        return False, f"cell {cell_depth_m:g} m inside blanking {blank_m:g} m"
    if bottom_range_m is not None and cell_depth_m >= bottom_range_m:
        return False, (f"cell {cell_depth_m:g} m at/beyond bottom-track range "
                       f"{bottom_range_m:g} m")
    return True, "ok"
