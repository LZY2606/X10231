"""Three-beam solutions and ambiguity candidates.

Rule: a three-beam solution is never merged with a four-beam solution into a
single value. When a beam fails, every legal candidate is kept:
  * "zero-error"      - three-beam solve assuming error velocity 0
  * "err-from-cell"   - three-beam solve using the error velocity of the
                        nearest valid four-beam cell in the same ping
  * "err-from-ping"   - three-beam solve using the error velocity of the same
                        cell in the nearest valid four-beam ping
Each candidate carries its own stable id and provenance. If no donor error
velocity exists, that candidate is omitted and the omission is recorded.
"""

from __future__ import annotations

from .transforms import beam_to_instrument_three


def three_beam_candidates(beam_vel_valid, valid_idx, theta_deg,
                          err_from_cell=None, err_from_ping=None):
    """Return (candidates, omissions).

    candidates: list of dicts {kind, vector, err_assumed, provenance}
    omissions:  list of dicts {kind, reason}
    """
    candidates = [{
        "kind": "zero-error",
        "vector": list(beam_to_instrument_three(
            beam_vel_valid, valid_idx, theta_deg, 0.0)),
        "err_assumed": 0.0,
        "provenance": "three-beam solve, error velocity assumed 0",
    }]
    omissions = []
    for kind, err, prov in (
        ("err-from-cell", err_from_cell,
         "three-beam solve, error velocity from nearest valid cell, same ping"),
        ("err-from-ping", err_from_ping,
         "three-beam solve, error velocity from same cell, nearest valid ping"),
    ):
        if err is None:
            omissions.append({"kind": kind, "reason": "no donor error velocity"})
        else:
            candidates.append({
                "kind": kind,
                "vector": list(beam_to_instrument_three(
                    beam_vel_valid, valid_idx, theta_deg, err)),
                "err_assumed": err,
                "provenance": prov,
            })
    return candidates, omissions
