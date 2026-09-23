"""Per-beam quality tier.

``quality-tier-v1``: a beam is invalid when ANY of the following holds:

* the hard ``bad_beam_flag`` is set (source-recorded, never recomputed away);
* normalised correlation is below ``min_correlation``;
* normalised amplitude is below ``min_amplitude_db``.

Thresholds are part of the configuration version; the raw measurements and
every failed check are retained as evidence.
"""
from __future__ import annotations

QUALITY_RULE = "quality-tier-v1"


def evaluate_beams(
    correlations,
    amplitudes_db,
    bad_flags,
    min_correlation,
    min_amplitude_db,
):
    """Return ``{beams:[...], valid:[bool x4]}``.

    Each entry records correlation/amplitude (already normalised) plus the
    ``reasons`` the beam was rejected (empty => valid).
    """
    beams = []
    valid = []
    for i in range(4):
        reasons = []
        if bad_flags[i]:
            reasons.append("bad_beam_flag")
        corr = correlations[i]
        amp = amplitudes_db[i]
        if corr is None:
            reasons.append("correlation_unconvertible")
        elif corr < min_correlation:
            reasons.append("correlation_below_threshold")
        if amp is None:
            reasons.append("amplitude_unconvertible")
        elif amp < min_amplitude_db:
            reasons.append("amplitude_below_threshold")
        beams.append(
            {
                "beam": i + 1,
                "correlation": corr,
                "amplitude_db": amp,
                "bad_beam_flag": bool(bad_flags[i]),
                "reasons": reasons,
            }
        )
        valid.append(not reasons)
    return {"rule": QUALITY_RULE, "beams": beams, "valid": valid}
