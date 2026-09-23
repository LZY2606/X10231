"""Per-cell beam quality evaluation.

Correlation, amplitude and bad-beam flags are recorded as separate sources;
this module only combines them into a per-(ping, cell) quality verdict.
Excluded cells are kept with diagnostics, never dropped.
"""

from __future__ import annotations


def evaluate_cell(corr_counts, amp_counts, bad_beam_flags, depth_m,
                  bottom_depth_m, corr_min, amp_min, amp_max):
    """Return dict(valid_beams, excluded, reasons).

    corr_counts / amp_counts: per-beam sequences (4 entries).
    bad_beam_flags: per-beam booleans for the ping.
    """
    reasons = []
    valid = []
    for b in range(4):
        beam_ok = True
        if bad_beam_flags[b]:
            reasons.append(f"beam{b + 1}: bad_beam_flag set")
            beam_ok = False
        if corr_counts[b] < corr_min:
            reasons.append(
                f"beam{b + 1}: correlation {corr_counts[b]} < {corr_min} count")
            beam_ok = False
        if not (amp_min <= amp_counts[b] <= amp_max):
            reasons.append(
                f"beam{b + 1}: amplitude {amp_counts[b]} outside "
                f"[{amp_min}, {amp_max}] count")
            beam_ok = False
        if beam_ok:
            valid.append(b)
    if bottom_depth_m is not None and depth_m > bottom_depth_m:
        reasons.append(
            f"cell center {depth_m:.3f} m below bottom {bottom_depth_m:.3f} m")
        valid = []
    excluded = len(valid) < 3
    if excluded and not any("fewer than 3" in r for r in reasons):
        reasons.append(f"fewer than 3 valid beams ({len(valid)})")
    return {"valid_beams": valid, "excluded": excluded, "reasons": reasons}
