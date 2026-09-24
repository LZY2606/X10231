"""Machine-readable event vocabulary.

Every judgment on the review path - beam exclusion, interpolation refusal,
multi-solution ambiguity, parameter revision, branch selection or rejection -
is persisted as an event with:

* a stable identity and monotonic sequence;
* the version it belongs to;
* a machine-readable ``code`` from :data:`EVENT_CODES`;
* ``affected`` stable object identities (so "contained to the right scope"
  is checkable mechanically);
* evidence inputs and the rule version that decided it.
"""

from __future__ import annotations

EVENT_CODES = {
    "case-imported": "raw case pack imported (all sources kept separate)",
    "baseline-derived": "first full successful derivation",
    "single-beam-failure": "one beam hard-failed; cell solved on 3 beams",
    "attitude-gap-refused": "attitude interpolation refused across a gap",
    "three-beam-ambiguity": "legal 3-beam and 4-beam candidates both kept",
    "invalid-depth-cell": "cell excluded (beyond bottom / blanking / bad raw)",
    "mount-revision": "install/mount angle version revised, propagation only",
    "transform-order-branch": "coordinate transform order branched",
    "branch-rejected": "branch reviewed and rejected, candidate retained",
    "local-source-edit": "one direct source changed, local recompute only",
    "decision-recorded": "human accepted a candidate; previous nodes retained",
    "decision-conflict": "two users edited the same base version; conflict",
    "conflict-resolved": "conflict resolved; both decisions stay auditable",
    "unparseable-raw": "raw token cannot be converted, kept with diagnostic",
    "derivation-incomplete": "a derivation failed; current pointer unchanged",
    "write-interrupted": "crash/rollback observed during version write",
    "exported": "archive exported",
    "imported-archive": "archive imported, identities/order/provenance kept",
    "view-restored": "historical event inspected; old conclusions restored",
}

# The three hard semantic invariants the replay tests assert against.
INVARIANTS = {
    "no-beam-solution-mixing":
        "three-beam and four-beam solutions are never merged into one value",
    "no-interpolation-across-gap":
        "attitude interpolation never spans a declared time gap",
    "fingerprint-covers-order-geometry-declination":
        "transform order, beam geometry and declination version enter the "
        "derived fingerprint",
}
