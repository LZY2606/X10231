import numpy as np

from app import geometry as g
from app.geometry import TransducerConfig, solve_four, solve_three_candidates, classify_beams

CFG = TransducerConfig()


def test_roundtrip_recovers_xyz():
    xyz = np.array([120.0, -340.0, -15.0])
    r = g.beam_matrix(CFG) @ xyz
    sol = solve_four(list(r), CFG)
    assert sol.candidate_id == "4b"
    assert np.allclose(sol.xyz_mm_s, xyz, atol=1e-8)


def test_homogeneous_flow_three_beams_agree_with_four():
    xyz = np.array([50.0, 600.0, -10.0])
    r = g.beam_matrix(CFG) @ xyz
    four = solve_four(list(r), CFG)
    sparse = [None] + [float(x) for x in r[1:]]
    cands = solve_three_candidates(sparse, CFG)
    ids = {c.candidate_id for c in cands}
    assert ids == {"3bA:pair(1,4)", "3bB"}
    for c in cands:
        assert np.allclose(c.xyz_mm_s, four.xyz_mm_s, atol=1e-6)


def test_inhomogeneous_flow_three_beams_diverge():
    r = g.beam_matrix(CFG) @ np.array([0.0, 600.0, 0.0])
    r[0] += 80.0  # perturb one beam: error velocity nonzero
    sparse = [None] + [float(x) for x in r[1:]]
    cands = solve_three_candidates(sparse, CFG)
    a = next(c for c in cands if c.candidate_id.startswith("3bA"))
    b = next(c for c in cands if c.candidate_id == "3bB")
    assert not np.allclose(a.xyz_mm_s, b.xyz_mm_s)


def test_two_failed_beams_is_indeterminate():
    assert solve_three_candidates([None, None, 1.0, 2.0], CFG) == []


def test_classify_beams_thresholds():
    rad = [10.0, 10.0, 10.0, 10.0]
    corr = [200, 10, 200, 200]
    amp = [120, 120, 5, 120]
    flags = [False, False, False, True]
    q = classify_beams(rad, corr, amp, flags, CFG)
    assert q.good_beams == (0,)
    assert "bad_beam_flag" in q.reasons[3]
    assert "correlation_below" in q.reasons[1]
    assert "amplitude_below" in q.reasons[2]


def test_candidate_ids_stable():
    for bad in range(4):
        xyz = np.array([10.0, -10.0, 5.0])
        r = [None if i == bad else float(x) for i, x in enumerate(g.beam_matrix(CFG) @ xyz)]
        cands = solve_three_candidates(r, CFG)
        assert cands[0].used_beams == tuple(i for i in range(4) if i != bad)
        assert all(c.candidate_id for c in cands)
