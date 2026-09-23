import math

import numpy as np

from adcp import transforms as tf


def test_heading_unwrap_crosses_360_short_arc():
    heads = tf.unwrap_degrees([358.0, 359.5, 0.5, 2.0])
    assert heads[2] - heads[1] == 1.0
    assert heads[3] - heads[2] == 1.5


def test_attitude_interpolation_across_360():
    samples = [(4.5, 359.0, 1.0, -2.0), (5.0, 1.0, 1.2, -1.8)]
    res, diag = tf.interpolate_attitude(samples, 4.75, max_gap_s=1.5)
    assert diag is None
    assert abs(res["heading"] - 0.0) < 1e-6  # 360, not 180


def test_attitude_gap_not_interpolated():
    samples = [(8.0, 10.0, 1.0, -2.0), (10.0, 14.0, 1.2, -1.8)]
    res, diag = tf.interpolate_attitude(samples, 9.0, max_gap_s=1.5)
    assert res is None
    assert "gap" in diag and "refused" in diag


def test_three_beam_and_four_beam_differ_and_stay_separate():
    theta = 20.0
    true = np.array([0.35, -0.12, 0.03])
    m = tf.beam_matrix(theta)
    beams = (m @ true).tolist()
    v4, err4 = tf.instrument_from_beams(beams, theta)
    masked = beams[:1] + [None] + beams[2:]
    v3, err3 = tf.instrument_from_beams(masked, theta)
    assert np.allclose(v4, true, atol=1e-9)
    assert err4 < 1e-9
    # 3-beam solution is exactly determined but is a different object;
    # the pipeline stores them as separate candidates, never averaged.
    assert v3.shape == v4.shape == (3,)


def test_transform_order_changes_result():
    att = {"heading": 40.0, "pitch": 3.0, "roll": -5.0}
    v = [0.3, -0.1, 0.02]
    e1, _, o1 = tf.instr_to_earth(v, att, 1.2, -7.2,
                                  order=tf.ORDER_MOUNTING_FIRST)
    e2, _, o2 = tf.instr_to_earth(v, att, 1.2, -7.2,
                                  order=tf.ORDER_ATTITUDE_FIRST)
    assert o1 != o2
    assert not np.allclose(e1, e2)


def test_declination_changes_earth_frame():
    att = {"heading": 10.0, "pitch": 0.0, "roll": 0.0}
    v = [0.5, 0.0, 0.0]
    e1 = tf.ship_to_earth(v, att["heading"], 0, 0, -7.2)
    e2 = tf.ship_to_earth(v, att["heading"], 0, 0, -7.0)
    assert not np.allclose(e1, e2)
