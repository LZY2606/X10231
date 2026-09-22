import math

import numpy as np

from app.attitude import AttitudeSample, attitude_at, circular_interp, wrap_deg
from app import transforms as tf


def test_circular_interp_crosses_zero():
    assert abs(circular_interp(358.0, 2.0, 0.5) - 0.0) < 1e-9
    assert abs(circular_interp(359.0, 1.0, 0.25) - 359.5) < 1e-9


def test_wrap():
    assert wrap_deg(360.0) == 0.0
    assert abs(wrap_deg(-1.0) - 359.0) < 1e-9


def test_no_interpolation_across_gap():
    samples = [AttitudeSample(0.0, 10.0, 0.0, 0.0),
               AttitudeSample(10.0, 20.0, 0.0, 0.0)]
    res = attitude_at(samples, 5.0, max_gap_s=2.5)
    assert res.status == "gap"
    assert res.heading_deg is None
    assert res.bracket == [0.0, 10.0]


def test_interpolation_within_gap():
    samples = [AttitudeSample(0.0, 358.0, 1.0, 0.0),
               AttitudeSample(1.0, 2.0, 1.0, 0.0)]
    res = attitude_at(samples, 0.5, max_gap_s=2.5)
    assert res.status == "interpolated"
    assert abs(res.heading_deg - 0.0) < 1e-9


def test_outside_window():
    samples = [AttitudeSample(0.0, 1.0, 0.0, 0.0),
               AttitudeSample(1.0, 2.0, 0.0, 0.0)]
    assert attitude_at(samples, 3.0, 5.0).status == "outside"


def test_transform_identity_and_order_differs():
    m = tf.earth_matrix(0.0, 0.0, 0.0, 0.0)
    assert np.allclose(m, np.eye(3), atol=1e-9)
    std = tf.earth_matrix(120.0, 5.0, -3.0, 7.5, "RDI_HRP")
    alt = tf.earth_matrix(120.0, 5.0, -3.0, 7.5, "ALT_PRH")
    assert not np.allclose(std, alt)
    assert np.allclose(std @ std.T, np.eye(3), atol=1e-9)


def test_declination_adds_east_positive():
    heading_only = tf.earth_matrix(90.0, 0.0, 0.0, 0.0)
    decl_only = tf.earth_matrix(0.0, 0.0, 0.0, 90.0)
    assert np.allclose(heading_only, decl_only, atol=1e-9)
    shifted = tf.earth_matrix(355.0, 0.0, 0.0, 10.0)
    assert np.allclose(shifted, tf.earth_matrix(5.0, 0.0, 0.0, 0.0), atol=1e-9)


def test_tilt_matrix_orthogonal():
    t = tf.tilt_matrix(4.0, -2.0)
    assert np.allclose(t @ t.T, np.eye(3), atol=1e-9)
