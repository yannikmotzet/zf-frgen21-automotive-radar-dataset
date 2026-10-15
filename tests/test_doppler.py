"""Doppler de-aliasing on a synthetic scene with alternating scan modes."""

import numpy as np
import pytest

from zf_radar.doppler import (
    STATUS_STATIC,
    STATUS_TRACKED,
    estimate_sensor_velocity,
    unit_vectors,
    unwrap_sequence,
    wrap,
)

A_LOW, A_HIGH = 32.499187, 43.572987
DT = 0.09
T0 = 1_626_684_577_435_105_592


def make_frame(i, v_sensor, rng, moving=None):
    """Static scene seen from a sensor moving with v_sensor, plus optional moving points
    given as (range, azimuth, radial velocity) at frame i."""
    n = 300
    az = rng.uniform(-np.radians(50), np.radians(50), n)
    el = rng.uniform(-np.radians(8), np.radians(8), n)
    rng_m = rng.uniform(5, 80, n)
    radial = None
    if moving is not None:
        r_mv, az_mv, v_mv = moving
        az = np.concatenate([az, az_mv])
        el = np.concatenate([el, np.zeros(len(az_mv))])
        rng_m = np.concatenate([rng_m, r_mv])
        radial = v_mv
    x, y, z = rng_m * np.cos(el) * np.cos(az), rng_m * np.cos(el) * np.sin(az), rng_m * np.sin(el)
    u = np.column_stack([x, y, z]) / rng_m[:, None]
    true_v = -(u @ v_sensor)
    if radial is not None:
        true_v[-len(radial):] = radial
    A = A_LOW if i % 2 == 0 else A_HIGH
    frame = {"x": x, "y": y, "z": z, "range": rng_m, "azimuth": az, "elevation": el,
             "velocity": wrap(true_v + rng.normal(0, 0.02, len(true_v)), A),
             "_meta": {"timestamp_ns": T0 + int(i * DT * 1e9), "velocity_ambiguity": A}}
    return frame, true_v


def test_wrap_range():
    v = np.array([-40.0, -16.3, 0.0, 16.2, 30.0])
    w = wrap(v, A_LOW)
    assert np.all((w >= -A_LOW / 2) & (w < A_LOW / 2))
    assert w[2] == 0.0 and w[3] == pytest.approx(16.2)
    assert w[4] == pytest.approx(30.0 - A_LOW)


def test_sensor_velocity_above_wrap_speed():
    rng = np.random.default_rng(0)
    v_true = np.array([25.0, 0.3, 0.0])  # 90 km/h, well above A/2
    frame, _ = make_frame(0, v_true, rng)
    v, inliers, fraction = estimate_sensor_velocity(unit_vectors(frame), frame["velocity"], A_LOW)
    np.testing.assert_allclose(v[:2], v_true[:2], atol=0.05)
    assert fraction > 0.95


def test_static_and_moving_points_unwrapped():
    rng = np.random.default_rng(1)
    v_sensor = np.array([22.0, 0.0, 0.0])
    # oncoming car straight ahead: closing at 45 m/s, wrapped in both modes
    frames, truths = [], []
    for i in range(4):
        r_car = 60.0 - 45.0 * DT * i
        moving = (np.full(3, r_car) + [0.0, 0.3, 0.6], np.radians([2.0, 2.2, 2.4]), np.full(3, -45.0))
        f, t = make_frame(i, v_sensor, rng, moving)
        frames.append(f)
        truths.append(t)

    results = list(unwrap_sequence(frames))
    assert len(results) == 4
    for (frame, res), truth in zip(results, truths):
        static = res["unwrap_status"] == STATUS_STATIC
        assert static[:-3].mean() > 0.95
        np.testing.assert_allclose(res["velocity_unwrapped"][:-3][static[:-3]], truth[:-3][static[:-3]], atol=0.1)
        car = slice(-3, None)
        assert np.all(res["unwrap_status"][car] == STATUS_TRACKED)
        np.testing.assert_allclose(res["velocity_unwrapped"][car], -45.0, atol=0.1)
        assert np.all(res["wrap_k"][car] != 0)
        assert res["wrap_k"].dtype == np.int8
