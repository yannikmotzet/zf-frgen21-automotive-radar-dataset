"""Tests for the route statistics behind the plots."""

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from zf_radar.plots import summarize


def test_summarize():
    n = 11
    pos = np.zeros((n, 3))
    pos[:, 0] = np.linspace(0, 100, n)  # 100 m straight line
    s = {
        "t": np.linspace(0, 60, n),
        "pos": pos,
        "speed": np.array([0, 5, 10, 15, 17, 17, 17, 22, 22, 10, 0], dtype=float),
        "yaw_rate": np.zeros(n),
        "num_points": np.full(n, 500),
        # alternating scan modes: wrap at 16.25 and 21.8 m/s
        "v_amb": np.tile([32.5, 43.6], n)[:n],
    }
    out = summarize("zf_99", s)
    assert out["frames"] == 11
    assert out["distance_km"] == 0.1
    assert out["duration_min"] == 1.0
    assert out["max_speed_kmh"] == pytest.approx(22 * 3.6, abs=0.1)
    # frames above their own wrap speed: idx 4 (17 > 16.25), 6 (17 > 16.25), 7 (22 > 21.8), 8 (22 > 16.25)
    assert out["aliased_frames_pct"] == pytest.approx(100 * 4 / 11, abs=0.1)
