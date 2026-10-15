"""The timing analysis must recover cycle, mode offset, drift and gaps from synthetic timestamps."""

import numpy as np
import pytest

from scripts.analyze_timing import analyze


def synthetic_timestamps(n=2000, cycle_ms=90.0, delta_ms=7.9, drift_ppm=-65, resync_every=174, skip_at=1000):
    slot = np.arange(n) + (np.arange(n) >= skip_at)  # one skipped cycle
    high = np.arange(n) % 2 == 1
    sawtooth = (slot % resync_every) * cycle_ms * drift_ppm * 1e-6
    sawtooth -= sawtooth.mean()
    t_ms = slot * cycle_ms - high * delta_ms / 2 + sawtooth
    ts = 1_626_684_577_435_105_592 + np.round(t_ms * 1e6).astype(np.int64)
    v_amb = np.where(high, 43.572987, 32.499187)
    return ts, v_amb


def test_recovers_timing_model():
    r = analyze(*synthetic_timestamps())
    assert r["strict_alternation"]
    assert r["cycle_ms"] == pytest.approx(90.0, abs=1e-3)
    assert r["delta_ms"] == pytest.approx(7.9, abs=0.05)
    assert r["skipped_cycles"] == 1
    assert r["drift_ppm"] == pytest.approx(-65, abs=3)
    assert r["resync_interval_s"] == pytest.approx(174 * 0.09, rel=0.02)
    assert r["dt_low_to_high"] == pytest.approx(86.05, abs=0.05)


def test_rejects_single_mode():
    ts, _ = synthetic_timestamps(n=100)
    with pytest.raises(ValueError, match="two scan modes"):
        analyze(ts, np.full(100, 32.5))
