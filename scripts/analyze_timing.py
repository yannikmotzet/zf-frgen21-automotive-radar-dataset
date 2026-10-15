#!/usr/bin/env python3
"""Analyze radar frame timing: alternating scan modes, frame cycle and chirp-sequence offsets.

Each frame's ``timestamp_ns`` marks the midpoint of its chirp sequence. Frames
start on a fixed cycle T, but the two alternating scan modes have chirp
sequences of different length, so the midpoint sits at a mode-dependent offset
after the cycle start. The script fits

    t_i = t_0 + slot_i * T + offset * is_high_mode_i

per sequence, where ``slot_i`` counts frame cycles (so skipped cycles show up
as jumps), and reports the cycle, its deviation from the nominal period in
ppm, the chirp-sequence length difference ``delta = -2 * offset`` and the
frames that deviate from the grid. The residual of that fit is a sawtooth:
between resyncs the timestamps drift off the grid at a constant rate
(``drift`` in ppm) and then step back; ``resyncs``/``every`` count those steps.

Usage:
    python scripts/analyze_timing.py -p /path/to/ZF_Dataset [--seq zf_05] [--tol 1.5]
"""

import argparse
import glob
import os

import numpy as np

from zf_radar.loaders import discover_sequences, load_pcd_meta


def frame_timing(seq_dir):
    files = sorted(glob.glob(os.path.join(seq_dir, "radar", "*.pcd")))
    metas = [load_pcd_meta(f) for f in files]
    ts = np.array([m["timestamp_ns"] for m in metas], dtype=np.int64)
    v_amb = np.array([m["velocity_ambiguity"] for m in metas])
    return ts, v_amb


def find_resyncs(residual, window=6, min_jump_ms=0.6):
    """Indices where the grid residual steps up (clock resync), one per step."""
    kernel = np.ones(window) / window
    mean = np.convolve(residual, kernel, mode="valid")  # mean[k] = residual[k:k+window]
    # step at i: mean of the window starting at i minus mean of the window ending before i
    step = np.full(len(residual), -np.inf)
    step[window:len(mean)] = mean[window:] - mean[:len(mean) - window]
    idx = []
    for i in np.flatnonzero(step > min_jump_ms):
        if idx and i - idx[-1] <= window:
            if step[i] > step[idx[-1]]:
                idx[-1] = i
        else:
            idx.append(i)
    return np.array(idx, dtype=int)


def analyze(ts, v_amb, tol_ms=1.5):
    t = (ts - ts[0]) * 1e-6  # ms since first frame
    dt = np.diff(t)
    modes = np.unique(v_amb)
    if len(modes) != 2:
        raise ValueError(f"expected two scan modes, found velocity_ambiguity values {modes.tolist()}")
    high = v_amb == modes[1]

    # Nominal cycle from consecutive pairs (one frame of each mode)
    T_nominal = np.median(t[2:] - t[:-2]) / 2
    slot = np.concatenate([[0], np.cumsum(np.round(dt / T_nominal).astype(int))])

    A = np.column_stack([np.ones_like(t), slot, high.astype(float)])
    (t0, T, offset), *_ = np.linalg.lstsq(A, t, rcond=None)
    residual = t - A @ [t0, T, offset]

    # Between resyncs the residual drifts linearly (sawtooth): fit a line per
    # segment, skipping a few frames around each step
    resyncs = find_resyncs(residual)
    slopes = []
    for a, b in zip(np.concatenate([[0], resyncs]), np.concatenate([resyncs, [len(t)]])):
        a, b = a + 3, b - 3
        if b - a > 20:
            slopes.append(np.polyfit(slot[a:b], residual[a:b], 1)[0])
    drift_ppm = np.median(slopes) / T * 1e6 if slopes else np.nan
    resync_interval_s = np.median(np.diff(slot[resyncs])) * T * 1e-3 if len(resyncs) > 1 else np.nan

    low_to_high = dt[~high[:-1] & high[1:]]
    high_to_low = dt[high[:-1] & ~high[1:]]
    return {
        "modes": modes,
        "frames": len(t),
        "strict_alternation": bool(np.all(high[1:] != high[:-1])),
        "dt_low_to_high": np.median(low_to_high) if len(low_to_high) else np.nan,
        "dt_high_to_low": np.median(high_to_low) if len(high_to_low) else np.nan,
        "cycle_ms": T,
        "cycle_ppm": (T / round(T, 1) - 1) * 1e6,
        "delta_ms": -2 * offset,
        "skipped_cycles": int(slot[-1] + 1 - len(t)),
        "drift_ppm": drift_ppm,
        "resyncs": len(resyncs),
        "resync_interval_s": resync_interval_s,
        "residual_std": residual.std(),
        "residual_p99": np.percentile(np.abs(residual), 99),
        "outliers": np.flatnonzero(np.abs(residual) > tol_ms),
        "residual": residual,
        "dt": dt,
    }


def main():
    parser = argparse.ArgumentParser(description="Radar frame timing analysis")
    parser.add_argument("-p", "--path", required=True, help="Dataset root containing sequence directories.")
    parser.add_argument("--seq", nargs="+", help="Only these sequences (default: all)")
    parser.add_argument("--tol", type=float, default=1.5, help="Grid deviation in ms reported as outlier")
    args = parser.parse_args()

    header = (f"{'seq':6} {'v_amb modes [m/s]':>18} {'alt':>4} {'dt lo->hi':>9} {'dt hi->lo':>9} "
              f"{'cycle [ms]':>11} {'[ppm]':>6} {'delta [ms]':>10} {'skipped':>7} "
              f"{'drift':>6} {'resyncs':>7} {'every':>6} {'res std':>7} {'res p99':>7} {'outliers':>8}")
    print(header)
    details = []
    for name in args.seq or discover_sequences(args.path):
        r = analyze(*frame_timing(os.path.join(args.path, name)), tol_ms=args.tol)
        print(f"{name:6} {r['modes'][0]:8.3f} /{r['modes'][1]:8.3f} {'yes' if r['strict_alternation'] else 'NO':>4} "
              f"{r['dt_low_to_high']:9.2f} {r['dt_high_to_low']:9.2f} {r['cycle_ms']:11.5f} {r['cycle_ppm']:+6.2f} "
              f"{r['delta_ms']:10.2f} {r['skipped_cycles']:7d} {r['drift_ppm']:+6.0f} {r['resyncs']:7d} "
              f"{r['resync_interval_s']:5.1f}s {r['residual_std']:7.2f} {r['residual_p99']:7.2f} "
              f"{len(r['outliers']):8d}")
        details.append((name, r))

    print(f"\nFrames deviating more than {args.tol} ms from the fitted grid "
          "(dt = interval to the previous frame):")
    for name, r in details:
        if not len(r["outliers"]):
            continue
        shown = r["outliers"][:10]
        items = ", ".join(
            f"{i} ({r['residual'][i]:+.1f} ms, dt {r['dt'][i - 1]:.1f})" if i else f"0 ({r['residual'][0]:+.1f} ms)"
            for i in shown
        )
        more = f", ... (+{len(r['outliers']) - len(shown)})" if len(r["outliers"]) > len(shown) else ""
        print(f"  {name}: {items}{more}")


if __name__ == "__main__":
    main()
