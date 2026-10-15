#!/usr/bin/env python3
"""Evaluate Doppler de-aliasing (zf_radar.doppler) against the ground truth.

The unwrapping itself is radar-only; the ground truth is used here only to
score it. For every point that matches the static-world model of the true ego
motion (wrapped residual below --tol), the correct wrap count is known; the
script reports how often the unwrapper recovers it, overall and in frames where
the ego speed exceeds the wrap speed. It also reports the error of the
radar-only sensor velocity estimate and the share of points per status.
Moving targets have no ground truth: as a cross-check, every point resolved
against the previous frame is resolved again, independently, against the next
frame, and the two results must agree.

Usage:
    python scripts/eval_doppler_unwrap.py -p /path/to/ZF_Dataset [--seq zf_03] [--limit N]
"""

import argparse
import glob
import os
import time

import numpy as np

from zf_radar.doppler import (
    STATUS_STATIC,
    STATUS_TRACKED,
    STATUS_UNRESOLVED,
    associate_moving,
    unit_vectors,
    unwrap_sequence,
    wrap,
)
from zf_radar.loaders import discover_sequences, load_extrinsics, load_ground_truth, load_pcd


def evaluate(seq_dir, limit=None, tol=0.3):
    files = sorted(glob.glob(os.path.join(seq_dir, "radar", "*.pcd")))[:limit]
    gt = load_ground_truth(seq_dir)
    T = load_extrinsics(seq_dir)
    lever = T[:3, 3]

    n_gt_static = n_ok = n_fast = n_fast_ok = n_wrapped = n_wrapped_ok = 0
    status_counts = np.zeros(3, int)
    vel_err = []
    n_cross = n_cross_ok = 0
    prev = None
    t0 = time.time()
    for i, (frame, res) in enumerate(unwrap_sequence(load_pcd(f) for f in files)):
        if prev is not None:
            p_frame, p_res = prev
            tracked = p_res["unwrap_status"] == STATUS_TRACKED
            dt = (p_frame["_meta"]["timestamp_ns"] - frame["_meta"]["timestamp_ns"]) * 1e-9
            v_fwd, ok = associate_moving(p_frame, frame, tracked, dt, p_res["sensor_velocity"][1] / lever[0])
            agree = np.abs(v_fwd - p_res["velocity_unwrapped"][tracked]) < 1.0
            n_cross += ok.sum()
            n_cross_ok += (ok & agree).sum()
        prev = (frame, res)
        A = frame["_meta"]["velocity_ambiguity"]
        m = np.asarray(frame["velocity"], float)
        v_true = T[:3, :3].T @ (gt["v_body"][i] + np.cross(gt["w_body"][i], lever))
        pred = -(unit_vectors(frame) @ v_true)
        gt_static = np.abs(wrap(m - pred, A)) < tol
        k_true = np.round((pred - m) / A)
        correct = (res["wrap_k"] == k_true) & gt_static

        n_gt_static += gt_static.sum()
        n_ok += correct.sum()
        wrapped = gt_static & (k_true != 0)
        n_wrapped += wrapped.sum()
        n_wrapped_ok += (correct & wrapped).sum()
        if np.linalg.norm(v_true) > A / 2:
            n_fast += gt_static.sum()
            n_fast_ok += correct.sum()
        status_counts += np.bincount(res["unwrap_status"], minlength=3)
        vel_err.append(np.linalg.norm(res["sensor_velocity"][:2] - v_true[:2]))
    vel_err = np.array(vel_err)
    pct = lambda a, b: 100.0 * a / b if b else float("nan")
    return {
        "frames": len(vel_err),
        "static_correct_pct": pct(n_ok, n_gt_static),
        "wrapped_static_correct_pct": pct(n_wrapped_ok, n_wrapped),
        "wrapped_static_points": n_wrapped,
        "fast_frames_static_correct_pct": pct(n_fast_ok, n_fast),
        "ego_vel_err_median": np.median(vel_err),
        "ego_vel_err_p99": np.percentile(vel_err, 99),
        "ego_vel_err_gt_1ms_pct": 100.0 * np.mean(vel_err > 1.0),
        "status_pct": 100.0 * status_counts / status_counts.sum(),
        "tracked_crosscheck_agree_pct": pct(n_cross_ok, n_cross),
        "tracked_crosscheck_n": n_cross,
        "ms_per_frame": 1000 * (time.time() - t0) / max(len(vel_err), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Doppler de-aliasing against ground truth")
    parser.add_argument("-p", "--path", required=True, help="Dataset root containing sequence directories.")
    parser.add_argument("--seq", nargs="+", help="Only these sequences (default: all)")
    parser.add_argument("--limit", type=int, help="Only the first N frames per sequence")
    parser.add_argument("--tol", type=float, default=0.3, help="Static-model tolerance in m/s (default 0.3)")
    args = parser.parse_args()

    print(f"{'seq':6} {'frames':>6} {'static ok':>9} {'wrapped ok':>10} {'(n wrapped)':>11} {'fast ok':>8} "
          f"{'v err med':>9} {'p99':>6} {'>1 m/s':>7} {'static':>7} {'tracked':>7} {'unres.':>7} {'x-check':>8} {'ms/fr':>6}")
    for name in args.seq or discover_sequences(args.path):
        r = evaluate(os.path.join(args.path, name), args.limit, args.tol)
        s = r["status_pct"]
        print(f"{name:6} {r['frames']:6d} {r['static_correct_pct']:8.2f}% {r['wrapped_static_correct_pct']:9.2f}% "
              f"{r['wrapped_static_points']:11d} {r['fast_frames_static_correct_pct']:7.2f}% "
              f"{r['ego_vel_err_median']:9.3f} {r['ego_vel_err_p99']:6.2f} {r['ego_vel_err_gt_1ms_pct']:6.2f}% "
              f"{s[STATUS_STATIC]:6.1f}% {s[STATUS_TRACKED]:6.1f}% {s[STATUS_UNRESOLVED]:6.1f}% "
              f"{r['tracked_crosscheck_agree_pct']:7.2f}% {r['ms_per_frame']:6.1f}",
              flush=True)


if __name__ == "__main__":
    main()
