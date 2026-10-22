#!/usr/bin/env python3
"""GT-free consistency check of the Doppler de-aliasing for moving targets.

Moving targets have no ground truth. Two symptoms of failure are measurable anyway:
- unresolved share: non-static points that keep their wrapped value (they flicker
  between the two scan modes from frame to frame);
- intra-object inconsistency: non-static points are clustered per frame (x/y, 2 m);
  within a cluster of >= 3 points a point whose unwrapped velocity differs by more than
  2 m/s from the cluster median is counted as inconsistent (an object cannot be
  approaching at -30 m/s and receding at +3 m/s at the same time).
- flicker: every non-static point is matched to the nearest non-static point of the
  previous frame (ego motion removed, within 1 m); a velocity jump of more than
  3 m/s counts as flicker. Reported for the raw and the unwrapped velocity; jumps
  above 25 m/s are wrap-count errors on one of the two sides.
Also reports implausible values (|v - static prediction| > 50 m/s, i.e. > 180 km/h
over ground).

Usage: python scripts/eval_moving_consistency.py -p /path/to/ZF_Dataset [--seq zf_03] [--two-sided]
"""
import argparse
import glob
import os

import numpy as np

from zf_radar.doppler import STATUS_STATIC, STATUS_UNRESOLVED, unit_vectors, unwrap_sequence
from zf_radar.loaders import discover_sequences, load_pcd


def in_frame(partner, dt, sensor_velocity, yaw_rate):
    """Points of the previous frame in the current sensor frame (ego motion removed)."""
    p = np.column_stack([partner["x"], partner["y"], partner["z"]]).astype(float) - sensor_velocity * dt
    c, s = np.cos(yaw_rate * dt), np.sin(yaw_rate * dt)
    return np.column_stack([c * p[:, 0] + s * p[:, 1], -s * p[:, 0] + c * p[:, 1], p[:, 2]])


def clusters(xy, eps=2.0):
    """Connected components of points closer than eps (single linkage)."""
    n = len(xy)
    labels = -np.ones(n, int)
    if n == 0:
        return labels
    d2 = np.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=-1)
    nbrs = [np.flatnonzero(row) for row in d2 < eps * eps]
    c = 0
    for i in range(n):
        if labels[i] >= 0:
            continue
        stack, labels[i] = [i], c
        while stack:
            for k in nbrs[stack.pop()]:
                if labels[k] < 0:
                    labels[k] = c
                    stack.append(k)
        c += 1
    return labels


def evaluate(seq_dir, two_sided, limit=None):
    files = sorted(glob.glob(os.path.join(seq_dir, "radar", "*.pcd")))[:limit]
    n_mov = n_unres = n_inclust = n_incons = n_implaus = n_pairs = n_flick_raw = n_flick = n_jump = 0
    prev = None
    for frame, res in unwrap_sequence((load_pcd(f) for f in files), two_sided=two_sided):
        st = res["unwrap_status"]
        mov = st != STATUS_STATIC
        xyz = np.column_stack([frame["x"], frame["y"], frame["z"]]).astype(float)[mov]
        if prev is not None and mov.any() and prev[1].any():
            p_frame, p_mov, p_res = prev
            dt = (frame["_meta"]["timestamp_ns"] - p_frame["_meta"]["timestamp_ns"]) * 1e-9
            v_s = res["sensor_velocity"]
            q = in_frame(p_frame, dt, v_s, v_s[1] / 3.925)[p_mov]
            d2 = np.sum((xyz[:, None] - q[None]) ** 2, axis=-1)
            nn = np.argmin(d2, axis=1)
            hit = d2[np.arange(len(nn)), nn] < 1.0
            raw = np.asarray(frame["velocity"], float)[mov][hit]
            p_raw = np.asarray(p_frame["velocity"], float)[p_mov][nn[hit]]
            unw = res["velocity_unwrapped"][mov][hit]
            p_unw = p_res["velocity_unwrapped"][p_mov][nn[hit]]
            n_pairs += hit.sum()
            n_flick_raw += np.sum(np.abs(raw - p_raw) > 3.0)
            n_flick += np.sum(np.abs(unw - p_unw) > 3.0)
            n_jump += np.sum(np.abs(unw - p_unw) > 25.0)
        prev = (frame, mov, res)
        if not mov.any():
            continue
        v = res["velocity_unwrapped"][mov]
        pred = -(unit_vectors(frame) @ res["sensor_velocity"])[mov]
        n_mov += mov.sum()
        n_unres += np.sum(st[mov] == STATUS_UNRESOLVED)
        n_implaus += np.sum(np.abs(v - pred) > 50.0)
        xy = np.column_stack([frame["x"], frame["y"]])[mov]
        lab = clusters(xy)
        for c in np.unique(lab):
            idx = lab == c
            if idx.sum() < 3:
                continue
            n_inclust += idx.sum()
            n_incons += np.sum(np.abs(v[idx] - np.median(v[idx])) > 2.0)
    return {"moving": n_mov, "unresolved_pct": 100 * n_unres / max(n_mov, 1),
            "inconsistent_pct": 100 * n_incons / max(n_inclust, 1), "in_clusters": n_inclust,
            "implausible_pct": 100 * n_implaus / max(n_mov, 1),
            "flicker_raw_pct": 100 * n_flick_raw / max(n_pairs, 1), "flicker_pct": 100 * n_flick / max(n_pairs, 1),
            "jump_pct": 100 * n_jump / max(n_pairs, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True)
    ap.add_argument("--seq", nargs="+")
    ap.add_argument("--two-sided", action="store_true")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    print(f"{'seq':6} {'non-static pts':>14} {'unresolved':>10} {'in clusters':>11} {'inconsistent':>12} {'implausible':>11} {'flicker raw':>11} {'unwrapped':>9} {'> 25 m/s':>8}")
    for name in args.seq or discover_sequences(args.path):
        r = evaluate(os.path.join(args.path, name), args.two_sided, args.limit)
        print(f"{name:6} {r['moving']:14d} {r['unresolved_pct']:9.1f}% {r['in_clusters']:11d} "
              f"{r['inconsistent_pct']:11.1f}% {r['implausible_pct']:10.2f}% "
              f"{r['flicker_raw_pct']:10.1f}% {r['flicker_pct']:8.1f}% {r['jump_pct']:7.2f}%", flush=True)


if __name__ == "__main__":
    main()
