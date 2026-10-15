#!/usr/bin/env python3
"""Static route plots and per-sequence statistics for the ZF FRGen21 dataset.

Writes, for every sequence found under the dataset root:
    routes_overview.png     top-down route of every sequence, colored by speed
    <seq>_profile.png       route + speed / height / yaw rate / detections over time
    summary.csv             one row of statistics per sequence

Routes are drawn in each sequence's local world frame (origin = first frame,
x forward, y left). The dataset ships no geodetic coordinates, so sequences
cannot be placed on a map or relative to one another.
"""

import argparse
import csv
import glob
import os

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize
except ImportError as e:
    raise SystemExit(
        f"Missing plotting dependency: {e.name}. The route plots need the [plot] extras:\n"
        '  pip install "zf-frgen21-radar-dataset[plot]"'
    ) from e

from zf_radar.loaders import discover_sequences, load_ground_truth, load_pcd_meta

MS_TO_KMH = 3.6

# Single-hue sequential ramp (light -> dark blue) for speed; starts at a step
# that still reads as a line on the white surface
SPEED_CMAP = LinearSegmentedColormap.from_list(
    "speed", ["#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#8a8984", "grid": "#e4e3df"}
SERIES = "#2a78d6"
REFERENCE = "#eb6834"


def _style():
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": INK["muted"],
        "axes.labelcolor": INK["secondary"],
        "axes.titlecolor": INK["primary"],
        "axes.grid": True,
        "grid.color": INK["grid"],
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK["secondary"],
        "ytick.color": INK["secondary"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "semibold",
    })


def load_sequence_stats(seq_dir):
    """Ground truth plus per-frame header metadata for one sequence."""
    gt = load_ground_truth(seq_dir)
    if gt is None:
        raise FileNotFoundError(f"{seq_dir}: ground_truth.csv missing")
    files = sorted(glob.glob(os.path.join(seq_dir, "radar", "*.pcd")))
    if len(files) != len(gt["t"]):
        raise ValueError(f"{seq_dir}: {len(files)} PCD files but {len(gt['t'])} ground-truth rows")

    metas = [load_pcd_meta(f) for f in files]
    pos = gt["pose"][:, :3, 3]
    speed = np.linalg.norm(gt["v_body"], axis=1)
    return {
        "t": gt["t"],
        "pos": pos,
        "speed": speed,
        "yaw_rate": gt["w_body"][:, 2],
        "num_points": np.array([m["num_points"] for m in metas]),
        "v_amb": np.array([m["velocity_ambiguity"] for m in metas]),
        "timestamp_ns": metas[0]["timestamp_ns"] if metas else None,
    }


def summarize(name, s):
    step = np.linalg.norm(np.diff(s["pos"], axis=0), axis=1)
    # Static targets straight ahead show radial velocity -speed, which wraps once
    # speed exceeds half the unambiguous interval of that frame
    aliased = s["speed"] > s["v_amb"] / 2
    return {
        "sequence": name,
        "frames": len(s["t"]),
        "duration_min": round(s["t"][-1] / 60, 2),
        "distance_km": round(step.sum() / 1000, 2),
        "mean_speed_kmh": round(s["speed"].mean() * MS_TO_KMH, 1),
        "max_speed_kmh": round(s["speed"].max() * MS_TO_KMH, 1),
        "height_range_m": round(np.ptp(s["pos"][:, 2]), 1),
        "frame_rate_hz": round((len(s["t"]) - 1) / s["t"][-1], 2),
        "mean_detections": round(s["num_points"].mean(), 1),
        "aliased_frames_pct": round(100 * aliased.mean(), 1),
    }


def _route(ax, s, norm):
    xy = s["pos"][:, :2]
    segments = np.stack([xy[:-1], xy[1:]], axis=1)
    lc = LineCollection(segments, cmap=SPEED_CMAP, norm=norm, linewidths=2, capstyle="round")
    lc.set_array(s["speed"][:-1] * MS_TO_KMH)
    ax.add_collection(lc)
    # Start ring drawn larger than the end square so loops (start == end) show both
    ax.plot(*xy[0], "o", ms=13, mfc="none", mec=INK["primary"], mew=1.8, zorder=3)
    ax.plot(*xy[-1], "s", ms=6, mfc=INK["primary"], mec="#fcfcfb", mew=1, zorder=4)
    ax.autoscale()
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    return lc


def plot_overview(stats, summaries, out_path):
    n = len(stats)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.8 * rows), squeeze=False)
    vmax = max(s["speed"].max() for s in stats.values()) * MS_TO_KMH
    norm = Normalize(0, vmax)

    for ax, (name, s) in zip(axes.flat, stats.items()):
        sm = summaries[name]
        lc = _route(ax, s, norm)
        ax.set_title(f"{name}  ·  {sm['distance_km']} km  ·  {sm['duration_min']:.1f} min", loc="left")
    for ax in axes.flat[n:]:
        ax.set_visible(False)

    cbar = fig.colorbar(lc, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("speed [km/h]")
    cbar.outline.set_visible(False)
    fig.suptitle(
        "Routes in local world frame (○ start, ■ end)", x=0.02, ha="left",
        color=INK["primary"], fontsize=12, fontweight="semibold",
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_profile(name, s, summary, out_path):
    fig = plt.figure(figsize=(14, 7.5))
    grid = fig.add_gridspec(4, 2, width_ratios=[1.1, 1], hspace=0.35, wspace=0.18)
    ax_route = fig.add_subplot(grid[:, 0])
    lc = _route(ax_route, s, Normalize(0, s["speed"].max() * MS_TO_KMH))
    cbar = fig.colorbar(lc, ax=ax_route, shrink=0.5, pad=0.02)
    cbar.set_label("speed [km/h]")
    cbar.outline.set_visible(False)
    ax_route.set_title("Route (○ start, ■ end)", loc="left")

    t = s["t"] / 60
    axes = [fig.add_subplot(grid[i, 1]) for i in range(4)]
    for ax in axes[1:]:
        ax.sharex(axes[0])

    axes[0].plot(t, s["speed"] * MS_TO_KMH, color=SERIES, lw=1.2)
    # Two scan modes alternate frame-to-frame; draw the threshold of each
    for v_amb in np.unique(np.round(s["v_amb"], 1)):
        axes[0].axhline(v_amb / 2 * MS_TO_KMH, color=REFERENCE, lw=1, ls="--")
        axes[0].annotate(
            f"Doppler wrap at {v_amb / 2 * MS_TO_KMH:.0f} km/h (v_amb {v_amb:.1f} m/s)", (1, v_amb / 2 * MS_TO_KMH),
            xycoords=("axes fraction", "data"), xytext=(-4, 3), textcoords="offset points",
            ha="right", fontsize=7.5, color=INK["secondary"],
        )
    axes[0].set_ylabel("speed\n[km/h]")
    axes[0].set_title(
        f"{summary['aliased_frames_pct']}% of frames above their Doppler wrap speed", loc="left",
    )

    axes[1].plot(t, s["pos"][:, 2], color=SERIES, lw=1.2)
    axes[1].set_ylabel("height\n[m]")

    axes[2].plot(t, np.degrees(s["yaw_rate"]), color=SERIES, lw=0.8)
    axes[2].set_ylabel("yaw rate\n[deg/s]")

    axes[3].plot(t, s["num_points"], color=SERIES, lw=0.6)
    axes[3].set_ylabel("detections\n/ frame")
    axes[3].set_xlabel("time since start [min]")
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)

    fig.suptitle(
        f"{name}  ·  {summary['frames']:,} frames  ·  {summary['distance_km']} km  ·  "
        f"{summary['duration_min']:.1f} min  ·  max {summary['max_speed_kmh']:.0f} km/h",
        x=0.02, ha="left", color=INK["primary"], fontsize=12, fontweight="semibold",
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Route plots and statistics for the ZF radar dataset")
    parser.add_argument("-p", "--path", required=True, help="Dataset root containing sequence directories.")
    parser.add_argument("-o", "--out", default="plots", help="Output directory (default: ./plots)")
    parser.add_argument("--seq", nargs="+", help="Only these sequences (default: all)")
    args = parser.parse_args()
    if not os.path.isdir(args.path):
        parser.error(f"dataset path is not a directory: {args.path}")

    sequences = args.seq or discover_sequences(args.path)
    os.makedirs(args.out, exist_ok=True)
    _style()

    stats, summaries = {}, {}
    for name in sequences:
        print(f"reading {name} ...", flush=True)
        stats[name] = load_sequence_stats(os.path.join(args.path, name))
        summaries[name] = summarize(name, stats[name])
        plot_profile(name, stats[name], summaries[name], os.path.join(args.out, f"{name}_profile.png"))

    plot_overview(stats, summaries, os.path.join(args.out, "routes_overview.png"))

    rows = list(summaries.values())
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    widths = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in rows[0]}
    print("  ".join(k.rjust(w) for k, w in widths.items()))
    for r in rows:
        print("  ".join(str(r[k]).rjust(w) for k, w in widths.items()))
    print(f"wrote plots and summary.csv to {args.out}")


if __name__ == "__main__":
    main()
