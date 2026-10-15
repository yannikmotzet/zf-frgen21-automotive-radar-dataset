"""Doppler de-aliasing for the ZF FRGen21 radar frames (numpy only, radar-only).

Measured radial velocities wrap into ±A/2, where A = ``velocity_ambiguity``
alternates between two scan modes on consecutive frames. Unwrapping uses no
ground truth:

1. **Static targets.** Per frame, the sensor's ego velocity is found by a search
   over velocity hypotheses scored with a *wrapped* residual (static points
   satisfy ``m = wrap(-u · v)``), then refined by least squares on the inliers.
   After the first frame the search stays in a small window around the previous
   estimate (the vehicle cannot change speed by metres per second within one
   frame), so large moving objects at standstill cannot pull it away; a wide
   search runs only at the start or after the track was lost for several frames.
   Inliers are unwrapped to ``-u · v`` (status 1).
2. **Moving targets.** Each remaining point is associated with a point of the
   neighbouring frame, which uses the other scan mode. The range change between
   the two gives a coarse radial velocity; it is accepted only if both wrapped
   measurements unwrap to the same velocity (staggered-PRF consistency, status 2).
3. Everything else keeps its measured value (status 0).

Radial velocity is the range rate in the sensor frame, negative = approaching.
"""

import numpy as np

STATUS_UNRESOLVED = 0
STATUS_STATIC = 1
STATUS_TRACKED = 2


def wrap(v, ambiguity):
    """Fold velocities into [-A/2, A/2)."""
    return (np.asarray(v) + ambiguity / 2) % ambiguity - ambiguity / 2


def polar(frame):
    """(range, azimuth, elevation) of a frame; derived from x/y/z if not stored
    (azimuth = atan2(y, x), elevation = arcsin(z / range))."""
    if all(k in frame for k in ("range", "azimuth", "elevation")):
        return tuple(np.asarray(frame[k], float) for k in ("range", "azimuth", "elevation"))
    x, y, z = (np.asarray(frame[k], float) for k in ("x", "y", "z"))
    r = np.sqrt(x * x + y * y + z * z)
    return r, np.arctan2(y, x), np.arcsin(np.clip(z / np.maximum(r, 1e-6), -1.0, 1.0))


def unit_vectors(frame):
    u = np.column_stack([frame["x"], frame["y"], frame["z"]]).astype(float)
    return u / np.maximum(polar(frame)[0], 1e-6)[:, None]


def _score(u, m, ambiguity, hyps, threshold):
    """Soft inlier count of every (vx, vy) hypothesis; residual is wrap-aware."""
    pred = -(u[:, :2] @ hyps.T)                          # (N, H)
    r = wrap(m[:, None] - pred, ambiguity) / threshold
    return np.sum(np.clip(1.0 - r * r, 0.0, None), axis=0)


def _grid(center, half_x, half_y, step, lateral_ratio):
    gx = center[0] + np.arange(-half_x, half_x + step / 2, step)
    gy = center[1] + np.arange(-half_y, half_y + step / 2, step)
    g = np.array(np.meshgrid(gx, gy, indexing="ij")).reshape(2, -1).T
    return g[np.abs(g[:, 1]) <= _lateral_limit(g[:, 0], lateral_ratio)]


def _lateral_limit(vx, lateral_ratio, margin=0.2):
    """Max lateral sensor velocity for a car: v_y = yaw_rate * lever with
    |yaw_rate| <= |v_x| / R_min, i.e. |v_y| <= (lever / R_min) |v_x| (+ margin)."""
    return lateral_ratio * np.abs(vx) + margin


def estimate_sensor_velocity(u, m, ambiguity, prior=None, threshold=0.25, wide=False, regularization=20.0,
                             lateral_ratio=3.925 / 5.0):
    """Ego velocity of the sensor (sensor frame, m/s) from one frame's Doppler.

    Returns ``(velocity (3,), inlier mask, inlier fraction)``. With a ``prior``
    (previous estimate) only a local window is searched, unless ``wide`` is set;
    without a prior the wide search always runs. ``lateral_ratio`` (lever arm /
    minimum turning radius) bounds the lateral velocity a car can have at the
    sensor, which keeps sparse standstill frames from drifting sideways.
    """
    if len(m) < 3:
        v = np.zeros(3) if prior is None else np.asarray(prior, float)
        return v, np.zeros(len(m), bool), 0.0

    best_v, best_s = None, -np.inf
    if prior is not None:
        # ~1 g longitudinally and ~10 deg/s yaw-rate change within one 90 ms frame
        hyps = _grid(prior[:2], 1.0, 0.8, 0.1, lateral_ratio)
        s = _score(u, m, ambiguity, hyps, threshold)
        best_v, best_s = hyps[np.argmax(s)], s.max()
    if prior is None or wide:
        hyps = _grid((25.0, 0.0), 35.0, 3.0, 0.25, lateral_ratio)
        s = _score(u, m, ambiguity, hyps, threshold)
        if best_v is None or s.max() > 1.5 * best_s:
            best_v, best_s = hyps[np.argmax(s)], s.max()

    v_grid = np.array([best_v[0], best_v[1], 0.0])
    v = v_grid
    inliers = np.zeros(len(m), bool)
    # ridge rows pull the refinement towards the grid solution: few or badly spread
    # inliers (standstill) cannot drag it off, many well-spread ones still dominate
    ridge = np.sqrt(regularization) * np.eye(3)
    for _ in range(3):
        pred = -(u @ v)
        k = np.round((pred - m) / ambiguity)
        unwrapped = m + k * ambiguity
        inliers = np.abs(unwrapped - pred) < threshold
        if inliers.sum() < 3:
            break
        A = np.vstack([u[inliers], ridge])
        b = np.concatenate([-unwrapped[inliers], ridge @ v_grid])
        v = np.linalg.lstsq(A, b, rcond=None)[0]
        limit = _lateral_limit(v[0], lateral_ratio)
        v[1] = np.clip(v[1], -limit, limit)
    return v, inliers, inliers.mean()


def associate_moving(frame, partner, candidates, dt, yaw_rate=0.0,
                     max_daz=np.radians(3.0), max_del=np.radians(2.5),
                     max_dr=6.0, consistency=1.0):
    """Unwrap ``candidates`` of ``frame`` against the neighbouring ``partner`` frame.

    ``dt`` is time(frame) - time(partner) in seconds. ``max_dr`` must cover the
    range change of fast closing traffic (2 x 100 km/h ~ 5 m per 90 ms frame).
    Returns ``(unwrapped, ok)`` for the candidate indices.
    """
    A_t = frame["_meta"]["velocity_ambiguity"]
    A_q = partner["_meta"]["velocity_ambiguity"]
    idx = np.flatnonzero(candidates)
    unwrapped = np.asarray(frame["velocity"], float)[idx].copy()
    ok = np.zeros(len(idx), bool)
    if len(idx) == 0 or len(partner["velocity"]) == 0:
        return unwrapped, ok

    r_t, az_t, el_t = (a[idx] for a in polar(frame))
    m_t = np.asarray(frame["velocity"], float)[idx]
    r_q, az_q, el_q = polar(partner)
    m_q = np.asarray(partner["velocity"], float)
    # static scene rotates by -yaw * dt in the sensor frame between the frames
    daz = np.abs(az_t[:, None] - (az_q[None, :] - yaw_rate * dt))
    gate = (daz < max_daz) & (np.abs(el_t[:, None] - el_q[None, :]) < max_del) \
        & (np.abs(r_t[:, None] - r_q[None, :]) < max_dr)

    rr = (r_t[:, None] - r_q[None, :]) / dt          # coarse range rate per pair
    v_t = m_t[:, None] + np.round((rr - m_t[:, None]) / A_t) * A_t
    v_q = m_q[None, :] + np.round((rr - m_q[None, :]) / A_q) * A_q
    mismatch = np.abs(v_t - v_q)
    valid = gate & (mismatch < consistency) & (np.abs(v_t - rr) < 0.25 * min(A_t, A_q))
    cost = np.where(valid, mismatch + daz / max_daz, np.inf)
    best = np.argmin(cost, axis=1)
    ok = np.isfinite(cost[np.arange(len(idx)), best])
    unwrapped[ok] = v_t[np.arange(len(idx)), best][ok]
    return unwrapped, ok


def unwrap_frame(frame, partner, dt, prior=None, lever_x=3.925, threshold=0.25, wide=False):
    """Unwrap one frame. Returns dict with ``velocity_unwrapped``, ``wrap_k``,
    ``unwrap_status`` (per point), ``sensor_velocity`` (3,) and ``static_fraction``."""
    m = np.asarray(frame["velocity"], float)
    A = frame["_meta"]["velocity_ambiguity"]
    u = unit_vectors(frame)
    v, static, static_fraction = estimate_sensor_velocity(u, m, A, prior, threshold, wide)

    unwrapped = m.copy()
    status = np.full(len(m), STATUS_UNRESOLVED, np.int8)
    pred = -(u @ v)
    unwrapped[static] = m[static] + np.round((pred[static] - m[static]) / A) * A
    status[static] = STATUS_STATIC

    if partner is not None:
        # vehicle has no lateral velocity at the rear axle: sensor vy = yaw_rate * lever
        yaw_rate = v[1] / lever_x
        moving_v, ok = associate_moving(frame, partner, ~static, dt, yaw_rate)
        moving = np.flatnonzero(~static)
        unwrapped[moving[ok]] = moving_v[ok]
        status[moving[ok]] = STATUS_TRACKED

    return {
        "velocity_unwrapped": unwrapped,
        "wrap_k": np.round((unwrapped - m) / A).astype(np.int8),
        "unwrap_status": status,
        "sensor_velocity": v,
        "static_fraction": static_fraction,
    }


def unwrap_sequence(frames, lever_x=3.925, threshold=0.25, lost_fraction=0.2, lost_frames=5):
    """Yield ``(frame, result)`` for consecutive frames (any iterable of
    ``load_pcd`` dicts). Each frame is paired with its predecessor, the first
    with its successor; the ego-velocity search is seeded with the previous
    frame's estimate. After ``lost_frames`` frames in a row with a static
    inlier fraction below ``lost_fraction`` the wide search is re-enabled."""
    it = iter(frames)
    prev = next(it, None)
    if prev is None:
        return
    nxt = next(it, None)
    ts = lambda f: f["_meta"]["timestamp_ns"] * 1e-9
    res = unwrap_frame(prev, nxt, ts(prev) - ts(nxt) if nxt is not None else 0.0,
                       None, lever_x, threshold)
    yield prev, res
    prior, lost = res["sensor_velocity"], 0
    while nxt is not None:
        cur, nxt = nxt, next(it, None)
        res = unwrap_frame(cur, prev, ts(cur) - ts(prev), prior, lever_x, threshold,
                           wide=lost >= lost_frames)
        yield cur, res
        lost = lost + 1 if res["static_fraction"] < lost_fraction else 0
        prev, prior = cur, res["sensor_velocity"]
