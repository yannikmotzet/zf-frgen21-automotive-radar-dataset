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
2. **Moving targets.** Each remaining point is associated with a nearby point
   of the neighbouring frame (ego motion removed), which uses the other scan
   mode. The two wrapped measurements are consistent with a single velocity
   within about ±65 m/s (staggered PRF); it is accepted if it also explains the
   range change between the frames (status 2).
3. **Objects.** Moving points are clustered into objects and tracked over
   time; all points of an object take the wrap count that matches its
   velocity, from its paired points (status 3) or from its track (status 4).
4. Everything else keeps its measured value (status 0).

Radial velocity is the range rate in the sensor frame, negative = approaching.
"""

import numpy as np

STATUS_UNRESOLVED = 0
STATUS_STATIC = 1
STATUS_TRACKED = 2
STATUS_OBJECT = 3
STATUS_PROPAGATED = 4


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


def _in_frame(partner, dt, sensor_velocity, yaw_rate):
    """Partner points expressed in the current sensor frame as if the world were
    static: the sensor moved by ``sensor_velocity * dt`` and turned by ``yaw_rate * dt``."""
    p = np.column_stack([partner["x"], partner["y"], partner["z"]]).astype(float) - np.asarray(sensor_velocity) * dt
    c, s = np.cos(yaw_rate * dt), np.sin(yaw_rate * dt)
    return np.column_stack([c * p[:, 0] + s * p[:, 1], -s * p[:, 0] + c * p[:, 1], p[:, 2]])


def associate_moving(frame, partner, candidates, dt, sensor_velocity, yaw_rate=0.0,
                     max_daz=np.radians(3.0), max_del=np.radians(2.5), max_lateral=2.0, max_vertical=1.0,
                     max_radial=1.5, consistency=1.0, max_speed=50.0, max_k=3):
    """Unwrap ``candidates`` of ``frame`` against the neighbouring ``partner`` frame.

    ``dt`` is time(frame) - time(partner) in seconds. Partner points are moved
    into the current frame with the ego motion; a pair is gated by its angular
    offset (within ``max_daz`` / ``max_del``, or within ``max_lateral`` /
    ``max_vertical`` metres at the point's range, which matters for close
    targets). Both frames use different ambiguities, so the pair has a single
    velocity within about ±65 m/s that is consistent with both wrapped
    measurements (within ``consistency`` m/s). It is accepted if it also explains
    the range change (within ``max_radial`` m) and is at most ``max_speed`` m/s
    over ground. Returns ``(unwrapped, ok)`` for the candidate indices.
    """
    A_t = frame["_meta"]["velocity_ambiguity"]
    A_q = partner["_meta"]["velocity_ambiguity"]
    idx = np.flatnonzero(candidates)
    m_all = np.asarray(frame["velocity"], float)
    unwrapped = m_all[idx].copy()
    ok = np.zeros(len(idx), bool)
    if len(idx) == 0 or len(partner["velocity"]) == 0:
        return unwrapped, ok

    r_t, az_t, el_t = (a[idx] for a in polar(frame))
    m_t = m_all[idx]
    pred = -(unit_vectors(frame)[idx] @ np.asarray(sensor_velocity))     # static-world radial velocity
    q = _in_frame(partner, dt, sensor_velocity, yaw_rate)
    _, az_q, el_q = polar({"x": q[:, 0], "y": q[:, 1], "z": q[:, 2]})
    r_q = polar(partner)[0]
    m_q = np.asarray(partner["velocity"], float)

    daz = np.abs(wrap(az_t[:, None] - az_q[None, :], 2 * np.pi))
    dl = np.abs(el_t[:, None] - el_q[None, :])
    gate = ((daz < max_daz) | (r_t[:, None] * daz < max_lateral)) & \
        ((dl < max_del) | (r_t[:, None] * dl < max_vertical))
    i, j = np.nonzero(gate)
    if len(i) == 0:
        return unwrapped, ok

    k = np.arange(-max_k, max_k + 1)
    v_t = m_t[i, None] + k[None, :] * A_t                                # (pairs, k)
    v_q = m_q[j, None] + np.round((v_t - m_q[j, None]) / A_q) * A_q
    mismatch = np.abs(v_t - v_q)
    radial = np.abs((r_t[i] - r_q[j])[:, None] - 0.5 * (v_t + v_q) * dt)
    valid = (mismatch < consistency) & (radial < max_radial) & (np.abs(v_t - pred[i, None]) < max_speed)
    lateral = (r_t[i] * daz[i, j])[:, None]
    cost = np.where(valid, mismatch / consistency + radial / max_radial + lateral / max_lateral, np.inf)
    kb = np.argmin(cost, axis=1)
    pair_cost = cost[np.arange(len(i)), kb]
    pair_v = v_t[np.arange(len(i)), kb]

    order = np.lexsort((pair_cost, i))                                    # best pair per candidate first
    first = order[np.unique(i[order], return_index=True)[1]]
    good = first[np.isfinite(pair_cost[first])]
    unwrapped[i[good]] = pair_v[good]
    ok[i[good]] = True
    return unwrapped, ok


def clusters(xy, eps=2.0):
    """Labels of single-linkage clusters (points closer than ``eps``)."""
    n = len(xy)
    labels = np.arange(n)
    if n < 2:
        return labels
    i, j = np.nonzero(np.triu(np.sum((xy[:, None] - xy[None]) ** 2, axis=-1) < eps * eps, 1))
    while True:
        a = np.minimum(labels[i], labels[j])
        if np.all((labels[i] == a) & (labels[j] == a)):
            break
        np.minimum.at(labels, i, a)
        np.minimum.at(labels, j, a)
        labels = labels[labels]
    return np.unique(labels, return_inverse=True)[1]


class ObjectTracker:
    """Object-level de-aliasing of moving points across frames.

    Per frame, the non-static points are clustered into objects. An object's
    velocity over ground along the line of sight (``g``) comes from its pair-
    resolved points (median) or, if fewer than ``min_support`` of them agree,
    from the object's track. Every point of the object then takes the wrap count
    that brings it closest to ``static prediction + g``, if within ``tolerance``
    (status 3 from the frame's own points, status 4 from the track); this also
    overrides single wrongly paired points. Objects are tracked by their
    centroid in an odometry frame integrated from the radar ego velocity; a
    track keeps its ``g`` for up to ``max_coast`` frames without support.
    """

    def __init__(self, lever_x=3.925, eps=2.0, tolerance=4.0, gate=4.0, min_support=3, max_missed=5, max_coast=10):
        self.lever_x, self.eps, self.tolerance, self.gate = lever_x, eps, tolerance, gate
        self.min_support, self.max_missed, self.max_coast = min_support, max_missed, max_coast
        self.pose = np.zeros(3)                       # x, y, yaw of the sensor in the odometry frame
        self.tracks = []                              # dicts: pos, vel, g, coast, missed

    def _move(self, v, dt):
        yaw = self.pose[2] + 0.5 * v[1] / self.lever_x * dt
        c, s = np.cos(yaw), np.sin(yaw)
        self.pose[:2] += dt * np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])
        self.pose[2] += v[1] / self.lever_x * dt

    def _to_odom(self, xy):
        c, s = np.cos(self.pose[2]), np.sin(self.pose[2])
        return self.pose[:2] + xy @ np.array([[c, s], [-s, c]])

    def _match(self, centroids, dt):
        pred = np.array([t["pos"] + t["vel"] * dt for t in self.tracks]).reshape(-1, 2)
        match = -np.ones(len(centroids), int)
        if len(pred) == 0 or len(centroids) == 0:
            return match
        d = np.linalg.norm(centroids[:, None] - pred[None], axis=-1)
        used = np.zeros(len(pred), bool)
        for flat in np.argsort(d, axis=None):
            c, t = np.unravel_index(flat, d.shape)
            if d[c, t] >= self.gate:
                break
            if match[c] < 0 and not used[t]:
                match[c], used[t] = t, True
        return match

    def update(self, frame, res, dt=0.0):
        """Refine ``res`` (an ``unwrap_frame`` result) in place. ``dt`` is the
        time since the previous frame passed to ``update``."""
        v = res["sensor_velocity"]
        self._move(v, dt)
        status, vel = res["unwrap_status"], res["velocity_unwrapped"]
        m = np.asarray(frame["velocity"], float)
        A = frame["_meta"]["velocity_ambiguity"]
        mov = np.flatnonzero(status != STATUS_STATIC)
        xy = np.column_stack([frame["x"], frame["y"]]).astype(float)[mov]
        pred = -(unit_vectors(frame)[mov] @ v)
        labels = clusters(xy, self.eps)
        n = labels.max() + 1 if len(labels) else 0
        cent = self._to_odom(np.array([xy[labels == c].mean(axis=0) for c in range(n)]).reshape(-1, 2))
        match = self._match(cent, dt)

        seen = np.zeros(len(self.tracks), bool)
        new_tracks = []
        for c in range(n):
            sel = np.flatnonzero(labels == c)
            pts = mov[sel]
            g_pts = vel[pts] - pred[sel]
            paired = status[pts] == STATUS_TRACKED
            trk = self.tracks[match[c]] if match[c] >= 0 else None
            g_trk = trk["g"] if trk is not None and trk["g"] is not None and trk["coast"] <= self.max_coast else None
            g_vote = np.median(g_pts[paired]) if paired.any() else None
            ref, source = None, None
            if g_vote is not None and (paired.sum() >= self.min_support or g_trk is None
                                       or abs(g_vote - g_trk) < self.tolerance):
                ref, source = g_vote, STATUS_OBJECT
            elif g_trk is not None:
                ref, source = g_trk, STATUS_PROPAGATED
            g_new = None
            if ref is not None:
                v_ref = pred[sel] + ref
                v_new = m[pts] + np.round((v_ref - m[pts]) / A) * A
                fit = np.abs(v_new - v_ref) < self.tolerance
                changed = fit & ((status[pts] == STATUS_UNRESOLVED) | (np.abs(v_new - vel[pts]) > 1e-3))
                vel[pts[fit]] = v_new[fit]
                status[pts[changed]] = source
                if fit.any():
                    g_new = float(np.median(v_new[fit] - pred[sel][fit]))
            if trk is not None:
                seen[match[c]] = True
                if dt > 0:
                    trk["vel"] = 0.5 * trk["vel"] + 0.5 * (cent[c] - trk["pos"]) / dt
                trk["pos"], trk["missed"] = cent[c], 0
                if g_new is not None:
                    trk["g"], trk["coast"] = g_new, 0
                else:
                    trk["coast"] += 1
            else:
                new_tracks.append({"pos": cent[c], "vel": np.zeros(2), "g": g_new, "coast": 0, "missed": 0})
        for t, s in zip(self.tracks, seen):
            if not s:
                t["missed"] += 1
                t["coast"] += 1
        self.tracks = [t for t in self.tracks if t["missed"] <= self.max_missed] + new_tracks
        res["wrap_k"] = np.round((vel - m) / A).astype(np.int8)
        return res


def unwrap_frame(frame, partners=(), prior=None, lever_x=3.925, threshold=0.25, wide=False,
                 consistency=1.0):
    """Unwrap one frame. ``partners`` is a sequence of ``(neighbour_frame, dt)``
    with ``dt = time(frame) - time(neighbour)`` (one or two neighbours).

    A moving point is resolved if a neighbour resolves it; with two neighbours
    that both resolve it, they must agree within ``consistency`` m/s, otherwise
    the point stays unresolved (conflict).

    Returns dict with ``velocity_unwrapped``, ``wrap_k``, ``unwrap_status`` (per
    point), ``sensor_velocity`` (3,), ``static_fraction`` and the moving-point
    counts ``n_both`` (resolved by both neighbours) and ``n_conflict``.
    """
    m = np.asarray(frame["velocity"], float)
    A = frame["_meta"]["velocity_ambiguity"]
    u = unit_vectors(frame)
    v, static, static_fraction = estimate_sensor_velocity(u, m, A, prior, threshold, wide)

    unwrapped = m.copy()
    status = np.full(len(m), STATUS_UNRESOLVED, np.int8)
    pred = -(u @ v)
    unwrapped[static] = m[static] + np.round((pred[static] - m[static]) / A) * A
    status[static] = STATUS_STATIC

    n_both = n_conflict = 0
    moving = np.flatnonzero(~static)
    if partners and len(moving):
        # vehicle has no lateral velocity at the rear axle: sensor vy = yaw_rate * lever
        yaw_rate = v[1] / lever_x
        sols = [associate_moving(frame, partner, ~static, dt, v, yaw_rate, consistency=consistency)
                for partner, dt in partners]
        vel, ok = sols[0]
        if len(sols) == 2:
            (v1, ok1), (v2, ok2) = sols
            both = ok1 & ok2
            conflict = both & (np.abs(v1 - v2) >= consistency)
            n_both, n_conflict = int(both.sum()), int(conflict.sum())
            vel = np.where(ok1, v1, v2)
            ok = (ok1 | ok2) & ~conflict
        unwrapped[moving[ok]] = vel[ok]
        status[moving[ok]] = STATUS_TRACKED

    return {
        "velocity_unwrapped": unwrapped,
        "wrap_k": np.round((unwrapped - m) / A).astype(np.int8),
        "unwrap_status": status,
        "sensor_velocity": v,
        "static_fraction": static_fraction,
        "n_both": n_both,
        "n_conflict": n_conflict,
    }


def unwrap_sequence(frames, two_sided=False, lever_x=3.925, threshold=0.25, lost_fraction=0.2, lost_frames=5,
                    objects=True):
    """Yield ``(frame, result)`` for consecutive frames (any iterable of
    ``load_pcd`` dicts).

    Causal by default: each frame's moving points are resolved against its
    predecessor (the first frame against its successor), so it can run online.
    ``two_sided=True`` also uses the successor (offline; one frame of lookahead),
    which resolves more moving points and rejects inconsistent ones. The
    ego-velocity search is seeded with the previous frame's estimate; after
    ``lost_frames`` frames in a row with a static inlier fraction below
    ``lost_fraction`` the wide search is re-enabled. With ``objects`` the
    moving points are refined per object and over time (``ObjectTracker``).
    """
    it = iter(frames)
    cur = next(it, None)
    if cur is None:
        return
    nxt = next(it, None)
    ts = lambda f: f["_meta"]["timestamp_ns"] * 1e-9
    prev, prior, lost = None, None, 0
    tracker = ObjectTracker(lever_x) if objects else None
    while cur is not None:
        partners = []
        if prev is not None:
            partners.append((prev, ts(cur) - ts(prev)))
        if nxt is not None and (two_sided or prev is None):
            partners.append((nxt, ts(cur) - ts(nxt)))
        res = unwrap_frame(cur, partners, prior, lever_x, threshold,
                           wide=prior is not None and lost >= lost_frames)
        if tracker is not None:
            tracker.update(cur, res, ts(cur) - ts(prev) if prev is not None else 0.0)
        yield cur, res
        lost = lost + 1 if res["static_fraction"] < lost_fraction else 0
        prior = res["sensor_velocity"]
        prev, cur, nxt = cur, nxt, next(it, None)
