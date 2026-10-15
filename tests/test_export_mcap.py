"""Round-trip tests for the MCAP exporter on a tiny synthetic sequence."""

import numpy as np
import pytest

pytest.importorskip("mcap_ros2")

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

from zf_radar.export_mcap import export_sequence, rotation_to_quaternion


T0 = 1626684577435105592

PCD = """\
# .PCD v0.7 - Point Cloud Data file format
# timestamp_ns {ts}
# velocity_ambiguity {vamb}
VERSION 0.7
FIELDS x y z velocity snr
SIZE 4 4 4 4 4
TYPE F F F F F
COUNT 1 1 1 1 1
WIDTH {n}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {n}
DATA ascii
"""


def yaw_matrix(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@pytest.fixture
def sequence(tmp_path):
    seq = tmp_path / "zf_01"
    (seq / "radar").mkdir(parents=True)
    (tmp_path / "extrinsics.csv").write_text("h\n1,0,0,3.925,0,1,0,0,0,0,1,0\n")
    frames = [
        (T0, 32.499187, [[1.0, 2.0, 0.5, -3.25, 20.0], [10.0, -1.0, 0.0, 4.5, 31.0]]),
        (T0 + 90_000_000, 43.6, [[5.0, 0.0, 1.0, 0.0, 15.0]]),
    ]
    rows = []
    for i, (ts, vamb, pts) in enumerate(frames):
        body = "\n".join(" ".join(str(v) for v in p) for p in pts) + "\n"
        (seq / "radar" / f"frame_{i:06d}.pcd").write_text(PCD.format(ts=ts, vamb=vamb, n=len(pts)) + body)
        R = yaw_matrix(0.3 * i)
        T = np.hstack([R, [[2.0 * i], [0.5 * i], [0.0]]])
        rows.append([(ts - T0) * 1e-9, *T.reshape(-1), 5.0, 0.0, 0.0, 0.0, 0.0, 0.1])
    np.savetxt(seq / "ground_truth.csv", rows, delimiter=",", header="t,...", comments="")
    return seq, frames


def read_all(path):
    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        out = {}
        for schema, channel, message, msg in reader.iter_decoded_messages():
            out.setdefault(channel.topic, []).append((message.log_time, msg, channel))
        return out


def test_round_trip(sequence, tmp_path):
    seq, frames = sequence
    out = tmp_path / "zf_01.mcap"
    assert export_sequence(str(seq), str(out), progress=False) == 2
    topics = read_all(out)

    clouds = topics["/radar/points"]
    assert [t for t, _, _ in clouds] == [frames[0][0], frames[1][0]]
    msg = clouds[0][1]
    assert msg.header.frame_id == "radar"
    assert [f.name for f in msg.fields] == ["x", "y", "z", "velocity", "snr"]
    pts = np.frombuffer(bytes(msg.data), dtype="<f4").reshape(msg.width, -1)
    np.testing.assert_allclose(pts, frames[0][2], rtol=1e-6)

    vamb = [m.data for _, m, _ in topics["/radar/velocity_ambiguity"]]
    np.testing.assert_allclose(vamb, [32.499187, 43.6], rtol=1e-6)

    static = topics["/tf_static"][0][1].transforms[0]
    assert (static.header.frame_id, static.child_frame_id) == ("base_link", "radar")
    assert static.transform.translation.x == pytest.approx(3.925)
    assert "durability: 1" in topics["/tf_static"][0][2].metadata["offered_qos_profiles"]

    odom = topics["/ground_truth/odom"][1][1]
    assert (odom.header.frame_id, odom.child_frame_id) == ("odom_ins", "base_link")
    tf = topics["/tf"][0][1].transforms[0]
    assert (tf.header.frame_id, tf.child_frame_id) == ("odom_ins", "base_link")
    assert odom.pose.pose.position.x == pytest.approx(2.0)
    assert odom.pose.pose.orientation.z == pytest.approx(np.sin(0.15))
    assert odom.twist.twist.linear.x == pytest.approx(5.0)
    assert len(topics["/ground_truth/path"][0][1].poses) == 2


def test_georef_topics(sequence, tmp_path):
    pyproj = pytest.importorskip("pyproj")
    seq, frames = sequence
    yaw = np.radians(30)
    T = np.eye(4)
    T[:3, :3] = yaw_matrix(yaw)
    T[:3, 3] = [535100.0, 5278200.0, 450.5]
    georef = {"T_utm_world": T.tolist(), "map_origin": [535000.0, 5278000.0, 450.0]}
    out = tmp_path / "geo.mcap"
    export_sequence(str(seq), str(out), progress=False, georef=georef, accuracy=(2.0, 3.0))
    topics = read_all(out)

    static = {t.child_frame_id: t for t in topics["/tf_static"][0][1].transforms}
    map_odom = static["odom_ins"]
    assert map_odom.header.frame_id == "map"
    translation = map_odom.transform.translation
    assert (translation.x, translation.y, translation.z) == pytest.approx((100.0, 200.0, 0.5))
    assert map_odom.transform.rotation.z == pytest.approx(np.sin(yaw / 2))
    assert "radar" in static  # extrinsics still there

    assert [t for t, _, _ in topics["/gnss/fix"]] == [frames[0][0], frames[1][0]]
    fix = topics["/gnss/fix"][1][1]
    # frame 1 sits at odom_ins (2.0, 0.5, 0)
    E, N = np.array([535100.0, 5278200.0]) + yaw_matrix(yaw)[:2, :2] @ [2.0, 0.5]
    lon, lat = pyproj.Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True).transform(E, N)
    assert fix.latitude == pytest.approx(lat, abs=1e-9)
    assert fix.longitude == pytest.approx(lon, abs=1e-9)
    assert fix.altitude == pytest.approx(450.5)
    assert fix.header.frame_id == "base_link"
    assert fix.position_covariance_type == 1
    assert [fix.position_covariance[i] for i in (0, 4, 8)] == pytest.approx([4.0, 4.0, 9.0])


def test_without_georef_has_no_map_frame(sequence, tmp_path):
    seq, _ = sequence
    out = tmp_path / "plain.mcap"
    export_sequence(str(seq), str(out), progress=False)
    topics = read_all(out)
    assert "/gnss/fix" not in topics
    assert [t.child_frame_id for t in topics["/tf_static"][0][1].transforms] == ["radar"]


def test_unwrap_doppler_fields(sequence, tmp_path):
    seq, frames = sequence
    out = tmp_path / "unwrap.mcap"
    export_sequence(str(seq), str(out), progress=False, unwrap_doppler=True)
    topics = read_all(out)

    cloud = topics["/radar/points"][0][1]
    names = [f.name for f in cloud.fields]
    assert names == ["x", "y", "z", "velocity", "snr", "velocity_unwrapped", "wrap_k", "unwrap_status"]
    types = {f.name: f.datatype for f in cloud.fields}
    assert (types["velocity_unwrapped"], types["wrap_k"], types["unwrap_status"]) == (7, 1, 1)
    assert cloud.point_step == 4 * 6 + 2

    # packed layout: 6 float32 then 2 int8 per point
    dtype = np.dtype([(n, "<f4") for n in names[:6]] + [("wrap_k", "i1"), ("unwrap_status", "i1")])
    pts = np.frombuffer(bytes(cloud.data), dtype=dtype)
    np.testing.assert_allclose(pts["x"], [p[0] for p in frames[0][2]], rtol=1e-6)
    # fewer than 3 points: nothing can be resolved, measured value is kept
    np.testing.assert_allclose(pts["velocity_unwrapped"], pts["velocity"])
    assert list(pts["unwrap_status"]) == [0, 0]

    assert [t for t, _, _ in topics["/radar/ego_velocity"]] == [frames[0][0], frames[1][0]]
    assert topics["/radar/ego_velocity"][0][1].header.frame_id == "radar"


@pytest.mark.parametrize("R", [
    np.eye(3),
    yaw_matrix(np.pi),  # trace < 0 branch
    np.diag([1.0, -1.0, -1.0]),
    np.diag([-1.0, 1.0, -1.0]),
])
def test_quaternion_matches_rotation(R):
    x, y, z, w = rotation_to_quaternion(R)
    back = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    np.testing.assert_allclose(back, R, atol=1e-12)
