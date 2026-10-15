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
    assert odom.pose.pose.position.x == pytest.approx(2.0)
    assert odom.pose.pose.orientation.z == pytest.approx(np.sin(0.15))
    assert odom.twist.twist.linear.x == pytest.approx(5.0)
    assert len(topics["/ground_truth/path"][0][1].poses) == 2


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
