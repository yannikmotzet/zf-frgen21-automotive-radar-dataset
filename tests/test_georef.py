"""Dataset-keyed encode/decode of start poses (synthetic poses only)."""

import json

import pytest

from zf_radar.georef import decode, encode, sequence_key


def make_seq(root, name, content):
    seq = root / name
    seq.mkdir()
    (seq / "ground_truth.csv").write_text(content)
    return seq


def test_round_trip_and_partial_dataset(tmp_path):
    a = make_seq(tmp_path, "zf_01", "t,T00\n0,1\n")
    b = make_seq(tmp_path, "zf_02", "t,T00\n0,1\n0.09,1\n")
    entries = {
        "zf_01": {"T_utm_world": [[1, 0, 0, 100.0], [0, 1, 0, 200.0], [0, 0, 1, 3.0], [0, 0, 0, 1]]},
        "zf_02": {"T_utm_world": [[0, -1, 0, 7.5], [1, 0, 0, 8.5], [0, 0, 1, 0.0], [0, 0, 0, 1]]},
    }
    container = encode(entries, {"zf_01": sequence_key(a), "zf_02": sequence_key(b)}, {"crs": "test"})

    blob = json.dumps(container)
    assert "100.0" not in blob and "T_utm_world" not in blob

    out = decode(container, str(tmp_path))
    assert out["sequences"] == entries
    assert out["meta"] == {"crs": "test"}

    # only zf_01 downloaded: zf_02 is skipped, not an error
    (b / "ground_truth.csv").unlink()
    assert list(decode(container, str(tmp_path))["sequences"]) == ["zf_01"]


def test_wrong_dataset_version_raises(tmp_path):
    seq = make_seq(tmp_path, "zf_01", "t,T00\n0,1\n")
    container = encode({"zf_01": {"x": 1}}, {"zf_01": sequence_key(seq)})
    (seq / "ground_truth.csv").write_text("t,T00\n0,2\n")
    with pytest.raises(ValueError, match="does not match"):
        decode(container, str(tmp_path))
