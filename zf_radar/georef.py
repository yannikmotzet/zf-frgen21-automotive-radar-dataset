#!/usr/bin/env python3
"""Geographic start poses stored keyed by the dataset itself.

The published dataset carries no geodetic coordinates. Start poses obtained
separately (e.g. by matching the trajectories against a map) can be stored in
obfuscated form: each sequence's entry is XOR-ed with a SHA-256 keystream
derived from that sequence's ``ground_truth.csv``, so it can only be read
alongside the dataset (Zenodo v1.0.0). This is not access control — anyone with
the dataset can decode it — it only keeps the coordinates out of plain text.

Decode to JSON (stdlib only):
    python -m zf_radar.georef -p /path/to/ZF_Dataset --enc poses.enc -o georef_poses.json

Decoded format (per sequence):
    T_utm_world   4x4, world frame -> UTM 32N grid (E, N, ellipsoidal height)
    map_origin    shared origin of the ``map`` frame in the same coordinates
"""

import argparse
import base64
import hashlib
import json
import os

DEFAULT_ENC = os.path.join(os.path.dirname(__file__), "data", "georef_poses.enc")
KEY_CONTEXT = b"zf-frgen21-georef-v1"


def sequence_key(seq_dir):
    """Key material for one sequence: SHA-256 of its ground_truth.csv."""
    h = hashlib.sha256(KEY_CONTEXT)
    with open(os.path.join(seq_dir, "ground_truth.csv"), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.digest()


def _keystream_xor(key, data):
    out = bytearray(len(data))
    for block in range(0, len(data), 32):
        pad = hashlib.sha256(key + block.to_bytes(8, "big")).digest()
        for i, b in enumerate(data[block:block + 32]):
            out[block + i] = b ^ pad[i]
    return bytes(out)


def _check(key):
    return hashlib.sha256(key + b"check").hexdigest()[:16]


def encode(entries, keys, meta=None):
    """{seq: dict} + {seq: key} -> obfuscated container (dict, JSON-serializable)."""
    return {
        "format": "zf-frgen21-georef",
        "version": 1,
        "meta": meta or {},
        "sequences": {
            seq: {
                "check": _check(keys[seq]),
                "data": base64.b64encode(_keystream_xor(keys[seq], json.dumps(entry).encode())).decode(),
            }
            for seq, entry in entries.items()
        },
    }


def decode(container, dataset_dir, sequences=None):
    """Decode the entries for every sequence present under ``dataset_dir``.

    Returns ``{"meta": ..., "sequences": {seq: entry}}``. Sequences whose
    ground_truth.csv is missing are skipped; a mismatching file raises.
    """
    out = {"meta": container.get("meta", {}), "sequences": {}}
    for seq, blob in container["sequences"].items():
        if sequences and seq not in sequences:
            continue
        seq_dir = os.path.join(dataset_dir, seq)
        if not os.path.exists(os.path.join(seq_dir, "ground_truth.csv")):
            continue
        key = sequence_key(seq_dir)
        if _check(key) != blob["check"]:
            raise ValueError(
                f"{seq}: ground_truth.csv does not match the dataset version the poses were made for "
                "(Zenodo v1.0.0)"
            )
        out["sequences"][seq] = json.loads(_keystream_xor(key, base64.b64decode(blob["data"])))
    return out


def load_georef(path):
    """Load a decoded georef JSON (as written by ``zf-radar-georef``)."""
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Decode the reconstructed start poses using the dataset as key")
    parser.add_argument("-p", "--path", required=True, help="Dataset root containing sequence directories.")
    parser.add_argument("-o", "--out", default="georef_poses.json", help="Output JSON (default: georef_poses.json)")
    parser.add_argument("--enc", default=DEFAULT_ENC, help="Obfuscated pose file (default: bundled)")
    args = parser.parse_args()
    if not os.path.isdir(args.path):
        parser.error(f"dataset path is not a directory: {args.path}")
    if not os.path.exists(args.enc):
        raise SystemExit(f"pose file not found: {args.enc} (none is bundled with this version)")

    with open(args.enc) as f:
        decoded = decode(json.load(f), args.path)
    if not decoded["sequences"]:
        raise SystemExit(f"no sequences with ground_truth.csv found under {args.path}")
    with open(args.out, "w") as f:
        json.dump(decoded, f, indent=2)
    print(f"decoded {', '.join(decoded['sequences'])} -> {args.out}")


if __name__ == "__main__":
    main()
