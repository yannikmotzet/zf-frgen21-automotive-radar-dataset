#!/usr/bin/env python3
"""Export ZF FRGen21 sequences to ROS 2 MCAP files (one file per sequence).

Topics (all stamped with the radar frame's ``timestamp_ns``):
    /radar/points               sensor_msgs/msg/PointCloud2   all PCD fields as float32, frame "radar"
    /radar/velocity_ambiguity   std_msgs/msg/Float32          unambiguous Doppler interval [m/s]
    /ground_truth/odom          nav_msgs/msg/Odometry         odom_ins -> base_link, twist in base_link
    /ground_truth/path          nav_msgs/msg/Path             full trajectory, once at start
    /tf                         tf2_msgs/msg/TFMessage        odom_ins -> base_link per frame
    /tf_static                  tf2_msgs/msg/TFMessage        base_link -> radar (extrinsics), once

With ``--georef poses.json`` (start pose per sequence, see zf_radar.georef) also:
    /tf_static                  map -> odom_ins               sequence start placed in the shared map frame
    /gnss/fix                   sensor_msgs/msg/NavSatFix     WGS 84 position of base_link per frame

Only needs ``mcap`` and ``mcap-ros2-support`` — no ROS installation (``pyproj``
for ``--georef``). The files open in Foxglove and play back with
``ros2 bag play -s mcap <file>``.
"""

import argparse
import glob
import os

import numpy as np

try:
    from mcap.writer import CompressionType, Writer
    from mcap_ros2._dynamic import serialize_dynamic
except ImportError as e:
    raise SystemExit(
        f"Missing MCAP dependency: {e.name}. The exporter needs the [mcap] extras:\n"
        '  pip install "zf-frgen21-radar-dataset[mcap]"'
    ) from e

from zf_radar.georef import load_georef
from zf_radar.loaders import discover_sequences, load_extrinsics, load_ground_truth, load_pcd

MAP_FRAME = "map"
# REP-105 odometry frame whose source is the INS ground truth; it is the dataset's
# "world" frame (vehicle pose at the first frame). Own estimates (e.g. radar
# odometry) can live next to it as odom_radar -> base_link_radar.
ODOM_FRAME = "odom_ins"
VEHICLE_FRAME = "base_link"
SENSOR_FRAME = "radar"

_SEP = "=" * 80

# ROS 2 message definitions (Humble), comments stripped, dependencies inlined
# in the concatenated format rosbag2 writes into MCAP schemas
_TIME = "MSG: builtin_interfaces/Time\nint32 sec\nuint32 nanosec"
_HEADER = "MSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id"
_VECTOR3 = "MSG: geometry_msgs/Vector3\nfloat64 x\nfloat64 y\nfloat64 z"
_POINT = "MSG: geometry_msgs/Point\nfloat64 x\nfloat64 y\nfloat64 z"
_QUAT = "MSG: geometry_msgs/Quaternion\nfloat64 x 0\nfloat64 y 0\nfloat64 z 0\nfloat64 w 1"
_POSE = "MSG: geometry_msgs/Pose\nPoint position\nQuaternion orientation"
_POSE_STAMPED = "MSG: geometry_msgs/PoseStamped\nstd_msgs/Header header\nPose pose"
_TRANSFORM = "MSG: geometry_msgs/Transform\nVector3 translation\nQuaternion rotation"
_TRANSFORM_STAMPED = (
    "MSG: geometry_msgs/TransformStamped\nstd_msgs/Header header\nstring child_frame_id\nTransform transform"
)


def _msgdef(main, *deps):
    return "\n".join([main] + [f"{_SEP}\n{d}" for d in deps])


MSGDEFS = {
    "sensor_msgs/msg/PointCloud2": _msgdef(
        "std_msgs/Header header\nuint32 height\nuint32 width\nPointField[] fields\nbool is_bigendian\n"
        "uint32 point_step\nuint32 row_step\nuint8[] data\nbool is_dense",
        _HEADER,
        _TIME,
        "MSG: sensor_msgs/PointField\nuint8 INT8=1\nuint8 UINT8=2\nuint8 INT16=3\nuint8 UINT16=4\n"
        "uint8 INT32=5\nuint8 UINT32=6\nuint8 FLOAT32=7\nuint8 FLOAT64=8\n"
        "string name\nuint32 offset\nuint8 datatype\nuint32 count",
    ),
    "std_msgs/msg/Float32": "float32 data",
    "tf2_msgs/msg/TFMessage": _msgdef(
        "geometry_msgs/TransformStamped[] transforms",
        _TRANSFORM_STAMPED, _HEADER, _TIME, _TRANSFORM, _VECTOR3, _QUAT,
    ),
    "nav_msgs/msg/Odometry": _msgdef(
        "std_msgs/Header header\nstring child_frame_id\ngeometry_msgs/PoseWithCovariance pose\n"
        "geometry_msgs/TwistWithCovariance twist",
        _HEADER,
        _TIME,
        "MSG: geometry_msgs/PoseWithCovariance\nPose pose\nfloat64[36] covariance",
        _POSE,
        _POINT,
        _QUAT,
        "MSG: geometry_msgs/TwistWithCovariance\nTwist twist\nfloat64[36] covariance",
        "MSG: geometry_msgs/Twist\nVector3 linear\nVector3 angular",
        _VECTOR3,
    ),
    "sensor_msgs/msg/NavSatFix": _msgdef(
        "std_msgs/Header header\nNavSatStatus status\nfloat64 latitude\nfloat64 longitude\nfloat64 altitude\n"
        "float64[9] position_covariance\nuint8 COVARIANCE_TYPE_UNKNOWN=0\nuint8 COVARIANCE_TYPE_APPROXIMATED=1\n"
        "uint8 COVARIANCE_TYPE_DIAGONAL_KNOWN=2\nuint8 COVARIANCE_TYPE_KNOWN=3\nuint8 position_covariance_type",
        _HEADER,
        _TIME,
        "MSG: sensor_msgs/NavSatStatus\nint8 STATUS_NO_FIX=-1\nint8 STATUS_FIX=0\nint8 STATUS_SBAS_FIX=1\n"
        "int8 STATUS_GBAS_FIX=2\nint8 status\nuint16 SERVICE_GPS=1\nuint16 SERVICE_GLONASS=2\n"
        "uint16 SERVICE_COMPASS=4\nuint16 SERVICE_GALILEO=8\nuint16 service",
    ),
    "nav_msgs/msg/Path": _msgdef(
        "std_msgs/Header header\ngeometry_msgs/PoseStamped[] poses",
        _HEADER, _TIME, _POSE_STAMPED, _POSE, _POINT, _QUAT,
    ),
}

# rosbag2 reads the offered QoS from channel metadata; latched topics need
# transient_local durability (1) so late subscribers still get them on playback
_QOS_TEMPLATE = (
    "- history: 3\n  depth: 0\n  reliability: 1\n  durability: {durability}\n"
    "  deadline:\n    sec: 2147483647\n    nsec: 4294967295\n"
    "  lifespan:\n    sec: 2147483647\n    nsec: 4294967295\n"
    "  liveliness: 1\n"
    "  liveliness_lease_duration:\n    sec: 2147483647\n    nsec: 4294967295\n"
    "  avoid_ros_namespace_conventions: false"
)
QOS_VOLATILE = _QOS_TEMPLATE.format(durability=2)
QOS_LATCHED = _QOS_TEMPLATE.format(durability=1)

TOPICS = {
    "/radar/points": ("sensor_msgs/msg/PointCloud2", QOS_VOLATILE),
    "/radar/velocity_ambiguity": ("std_msgs/msg/Float32", QOS_VOLATILE),
    "/ground_truth/odom": ("nav_msgs/msg/Odometry", QOS_VOLATILE),
    "/ground_truth/path": ("nav_msgs/msg/Path", QOS_LATCHED),
    "/tf": ("tf2_msgs/msg/TFMessage", QOS_VOLATILE),
    "/tf_static": ("tf2_msgs/msg/TFMessage", QOS_LATCHED),
}
GEO_TOPICS = {
    "/gnss/fix": ("sensor_msgs/msg/NavSatFix", QOS_VOLATILE),
}

POINTFIELD_FLOAT32 = 7
NAVSAT_COVARIANCE_APPROXIMATED = 1
DEFAULT_GEO_CRS = "EPSG:32632"


def rotation_to_quaternion(R):
    """3x3 rotation matrix -> (x, y, z, w) unit quaternion (Shepperd's method)."""
    trace = np.trace(R)
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def _stamp(ns):
    return {"sec": ns // 1_000_000_000, "nanosec": ns % 1_000_000_000}


def _xyz(v):
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _pose(T):
    x, y, z, w = rotation_to_quaternion(T[:3, :3])
    return {"position": _xyz(T[:3, 3]), "orientation": {"x": x, "y": y, "z": z, "w": w}}


def _transform(ns, parent, child, T):
    pose = _pose(T)
    return {
        "header": {"stamp": _stamp(ns), "frame_id": parent},
        "child_frame_id": child,
        "transform": {"translation": pose["position"], "rotation": pose["orientation"]},
    }


def pointcloud_msg(ns, pcd, fields):
    """PointCloud2 with every PCD field packed as little-endian float32."""
    n = len(pcd[fields[0]])
    packed = np.empty((n, len(fields)), dtype="<f4")
    for i, name in enumerate(fields):
        packed[:, i] = pcd[name]
    point_step = 4 * len(fields)
    return {
        "header": {"stamp": _stamp(ns), "frame_id": SENSOR_FRAME},
        "height": 1,
        "width": n,
        "fields": [
            {"name": name, "offset": 4 * i, "datatype": POINTFIELD_FLOAT32, "count": 1}
            for i, name in enumerate(fields)
        ],
        "is_bigendian": False,
        "point_step": point_step,
        "row_step": point_step * n,
        "data": packed.tobytes(),
        "is_dense": True,
    }


class McapExporter:
    """Thin wrapper over the MCAP writer: one channel per topic, CDR-encoded."""

    def __init__(self, path, topics=TOPICS):
        self._file = open(path, "wb")
        self._writer = Writer(self._file, compression=CompressionType.ZSTD)
        self._writer.start(profile="ros2", library="zf_radar.export_mcap")
        self._channels = {}
        self._encoders = {}
        for topic, (datatype, qos) in topics.items():
            schema_id = self._writer.register_schema(datatype, "ros2msg", MSGDEFS[datatype].encode())
            self._channels[topic] = self._writer.register_channel(
                topic, "cdr", schema_id, metadata={"offered_qos_profiles": qos}
            )
            self._encoders[topic] = serialize_dynamic(datatype, MSGDEFS[datatype])[datatype]

    def write(self, topic, ns, msg):
        self._writer.add_message(
            self._channels[topic], log_time=ns, publish_time=ns, data=self._encoders[topic](msg)
        )

    def close(self):
        self._writer.finish()
        self._file.close()


def geo_positions(T_utm_world, xyz, crs=DEFAULT_GEO_CRS):
    """World-frame points -> (lat, lon, height) via the start pose ``T_utm_world``."""
    try:
        from pyproj import Transformer
    except ImportError as e:
        raise SystemExit(
            "Missing dependency for --georef: pyproj. Install the [geo] extras:\n"
            '  pip install "zf-frgen21-radar-dataset[geo]"'
        ) from e
    projected = xyz @ T_utm_world[:3, :3].T + T_utm_world[:3, 3]
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(projected[:, 0], projected[:, 1])
    return np.asarray(lat), np.asarray(lon), projected[:, 2]


def navsatfix_msg(ns, lat, lon, alt, sigma_h, sigma_v):
    return {
        "header": {"stamp": _stamp(ns), "frame_id": VEHICLE_FRAME},
        "status": {"status": 0, "service": 1},
        "latitude": float(lat),
        "longitude": float(lon),
        "altitude": float(alt),
        "position_covariance": [sigma_h ** 2, 0.0, 0.0, 0.0, sigma_h ** 2, 0.0, 0.0, 0.0, sigma_v ** 2],
        "position_covariance_type": NAVSAT_COVARIANCE_APPROXIMATED,
    }


def export_sequence(seq_dir, out_path, limit=None, progress=True, georef=None, accuracy=(2.0, 2.0)):
    """Write one sequence to ``out_path``. Returns the number of radar frames written.

    ``georef`` is one sequence entry of a decoded georef file (``T_utm_world``,
    ``map_origin``, optional ``crs``); it adds ``map -> odom_ins`` and ``/gnss/fix``.
    ``accuracy`` is the (horizontal, vertical) 1-sigma in metres for the fix covariance.
    """
    files = sorted(glob.glob(os.path.join(seq_dir, "radar", "*.pcd")))
    gt = load_ground_truth(seq_dir)
    if gt is not None and len(gt["t"]) != len(files):
        raise ValueError(f"{seq_dir}: {len(files)} PCD files but {len(gt['t'])} ground-truth rows")
    T_sensor_vehicle = load_extrinsics(seq_dir)
    if limit is not None:
        files = files[:limit]

    topics = dict(TOPICS)
    if georef is not None:
        if gt is None:
            raise ValueError(f"{seq_dir}: --georef needs ground_truth.csv")
        T_utm_world = np.array(georef["T_utm_world"], dtype=float)
        T_map_odom = T_utm_world.copy()
        T_map_odom[:3, 3] -= georef["map_origin"]
        lat, lon, alt = geo_positions(T_utm_world, gt["pose"][:, :3, 3], georef.get("crs", DEFAULT_GEO_CRS))
        topics.update(GEO_TOPICS)

    exporter = McapExporter(out_path, topics)
    try:
        for i, path in enumerate(files):
            pcd = load_pcd(path)
            meta = pcd.pop("_meta")
            ns = meta["timestamp_ns"]
            fields = list(pcd)

            if i == 0:
                static = [_transform(ns, VEHICLE_FRAME, SENSOR_FRAME, T_sensor_vehicle)]
                if georef is not None:
                    static.append(_transform(ns, MAP_FRAME, ODOM_FRAME, T_map_odom))
                exporter.write("/tf_static", ns, {"transforms": static})
                if gt is not None:
                    # Ground-truth t is relative to the first radar frame by construction
                    stamps = ns + np.round(gt["t"][:len(files)] * 1e9).astype(np.int64)
                    exporter.write("/ground_truth/path", ns, {
                        "header": {"stamp": _stamp(ns), "frame_id": ODOM_FRAME},
                        "poses": [
                            {"header": {"stamp": _stamp(int(s)), "frame_id": ODOM_FRAME}, "pose": _pose(T)}
                            for s, T in zip(stamps, gt["pose"])
                        ],
                    })

            exporter.write("/radar/points", ns, pointcloud_msg(ns, pcd, fields))
            exporter.write("/radar/velocity_ambiguity", ns, {"data": float(meta["velocity_ambiguity"])})

            if gt is not None:
                T = gt["pose"][i]
                exporter.write("/tf", ns, {"transforms": [_transform(ns, ODOM_FRAME, VEHICLE_FRAME, T)]})
                exporter.write("/ground_truth/odom", ns, {
                    "header": {"stamp": _stamp(ns), "frame_id": ODOM_FRAME},
                    "child_frame_id": VEHICLE_FRAME,
                    "pose": {"pose": _pose(T), "covariance": [0.0] * 36},
                    "twist": {
                        "twist": {"linear": _xyz(gt["v_body"][i]), "angular": _xyz(gt["w_body"][i])},
                        "covariance": [0.0] * 36,
                    },
                })
                if georef is not None:
                    exporter.write("/gnss/fix", ns, navsatfix_msg(ns, lat[i], lon[i], alt[i], *accuracy))

            if progress and (i % 500 == 0 or i == len(files) - 1):
                print(f"\r  {i + 1}/{len(files)} frames", end="", flush=True)
    finally:
        exporter.close()
    if progress:
        print()
    return len(files)


def main():
    parser = argparse.ArgumentParser(description="Export ZF radar sequences to ROS 2 MCAP")
    parser.add_argument("-p", "--path", required=True, help="Dataset root containing sequence directories.")
    parser.add_argument("-o", "--out", default="mcap", help="Output directory (default: ./mcap)")
    parser.add_argument("--seq", nargs="+", help="Only these sequences (default: all)")
    parser.add_argument("--limit", type=int, help="Export only the first N frames of each sequence")
    parser.add_argument("--georef", help="Decoded start-pose JSON: adds map -> odom_ins and /gnss/fix")
    args = parser.parse_args()
    if not os.path.isdir(args.path):
        parser.error(f"dataset path is not a directory: {args.path}")

    georef, accuracy = {}, (2.0, 2.0)
    if args.georef:
        decoded = load_georef(args.georef)
        georef = decoded["sequences"]
        acc = decoded.get("meta", {}).get("accuracy_m", {})
        accuracy = (acc.get("horizontal", 2.0), acc.get("vertical", 2.0))

    os.makedirs(args.out, exist_ok=True)
    for name in args.seq or discover_sequences(args.path):
        out_path = os.path.join(args.out, f"{name}.mcap")
        print(f"{name} -> {out_path}")
        if args.georef and name not in georef:
            print(f"  warning: {name} has no entry in {args.georef}, exporting without map frame / fix")
        export_sequence(os.path.join(args.path, name), out_path, limit=args.limit,
                        georef=georef.get(name), accuracy=accuracy)


if __name__ == "__main__":
    main()
