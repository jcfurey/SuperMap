#!/usr/bin/env python3
"""Convert a rosbag2 recording into the offline sequence layout.

    python examples/rosbag_to_sequence.py <bag_dir> --out_dir data/my_capture \\
        --rgb_topic /camera/color/image_raw --camera_info_topic /camera/color/camera_info \\
        --pointcloud_topic /lidar/points --odometry_topic /odometry \\
        --world_frame map --camera_frame camera_color_optical_frame

    # RGB-D camera with depth already aligned to the color image:
    python examples/rosbag_to_sequence.py <bag_dir> --out_dir data/my_capture \\
        --depth_topic /camera/aligned_depth_to_color/image_raw --depth_scale 1000

    # Also run a detector during conversion so example.py / evaluate.py can replay it offline:
    python examples/rosbag_to_sequence.py <bag_dir> --out_dir data/my_capture --detector yoloe

Any "good" odometry works: the camera pose P_t (Eq. 3) is resolved through
the bag's TF tree (``/tf``, ``/tf_static``, plus the odometry topic, which
is injected as a ``header.frame_id -> child_frame_id`` transform exactly as
a SLAM backbone would broadcast it), so the world-to-camera chain composes
the dynamic pose with whatever static extrinsic the recording carries.
Depth comes either from a point cloud rasterized into the camera (like the
live node) or from an aligned depth image. RGB frames are paired with the
nearest depth source within ``--sync_slop`` seconds.

The output is what :class:`semantic_mapping.datasets.SequenceDataset` reads
(``intrinsics.json``, ``frames/``, optional ``detections/``), so
``examples/example.py``, ``evaluate.py``, ``benchmark.py``, and ``query.py``
run on it unchanged. Requires a sourced ROS 2 environment (rosbag2_py, tf2_ros).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_mapping.geometry_utils import rasterize_depth, rotation_matrix_to_quaternion, transform_points  # noqa: E402
from semantic_mapping.ros_msgs import (  # noqa: E402
    camera_info_to_intrinsics, depth_image_to_meters, image_to_numpy, pointcloud_to_xyz, pose_to_se3,
    stamp_to_seconds, transform_to_se3,
)


def _stamp_of(msg, bag_time_ns: int) -> float:
    """Header stamp in seconds; a zero header stamp falls back to the bag's receive time."""
    stamp = msg.header.stamp
    if stamp.sec == 0 and stamp.nanosec == 0:
        return bag_time_ns * 1e-9
    return stamp_to_seconds(stamp)


class BagReader:
    """Thin wrapper over rosbag2_py's sequential reader with typed deserialization."""

    def __init__(self, bag_dir: str | Path) -> None:
        import rosbag2_py

        self._rosbag2_py = rosbag2_py
        self.bag_dir = str(bag_dir)
        reader = self._open()
        self.topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
        metadata = reader.get_metadata()
        self.duration_s = metadata.duration.nanoseconds * 1e-9
        self.start_s = metadata.starting_time.nanoseconds * 1e-9
        del reader

    def _open(self):
        reader = self._rosbag2_py.SequentialReader()
        reader.open(self._rosbag2_py.StorageOptions(uri=self.bag_dir),
                    self._rosbag2_py.ConverterOptions("cdr", "cdr"))
        return reader

    def messages(self, topics: list[str]):
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message

        topics = [t for t in topics if t in self.topic_types]
        if not topics:
            return
        classes = {t: get_message(self.topic_types[t]) for t in topics}
        reader = self._open()
        reader.set_filter(self._rosbag2_py.StorageFilter(topics=topics))
        while reader.has_next():
            topic, data, t_ns = reader.read_next()
            yield topic, deserialize_message(data, classes[topic]), t_ns


def build_tf_buffer(reader: BagReader, odometry_topic: str | None, tf_topic: str, tf_static_topic: str):
    """Replay the bag's transforms into a tf2 BufferCore covering the whole recording."""
    from geometry_msgs.msg import TransformStamped
    from rclpy.duration import Duration
    from tf2_ros import BufferCore

    buffer = BufferCore(Duration(seconds=reader.duration_s + 60.0))
    counts: Counter = Counter()
    topics = [tf_topic, tf_static_topic] + ([odometry_topic] if odometry_topic else [])
    for topic, msg, _t_ns in reader.messages(topics):
        if topic == tf_static_topic:
            for transform in msg.transforms:
                buffer.set_transform_static(transform, "bag")
                counts["static"] += 1
        elif topic == tf_topic:
            for transform in msg.transforms:
                buffer.set_transform(transform, "bag")
                counts["dynamic"] += 1
        else:  # odometry -> header.frame_id -> child_frame_id, as a SLAM backbone broadcasts it
            transform = TransformStamped()
            transform.header = msg.header
            transform.child_frame_id = msg.child_frame_id
            transform.transform.translation.x = msg.pose.pose.position.x
            transform.transform.translation.y = msg.pose.pose.position.y
            transform.transform.translation.z = msg.pose.pose.position.z
            transform.transform.rotation = msg.pose.pose.orientation
            buffer.set_transform(transform, "bag")
            counts["odometry"] += 1
    return buffer, counts


def lookup_se3(buffer, target_frame: str, source_frame: str, stamp_s: float) -> np.ndarray:
    from rclpy.time import Time

    return transform_to_se3(buffer.lookup_transform_core(target_frame, source_frame, Time(seconds=stamp_s)))


def pair_nearest(rgb_stamps: list[float], depth_stamps: list[float], slop: float) -> dict[float, float]:
    """RGB stamp -> nearest depth-source stamp within ``slop`` seconds."""
    if not depth_stamps:
        return {}
    depth = np.array(sorted(depth_stamps))
    pairs: dict[float, float] = {}
    for stamp in rgb_stamps:
        i = int(np.searchsorted(depth, stamp))
        candidates = [depth[j] for j in (i - 1, i) if 0 <= j < depth.size]
        best = min(candidates, key=lambda d: abs(d - stamp))
        if abs(best - stamp) <= slop:
            pairs[stamp] = float(best)
    return pairs


def write_detections(detections_dir: Path, frame_id: int, detections) -> None:
    detections_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i, det in enumerate(detections):
        record = {"bbox": [float(v) for v in det.bbox], "label": det.label, "score": float(det.score)}
        if det.mask is not None:
            mask_name = f"{frame_id:06d}_{i}.npy"
            np.save(detections_dir / mask_name, det.mask.astype(bool))
            record["mask"] = mask_name
        records.append(record)
    (detections_dir / f"{frame_id:06d}.json").write_text(json.dumps({"detections": records}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bag", help="rosbag2 directory (sqlite3 or mcap).")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--rgb_topic", default="/camera/color/image_raw")
    parser.add_argument("--camera_info_topic", default="/camera/color/camera_info")
    parser.add_argument("--pointcloud_topic", default="/lidar/points",
                        help="PointCloud2 rasterized into the camera for depth (ignored when --depth_topic is set).")
    parser.add_argument("--depth_topic", default=None, help="Depth image aligned to the RGB camera.")
    parser.add_argument("--depth_scale", type=float, default=1000.0, help="Units per meter for 16-bit depth images.")
    parser.add_argument("--odometry_topic", default="/odometry",
                        help="nav_msgs/Odometry injected into TF as header.frame_id -> child_frame_id ('' to skip).")
    parser.add_argument("--tf_topic", default="/tf")
    parser.add_argument("--tf_static_topic", default="/tf_static")
    parser.add_argument("--world_frame", default="map")
    parser.add_argument("--camera_frame", default="camera_color_optical_frame")
    parser.add_argument("--sync_slop", type=float, default=0.05, help="Max RGB / depth-source stamp gap in seconds.")
    parser.add_argument("--frame_skip", type=int, default=1, help="Keep every N-th RGB frame.")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--start", type=float, default=None, help="Skip frames before this bag-relative time (s).")
    parser.add_argument("--end", type=float, default=None, help="Skip frames after this bag-relative time (s).")
    parser.add_argument("--detector", choices=["none", "yoloe", "groundingdino"], default="none",
                        help="Run a detector per frame and store its detections (+ masks) for offline replay.")
    parser.add_argument("--config", default="config/semantic_mapping.yaml", help="Detector settings (per backend).")
    parser.add_argument("--prompts", default="config/prompts.yaml")
    args = parser.parse_args()

    reader = BagReader(args.bag)
    depth_topic = args.depth_topic or args.pointcloud_topic
    for topic in (args.rgb_topic, args.camera_info_topic, depth_topic):
        if topic not in reader.topic_types:
            available = "\n  ".join(f"{k}: {v}" for k, v in sorted(reader.topic_types.items()))
            raise SystemExit(f"topic {topic!r} not in bag. Available topics:\n  {available}")
    odometry_topic = args.odometry_topic or None
    if odometry_topic and odometry_topic not in reader.topic_types:
        print(f"note: odometry topic {odometry_topic!r} not in bag; relying on /tf alone")
        odometry_topic = None

    buffer, tf_counts = build_tf_buffer(reader, odometry_topic, args.tf_topic, args.tf_static_topic)
    print(f"TF: {tf_counts['static']} static, {tf_counts['dynamic']} dynamic, {tf_counts['odometry']} odometry transforms")

    # Pass 1: stamps only, to decide which RGB frames to keep and which depth source each gets.
    rgb_stamps, depth_stamps = [], []
    for topic, msg, t_ns in reader.messages([args.rgb_topic, depth_topic]):
        (rgb_stamps if topic == args.rgb_topic else depth_stamps).append(_stamp_of(msg, t_ns))
    t0 = rgb_stamps[0] if rgb_stamps else reader.start_s
    selected = [s for s in rgb_stamps
                if (args.start is None or s - t0 >= args.start) and (args.end is None or s - t0 <= args.end)]
    selected = selected[:: max(args.frame_skip, 1)]
    if args.max_frames is not None:
        selected = selected[: args.max_frames]
    pairs = pair_nearest(selected, depth_stamps, args.sync_slop)
    needed_depth = set(pairs.values())
    print(f"{len(rgb_stamps)} RGB frames in bag, {len(selected)} selected, {len(pairs)} paired with a depth source")

    detector = None
    prompts = None
    if args.detector != "none":
        from semantic_mapping.datasets import load_prompts, load_yaml_params
        from semantic_mapping.detectors import build_detector

        params = load_yaml_params(args.config)
        detector = build_detector(args.detector, **params.get(args.detector, {}))
        prompts = load_prompts(args.prompts)

    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    detections_dir = out_dir / "detections"

    # Pass 2: decode and write. Depth sources are held only until their RGB partner is written.
    intrinsics = None
    pending_rgb: dict[float, np.ndarray] = {}
    depth_cache: dict[float, tuple[np.ndarray, str, float]] = {}
    frame_id = 0
    skipped: Counter = Counter()

    def emit(rgb_stamp: float, rgb: np.ndarray) -> None:
        nonlocal frame_id, intrinsics
        depth_source, source_frame, source_stamp = depth_cache.pop(pairs[rgb_stamp])
        if intrinsics is None:
            skipped["no_camera_info_yet"] += 1
            return
        try:
            T_world_from_cam = lookup_se3(buffer, args.world_frame, args.camera_frame, rgb_stamp)
            if args.depth_topic:
                depth = depth_source
            else:
                T_cam_from_cloud = lookup_se3(buffer, args.camera_frame, source_frame, source_stamp)
                points_cam = transform_points(T_cam_from_cloud, depth_source)
                depth = rasterize_depth(points_cam, intrinsics.K, intrinsics.width, intrinsics.height)
        except Exception as exc:  # tf2 lookup / extrapolation errors
            skipped["no_transform"] += 1
            if skipped["no_transform"] <= 3:
                print(f"skipping frame at t={rgb_stamp:.3f}: {exc}")
            return
        if rgb.shape[:2] != (intrinsics.height, intrinsics.width):
            import cv2

            rgb = cv2.resize(rgb, (intrinsics.width, intrinsics.height), interpolation=cv2.INTER_AREA)
        if depth.shape != (intrinsics.height, intrinsics.width):
            skipped["depth_size_mismatch"] += 1
            return

        prefix = frames_dir / f"{frame_id:06d}"
        np.save(f"{prefix}_rgb.npy", np.ascontiguousarray(rgb))
        np.save(f"{prefix}_depth.npy", depth.astype(np.float32))
        pose = {"stamp": rgb_stamp, "translation": T_world_from_cam[:3, 3].tolist(),
                "quaternion": rotation_matrix_to_quaternion(T_world_from_cam[:3, :3]).tolist()}
        (frames_dir / f"{frame_id:06d}_pose.json").write_text(json.dumps(pose))
        if detector is not None:
            write_detections(detections_dir, frame_id, detector.detect(rgb, prompts=prompts))
        frame_id += 1

    for topic, msg, t_ns in reader.messages([args.rgb_topic, args.camera_info_topic, depth_topic]):
        stamp = _stamp_of(msg, t_ns)
        if topic == args.camera_info_topic:
            if intrinsics is None:
                intrinsics = camera_info_to_intrinsics(msg)
                (out_dir / "intrinsics.json").write_text(json.dumps(intrinsics.__dict__, indent=2))
            continue
        if topic == depth_topic:
            if stamp in needed_depth and stamp not in depth_cache:
                source = depth_image_to_meters(msg, args.depth_scale) if args.depth_topic else pointcloud_to_xyz(msg)
                depth_cache[stamp] = (source, msg.header.frame_id, stamp)
        else:
            if stamp not in pairs:
                if stamp in selected:
                    skipped["no_depth_within_slop"] += 1
                continue
            pending_rgb[stamp] = image_to_numpy(msg)
        for rgb_stamp in [s for s in pending_rgb if pairs[s] in depth_cache]:
            emit(rgb_stamp, pending_rgb.pop(rgb_stamp))

    skipped["depth_source_never_arrived"] += len(pending_rgb)
    info = {
        "bag": str(Path(args.bag).resolve()), "frames": frame_id, "skipped": dict(skipped),
        "topics": {"rgb": args.rgb_topic, "camera_info": args.camera_info_topic, "depth": depth_topic,
                   "odometry": odometry_topic}, "frames_in_bag": len(rgb_stamps),
        "world_frame": args.world_frame, "camera_frame": args.camera_frame, "detector": args.detector,
    }
    (out_dir / "sequence_info.json").write_text(json.dumps(info, indent=2))
    print(f"Wrote {frame_id} frames to {out_dir}/" + (f" (skipped: {dict(skipped)})" if skipped else ""))
    if frame_id == 0:
        raise SystemExit("no frames written; check the topic names, frames, and --sync_slop")


if __name__ == "__main__":
    main()
