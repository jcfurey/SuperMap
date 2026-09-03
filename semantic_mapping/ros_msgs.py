"""Conversions between ROS 2 messages and the pipeline's numpy types.

Shared by the live node and the rosbag converter so both decode sensor data
identically. Only numpy is needed at import time; ROS message modules are
imported inside the functions that build messages, which keeps this module
importable in the offline (non-ROS) environment.
"""
from __future__ import annotations

import numpy as np

from semantic_mapping.geometry_utils import se3_from_translation_quaternion
from semantic_mapping.types import CameraIntrinsics

_IMAGE_ENCODINGS: dict[str, tuple[type, int]] = {
    "rgb8": (np.uint8, 3), "bgr8": (np.uint8, 3), "rgba8": (np.uint8, 4), "bgra8": (np.uint8, 4),
    "mono8": (np.uint8, 1), "8UC1": (np.uint8, 1), "8UC3": (np.uint8, 3),
    "bayer_rggb8": (np.uint8, 1), "bayer_bggr8": (np.uint8, 1),
    "bayer_gbrg8": (np.uint8, 1), "bayer_grbg8": (np.uint8, 1),
    "mono16": (np.uint16, 1), "16UC1": (np.uint16, 1), "32FC1": (np.float32, 1),
}

_BAYER_ENCODINGS = {
    # These OpenCV conversion constants yield RGB channel order for the ROS
    # top-left-pixel pattern named by each sensor_msgs encoding.
    "bayer_rggb8": "COLOR_BayerRG2BGR",
    "bayer_bggr8": "COLOR_BayerBG2BGR",
    "bayer_gbrg8": "COLOR_BayerGB2BGR",
    "bayer_grbg8": "COLOR_BayerGR2BGR",
}


def stamp_to_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def camera_info_to_intrinsics(info) -> CameraIntrinsics:
    """Pinhole intrinsics from a sensor_msgs/CameraInfo (its ``k`` matrix)."""
    return CameraIntrinsics(
        fx=float(info.k[0]), fy=float(info.k[4]), cx=float(info.k[2]), cy=float(info.k[5]),
        width=int(info.width), height=int(info.height),
    )


def transform_to_se3(transform_stamped) -> np.ndarray:
    """4x4 matrix mapping child-frame points into the header frame (TF2 semantics)."""
    t, q = transform_stamped.transform.translation, transform_stamped.transform.rotation
    return se3_from_translation_quaternion(np.array([t.x, t.y, t.z]), np.array([q.x, q.y, q.z, q.w]))


def pose_to_se3(pose) -> np.ndarray:
    """4x4 matrix from a geometry_msgs/Pose."""
    p, q = pose.position, pose.orientation
    return se3_from_translation_quaternion(np.array([p.x, p.y, p.z]), np.array([q.x, q.y, q.z, q.w]))


def pointcloud_to_xyz(msg) -> np.ndarray:
    """(N, 3) float64 xyz from a sensor_msgs/PointCloud2, NaNs dropped.

    Stays vectorized: a LiDAR scan carries tens of thousands of points and
    this runs on every synchronized frame.
    """
    from sensor_msgs_py import point_cloud2 as pc2

    cloud = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    if cloud.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return np.stack([cloud["x"], cloud["y"], cloud["z"]], axis=-1).astype(np.float64)


def image_to_numpy(msg) -> np.ndarray:
    """Decode a sensor_msgs/Image or CompressedImage into an (H, W[, C]) array.

    Color images come back as RGB regardless of the wire encoding; depth
    encodings (16UC1 / mono16 / 32FC1) come back unscaled. Row padding
    (``step`` larger than the row payload) and big-endian data are handled.
    """
    if type(msg).__name__ == "CompressedImage":
        import cv2

        decoded = cv2.imdecode(np.frombuffer(bytes(msg.data), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ValueError(f"could not decode CompressedImage with format {msg.format!r}")
        if decoded.ndim == 3 and decoded.shape[2] >= 3:
            decoded = cv2.cvtColor(decoded[:, :, :3], cv2.COLOR_BGR2RGB)  # ROS compresses color as BGR
        return decoded

    try:
        dtype, channels = _IMAGE_ENCODINGS[msg.encoding]
    except KeyError as exc:
        raise ValueError(f"unsupported image encoding {msg.encoding!r}") from exc
    dtype = np.dtype(dtype).newbyteorder(">" if msg.is_bigendian else "<")
    row_values = msg.step // dtype.itemsize
    data = np.frombuffer(bytes(msg.data), dtype=dtype).reshape(msg.height, row_values)[:, : msg.width * channels]
    image = data.reshape(msg.height, msg.width, channels) if channels > 1 else data.reshape(msg.height, msg.width)
    image = image.astype(dtype.newbyteorder("="))
    if msg.encoding in _BAYER_ENCODINGS:
        import cv2

        image = cv2.cvtColor(image, getattr(cv2, _BAYER_ENCODINGS[msg.encoding]))
    elif msg.encoding in ("bgr8", "bgra8"):
        image = image[:, :, :3][:, :, ::-1]
    elif msg.encoding == "rgba8":
        image = image[:, :, :3]
    return np.ascontiguousarray(image)


def depth_image_to_meters(msg, depth_scale: float = 1000.0) -> np.ndarray:
    """Depth image (16UC1 / mono16 in ``1/depth_scale`` m, or 32FC1 in m) as float32 meters, invalid = 0."""
    depth = image_to_numpy(msg)
    if depth.ndim != 2:
        raise ValueError(f"depth image must be single-channel, got encoding {msg.encoding!r}")
    if depth.dtype == np.uint16:
        depth = depth.astype(np.float32) / float(depth_scale)
    else:
        depth = depth.astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    return depth


def numpy_to_image(array: np.ndarray, encoding: str, header=None):
    """Build a sensor_msgs/Image from an array whose layout matches ``encoding``."""
    from sensor_msgs.msg import Image

    array = np.ascontiguousarray(array)
    dtype, channels = _IMAGE_ENCODINGS[encoding]
    if array.dtype != dtype:
        raise ValueError(f"array dtype {array.dtype} does not match encoding {encoding!r}")
    msg = Image()
    if header is not None:
        msg.header = header
    msg.height, msg.width = int(array.shape[0]), int(array.shape[1])
    msg.encoding = encoding
    msg.is_bigendian = False
    msg.step = int(msg.width * channels * np.dtype(dtype).itemsize)
    msg.data = array.tobytes()
    return msg
