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


def camera_info_has_distortion(info, tolerance: float = 1e-12) -> bool:
    """True when CameraInfo carries any nonzero (or non-finite) distortion coefficient."""
    d = np.asarray(info.d, dtype=np.float64)
    return bool(d.size and (not np.isfinite(d).all() or np.any(np.abs(d) > tolerance)))


def camera_info_to_intrinsics(info, *, rectified: bool = False, allow_distortion: bool = True) -> CameraIntrinsics:
    """Convert full-resolution calibration into the emitted image's pixel coordinates.

    ``rectified=False`` (the default, for ``image_raw``-style topics) uses
    ``K``; ``rectified=True`` uses the projection matrix ``P``, which is what
    describes ``image_rect*`` topics (image_proc's output), and ignores ``D``
    since those images are already undistorted. The pinhole model here cannot
    represent lens distortion: with ``allow_distortion=False`` a raw stream
    whose ``D`` is nonzero is rejected instead of silently projected with
    ``K`` alone. The default keeps the historical permissive behaviour for
    callers that accept the error (review item C29).

    CameraInfo dimensions and ROI are unbinned sensor coordinates. Subtract
    the crop origin before dividing intrinsics and ROI size by binning; zero
    binning means one, and an all-zero ROI means the full image. The incoming
    message is never modified.
    """
    full_width, full_height = int(info.width), int(info.height)
    if full_width <= 0 or full_height <= 0:
        raise ValueError("CameraInfo must have positive calibration dimensions")
    if rectified:
        fx, fy, cx, cy = (float(info.p[i]) for i in (0, 5, 2, 6))
        matrix = "P"
    else:
        if not allow_distortion and camera_info_has_distortion(info):
            raise ValueError("CameraInfo has lens distortion; use a rectified image topic (rectified=True) "
                             "or undistort before projecting with the pinhole model")
        fx, fy, cx, cy = (float(info.k[i]) for i in (0, 4, 2, 5))
        matrix = "K"
    if not np.all(np.isfinite([fx, fy, cx, cy])) or fx <= 0 or fy <= 0:
        raise ValueError(f"CameraInfo {matrix} must contain finite, positive focal lengths and a finite principal point")
    x, y, width, height = (int(getattr(info.roi, name))
                           for name in ('x_offset', 'y_offset', 'width', 'height'))
    if (x, y, width, height) == (0, 0, 0, 0):
        width, height = full_width, full_height
    elif (x < 0 or y < 0 or width <= 0 or height <= 0
          or x + width > full_width or y + height > full_height):
        raise ValueError("CameraInfo ROI must be a nonempty rectangle within the calibration dimensions")
    bx, by = max(1, int(info.binning_x)), max(1, int(info.binning_y))
    width, height = width // bx, height // by
    if width <= 0 or height <= 0:
        raise ValueError("CameraInfo binning leaves an empty image")
    return CameraIntrinsics(
        fx=fx / bx, fy=fy / by, cx=(cx - x) / bx, cy=(cy - y) / by,
        width=width, height=height,
    )


def rectified_camera_info(info):
    """CameraInfo describing the *rectified* image of ``info``: ``K := P[:, :3]``, ``D := 0``, ``R := I``.

    image_proc republishes the camera's original CameraInfo next to
    ``image_rect*``; its ``D``/``R``/``K`` describe the raw sensor, while ``P``
    describes the rectified pixels. Consumers that assume an undistorted
    pinhole (mask projection) need this form. The input is not modified.
    """
    import copy

    fx, fy, cx, cy = (float(info.p[i]) for i in (0, 5, 2, 6))
    if not np.all(np.isfinite([fx, fy, cx, cy])) or fx <= 0 or fy <= 0:
        raise ValueError("CameraInfo P must contain finite, positive focal lengths for a rectified image")
    out = copy.deepcopy(info)
    out.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    out.d = [0.0] * len(info.d) if len(info.d) else []
    out.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    return out


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

        decoded = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
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
    height, width = int(msg.height), int(msg.width)
    row_values = int(msg.step) // dtype.itemsize
    if row_values < width * channels or len(msg.data) < height * int(msg.step):
        raise ValueError("image step/data too small for its width, height and encoding")
    # A view of the message buffer: no bytes() copy, and padding is sliced away lazily.
    data = np.frombuffer(msg.data, dtype=dtype, count=height * row_values).reshape(height, row_values)
    data = data[:, : width * channels]
    image = data.reshape(height, width, channels) if channels > 1 else data.reshape(height, width)
    if msg.encoding in _BAYER_ENCODINGS:
        import cv2

        # cvtColor allocates the output; it only needs a contiguous input.
        return cv2.cvtColor(np.ascontiguousarray(image), getattr(cv2, _BAYER_ENCODINGS[msg.encoding]))
    if msg.encoding in ("bgr8", "bgra8"):
        image = image[:, :, :3][:, :, ::-1]
    elif msg.encoding == "rgba8":
        image = image[:, :, :3]
    if dtype.byteorder not in ("=", "|") and dtype.byteorder != np.dtype(dtype.type).byteorder:
        # Byte-swapped wire data: one converting copy, already C-contiguous.
        return np.ascontiguousarray(image.astype(dtype.newbyteorder("=")))
    # Exactly one copy: the result owns its memory and never aliases the message.
    return np.array(image, dtype=image.dtype.newbyteorder("="), order="C", copy=True)


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
