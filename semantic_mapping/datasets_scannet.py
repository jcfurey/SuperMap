"""ScanNet scene loaders for the Sec. V-B benchmark (Tables II-III).

Two inputs are supported: the raw ``<scene>.sens`` capture as downloaded
(:class:`ScanNetSensSequence`, decoded lazily, no export step needed) and a
scene exported with ScanNet's ``SensReader`` (:class:`ScanNetScene`, the
layout most ScanNet pipelines consume). Both read the annotated mesh next to
the capture::

    <scene_dir>/                                 e.g. scans/scene0000_00
      color/<frame>.jpg                           RGB (typically 1296x968)
      depth/<frame>.png                           16-bit depth, millimeters (640x480)
      pose/<frame>.txt                            4x4 camera-to-world (world-from-camera)
      intrinsic/intrinsic_depth.txt               4x4 depth intrinsics
      <scene>_vh_clean_2.labels.ply               mesh vertices with NYU40 ``label``
      <scene>_vh_clean_2.0.010000.segs.json       over-segmentation (segment id per vertex)
      <scene>_vh_clean.aggregation.json           instances = groups of segments (+ raw label)
      detections/<frame:06d>.json                 optional pre-baked detections (detectors/offline.py)

RGB is resized to the depth resolution so a single set of intrinsics
describes both, which is what the pipeline's back-projection and membership
updates assume. Poses that ScanNet marks invalid (``-inf`` entries) are
skipped. Frames carry synthetic timestamps at the 30 Hz capture rate.

No ScanNet data is bundled (it requires the dataset's terms of use); the
loader is exercised in tests on a miniature scene written in the same layout.
"""
from __future__ import annotations

import json
import re
import struct
import zlib
from pathlib import Path
from typing import Iterator

import numpy as np

from semantic_mapping.datasets import SequenceFrame
from semantic_mapping.segmentation_metrics import UNLABELED, GroundTruthPoints
from semantic_mapping.types import CameraIntrinsics, Detection2D, Observation, StampedPose

SCANNET_FPS = 30.0

NYU40_CLASSES = [
    "unlabeled", "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf",
    "picture", "counter", "blinds", "desk", "shelves", "curtain", "dresser", "pillow", "mirror", "floor mat",
    "clothes", "ceiling", "books", "refridgerator", "television", "paper", "towel", "shower curtain", "box",
    "whiteboard", "person", "nightstand", "toilet", "sink", "lamp", "bathtub", "bag", "otherstructure",
    "otherfurniture", "otherprop",
]
"""NYU40 class names indexed by the ``label`` field of ``*_vh_clean_2.labels.ply``
(the benchmark keeps the dataset's own spelling of "refridgerator")."""

NYU40_ALIASES = {
    "refrigerator": "refridgerator", "fridge": "refridgerator", "couch": "sofa", "tv": "television",
    "monitor": "television", "bookcase": "bookshelf", "book shelf": "bookshelf", "shelf": "shelves",
    "night stand": "nightstand", "bath tub": "bathtub", "white board": "whiteboard", "rug": "floor mat",
    "book": "books", "trash can": "otherprop", "garbage can": "otherprop", "bin": "otherprop",
}
"""Detector vocabulary spellings folded onto NYU40 class names before scoring."""


# ------------------------------------------------------------------ images
def _read_image(path: Path, grayscale16: bool = False) -> np.ndarray:
    try:
        import cv2

        flags = cv2.IMREAD_UNCHANGED if grayscale16 else cv2.IMREAD_COLOR
        image = cv2.imread(str(path), flags)
        if image is None:
            raise FileNotFoundError(path)
        return image if grayscale16 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except ImportError:
        from PIL import Image

        image = Image.open(path)
        return np.array(image if grayscale16 else image.convert("RGB"))


def _resize_rgb(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    if rgb.shape[1] == width and rgb.shape[0] == height:
        return rgb
    try:
        import cv2

        return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    except ImportError:
        from PIL import Image

        return np.array(Image.fromarray(rgb).resize((width, height), Image.BILINEAR))


# --------------------------------------------------------------------- PLY
_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def read_ply_vertices(path: str | Path) -> dict[str, np.ndarray]:
    """Read the ``vertex`` element of an ASCII or binary PLY file as {property: array}.

    Only the vertex block is parsed (faces and other elements are skipped),
    which is all the annotated ScanNet mesh needs: coordinates and labels.
    """
    with open(path, "rb") as f:
        header: list[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: unterminated PLY header")
            header.append(line.decode("ascii", errors="replace").strip())
            if header[-1] == "end_header":
                break
        body_offset = f.tell()

    fmt_match = next((re.match(r"format\s+(\S+)", h) for h in header if h.startswith("format")), None)
    if fmt_match is None:
        raise ValueError(f"{path}: missing PLY format line")
    fmt = fmt_match.group(1)

    elements: list[tuple[str, int, list[tuple[str, str]]]] = []
    for line in header:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "element":
            elements.append((parts[1], int(parts[2]), []))
        elif parts[0] == "property" and elements:
            if parts[1] == "list":
                elements[-1][2].append((parts[4], f"list:{parts[2]}:{parts[3]}"))
            else:
                elements[-1][2].append((parts[2], parts[1]))

    if not elements or elements[0][0] != "vertex":
        raise ValueError(f"{path}: expected 'vertex' to be the first PLY element")
    _name, count, props = elements[0]
    if any(t.startswith("list") for _, t in props):
        raise ValueError(f"{path}: list properties on vertices are not supported")

    if fmt == "ascii":
        rows = np.loadtxt(path, skiprows=len(header), max_rows=count, ndmin=2)
        return {name: rows[:, i].astype(_PLY_TYPES[t]) for i, (name, t) in enumerate(props)}

    endian = "<" if fmt == "binary_little_endian" else ">"
    dtype = np.dtype([(name, endian + _PLY_TYPES[t]) for name, t in props])
    with open(path, "rb") as f:
        f.seek(body_offset)
        data = np.fromfile(f, dtype=dtype, count=count)
    if data.shape[0] != count:
        raise ValueError(f"{path}: vertex block truncated ({data.shape[0]} of {count})")
    return {name: np.ascontiguousarray(data[name]) for name, _ in props}


# ------------------------------------------------------------------- scene
class ScanNetScene:
    """Iterate one ScanNet scene as :class:`~semantic_mapping.datasets.SequenceFrame` objects."""

    def __init__(
        self,
        scene_dir: str | Path,
        frame_skip: int = 1,
        max_frames: int | None = None,
        depth_scale: float = 1000.0,
    ) -> None:
        self.data_dir = Path(scene_dir)
        self.scene_id = self.data_dir.name
        self.depth_scale = depth_scale
        intrinsic_path = self.data_dir / "intrinsic" / "intrinsic_depth.txt"
        if not intrinsic_path.exists():
            raise FileNotFoundError(f"No ScanNet scene at {self.data_dir} (missing {intrinsic_path.name})")
        K = np.loadtxt(intrinsic_path)

        depth_files = {int(p.stem): p for p in (self.data_dir / "depth").glob("*.png") if p.stem.isdigit()}
        if not depth_files:
            raise FileNotFoundError(f"{self.data_dir}/depth contains no <frame>.png images")
        first = _read_image(depth_files[min(depth_files)], grayscale16=True)
        self.intrinsics = CameraIntrinsics(
            fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]),
            width=int(first.shape[1]), height=int(first.shape[0]),
        )

        frame_ids = sorted(depth_files)[:: max(int(frame_skip), 1)]
        self.frame_ids = [fid for fid in frame_ids if self._pose_valid(fid)]
        if max_frames is not None:
            self.frame_ids = self.frame_ids[:max_frames]
        self.detections_dir = self.data_dir / "detections"

    def _pose_path(self, frame_id: int) -> Path:
        return self.data_dir / "pose" / f"{frame_id}.txt"

    def _pose_valid(self, frame_id: int) -> bool:
        path = self._pose_path(frame_id)
        return path.exists() and bool(np.all(np.isfinite(np.loadtxt(path))))

    def __len__(self) -> int:
        return len(self.frame_ids)

    def __iter__(self) -> Iterator[SequenceFrame]:
        for frame_id in self.frame_ids:
            depth_raw = _read_image(self.data_dir / "depth" / f"{frame_id}.png", grayscale16=True)
            depth = depth_raw.astype(np.float32) / self.depth_scale
            rgb = _resize_rgb(
                _read_image(self.data_dir / "color" / f"{frame_id}.jpg"), self.intrinsics.width, self.intrinsics.height,
            )
            yield SequenceFrame(
                frame_id=frame_id,
                stamp=frame_id / SCANNET_FPS,
                rgb=np.ascontiguousarray(rgb),
                depth=depth,
                T_world_from_cam=np.loadtxt(self._pose_path(frame_id)).reshape(4, 4),
            )

    def observation(self, frame: SequenceFrame, detections: list[Detection2D]) -> Observation:
        return Observation(
            stamp=frame.stamp,
            pose=StampedPose(stamp=frame.stamp, T_world_from_frame=frame.T_world_from_cam),
            intrinsics=self.intrinsics,
            frame_id=frame.frame_id,
            rgb=frame.rgb,
            depth=frame.depth,
            detections=detections,
        )

    def ground_truth_points(self) -> GroundTruthPoints | None:
        return scannet_ground_truth_points(self.data_dir)


# ------------------------------------------------------------ ground truth
def _find_in(scene_dir: Path, *suffixes: str) -> Path | None:
    for suffix in suffixes:
        candidates = sorted(scene_dir.glob(f"*{suffix}"))
        if candidates:
            return candidates[0]
    return None


def scannet_ground_truth_points(scene_dir: str | Path) -> GroundTruthPoints | None:
    """Annotated mesh vertices with NYU40 class names and instance ids.

    Returns ``None`` when the scene has no ``*_vh_clean_2.labels.ply``.
    Instance ids come from the segmentation + aggregation files when both
    are present, otherwise every point is instance -1 (class-level only).
    """
    scene_dir = Path(scene_dir)
    labels_ply = _find_in(scene_dir, "_vh_clean_2.labels.ply")
    if labels_ply is None:
        return None
    vertices = read_ply_vertices(labels_ply)
    points = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float64)
    label_ids = np.asarray(vertices.get("label", np.zeros(points.shape[0])), dtype=np.int64)
    names = np.array(NYU40_CLASSES, dtype=str)
    valid = (label_ids > 0) & (label_ids < len(NYU40_CLASSES))
    labels = np.where(valid, names[np.clip(label_ids, 0, len(NYU40_CLASSES) - 1)], UNLABELED)

    instance_ids = np.full(points.shape[0], -1, dtype=np.int64)
    segs_json = _find_in(scene_dir, "_vh_clean_2.0.010000.segs.json")
    aggregation_json = _find_in(scene_dir, "_vh_clean.aggregation.json", ".aggregation.json")
    if segs_json is not None and aggregation_json is not None:
        seg_indices = np.asarray(json.loads(segs_json.read_text())["segIndices"], dtype=np.int64)
        if seg_indices.shape[0] == points.shape[0]:
            seg_to_object: dict[int, int] = {}
            for group in json.loads(aggregation_json.read_text())["segGroups"]:
                for seg in group["segments"]:
                    seg_to_object[int(seg)] = int(group["objectId"])
            lookup = np.full(int(seg_indices.max()) + 1, -1, dtype=np.int64)
            for seg, obj in seg_to_object.items():
                if seg < lookup.shape[0]:
                    lookup[seg] = obj
            instance_ids = lookup[seg_indices]
    return GroundTruthPoints(points=points, labels=labels, instance_ids=instance_ids)


# ---------------------------------------------------------------- .sens
SENS_COLOR_COMPRESSION = {0: "raw", 1: "png", 2: "jpeg"}
SENS_DEPTH_COMPRESSION = {0: "raw_ushort", 1: "zlib_ushort", 2: "occi_ushort"}


class ScanNetSensSequence:
    """Read a raw ScanNet ``.sens`` capture without exporting it first.

    The file is a small header (sensor name, colour and depth intrinsics and
    extrinsics, compression types, image sizes, depth shift, frame count)
    followed by one record per frame: the camera-to-world pose, two
    timestamps, and the compressed colour (JPEG) and depth (zlib, 16-bit
    millimetres) payloads. Construction reads only the fixed-size record
    heads to index frame offsets and poses; frames are decoded on demand.
    Colour is resized to the depth resolution and the depth intrinsics are
    used, as with the exported layout.
    """

    HEADER_FLOATS = 16

    def __init__(self, sens_path: str | Path, frame_skip: int = 1, max_frames: int | None = None) -> None:
        self.sens_path = Path(sens_path)
        self.data_dir = self.sens_path.parent
        self.scene_id = self.sens_path.stem
        self.detections_dir = self.data_dir / "detections"
        with open(self.sens_path, "rb") as f:
            self._read_header(f)
            self._index_frames(f)
        self.intrinsics = CameraIntrinsics(
            fx=float(self.intrinsic_depth[0, 0]), fy=float(self.intrinsic_depth[1, 1]),
            cx=float(self.intrinsic_depth[0, 2]), cy=float(self.intrinsic_depth[1, 2]),
            width=int(self.depth_width), height=int(self.depth_height),
        )
        valid = [i for i in range(self.num_frames) if np.all(np.isfinite(self._poses[i]))]
        self.frame_ids = valid[:: max(int(frame_skip), 1)]
        if max_frames is not None:
            self.frame_ids = self.frame_ids[:max_frames]

    def _read_header(self, f) -> None:
        (self.version,) = struct.unpack("<I", f.read(4))
        (name_length,) = struct.unpack("<Q", f.read(8))
        self.sensor_name = f.read(name_length).decode("utf-8", errors="replace")
        matrices = []
        for _ in range(4):
            matrices.append(np.array(struct.unpack("<16f", f.read(64)), dtype=np.float64).reshape(4, 4))
        self.intrinsic_color, self.extrinsic_color, self.intrinsic_depth, self.extrinsic_depth = matrices
        color_compression, depth_compression = struct.unpack("<ii", f.read(8))
        self.color_compression = SENS_COLOR_COMPRESSION.get(color_compression, str(color_compression))
        self.depth_compression = SENS_DEPTH_COMPRESSION.get(depth_compression, str(depth_compression))
        self.color_width, self.color_height, self.depth_width, self.depth_height = struct.unpack("<4I", f.read(16))
        (self.depth_shift,) = struct.unpack("<f", f.read(4))
        (self.num_frames,) = struct.unpack("<Q", f.read(8))
        if self.depth_compression not in ("raw_ushort", "zlib_ushort"):
            raise ValueError(f"{self.sens_path}: unsupported depth compression {self.depth_compression!r}")

    def _index_frames(self, f) -> None:
        self._poses = np.zeros((self.num_frames, 4, 4), dtype=np.float64)
        self._stamps = np.zeros(self.num_frames, dtype=np.float64)
        self._records: list[tuple[int, int, int]] = []  # (offset of colour payload, colour bytes, depth bytes)
        for i in range(self.num_frames):
            head = f.read(64 + 8 + 8 + 8 + 8)
            if len(head) < 96:
                raise ValueError(f"{self.sens_path}: truncated at frame {i}")
            self._poses[i] = np.array(struct.unpack("<16f", head[:64]), dtype=np.float64).reshape(4, 4)
            stamp_color, stamp_depth, color_bytes, depth_bytes = struct.unpack("<QQQQ", head[64:96])
            self._stamps[i] = (stamp_depth or stamp_color) * 1e-6 if (stamp_depth or stamp_color) else i / SCANNET_FPS
            self._records.append((f.tell(), color_bytes, depth_bytes))
            f.seek(color_bytes + depth_bytes, 1)

    def __len__(self) -> int:
        return len(self.frame_ids)

    def _decode(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        offset, color_bytes, depth_bytes = self._records[index]
        with open(self.sens_path, "rb") as f:
            f.seek(offset)
            color_data = f.read(color_bytes)
            depth_data = f.read(depth_bytes)
        if self.color_compression == "raw":
            rgb = np.frombuffer(color_data, dtype=np.uint8).reshape(self.color_height, self.color_width, -1)[:, :, :3]
        else:
            import cv2

            decoded = cv2.imdecode(np.frombuffer(color_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError(f"{self.sens_path}: could not decode colour frame {index}")
            rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        raw = zlib.decompress(depth_data) if self.depth_compression == "zlib_ushort" else depth_data
        depth = np.frombuffer(raw, dtype="<u2").reshape(self.depth_height, self.depth_width)
        return rgb, depth.astype(np.float32) / float(self.depth_shift or 1000.0)

    def __iter__(self) -> Iterator[SequenceFrame]:
        for frame_id in self.frame_ids:
            rgb, depth = self._decode(frame_id)
            yield SequenceFrame(
                frame_id=frame_id,
                stamp=float(self._stamps[frame_id]),
                rgb=np.ascontiguousarray(_resize_rgb(rgb, self.intrinsics.width, self.intrinsics.height)),
                depth=depth,
                T_world_from_cam=self._poses[frame_id].copy(),
            )

    def observation(self, frame: SequenceFrame, detections: list[Detection2D]) -> Observation:
        return Observation(
            stamp=frame.stamp,
            pose=StampedPose(stamp=frame.stamp, T_world_from_frame=frame.T_world_from_cam),
            intrinsics=self.intrinsics,
            frame_id=frame.frame_id,
            rgb=frame.rgb,
            depth=frame.depth,
            detections=detections,
        )

    def ground_truth_points(self) -> GroundTruthPoints | None:
        return scannet_ground_truth_points(self.data_dir)


def is_scannet_scene(path: str | Path) -> bool:
    return (Path(path) / "intrinsic" / "intrinsic_depth.txt").exists()


def find_sens(path: str | Path) -> Path | None:
    """The ``.sens`` file a path denotes: the file itself, or the only one in a scene directory."""
    path = Path(path)
    if path.is_file() and path.suffix == ".sens":
        return path
    if path.is_dir():
        candidates = sorted(path.glob("*.sens"))
        if len(candidates) == 1:
            return candidates[0]
    return None
