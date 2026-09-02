"""ScanNet scene loader for the Sec. V-B benchmark (Tables II-III).

Reads a scene exported with ScanNet's ``SensReader`` (the standard layout
every ScanNet pipeline consumes) plus the annotated mesh::

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
            rgb=frame.rgb,
            depth=frame.depth,
            detections=detections,
        )

    # ------------------------------------------------------------ ground truth
    def _find(self, *suffixes: str) -> Path | None:
        for suffix in suffixes:
            candidates = sorted(self.data_dir.glob(f"*{suffix}"))
            if candidates:
                return candidates[0]
        return None

    def ground_truth_points(self) -> GroundTruthPoints | None:
        """Annotated mesh vertices with NYU40 class names and instance ids.

        Returns ``None`` when the scene has no ``*_vh_clean_2.labels.ply``.
        Instance ids come from the segmentation + aggregation files when both
        are present, otherwise every point is instance -1 (class-level only).
        """
        labels_ply = self._find("_vh_clean_2.labels.ply")
        if labels_ply is None:
            return None
        vertices = read_ply_vertices(labels_ply)
        points = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float64)
        label_ids = np.asarray(vertices.get("label", np.zeros(points.shape[0])), dtype=np.int64)
        names = np.array(NYU40_CLASSES, dtype=str)
        valid = (label_ids > 0) & (label_ids < len(NYU40_CLASSES))
        labels = np.where(valid, names[np.clip(label_ids, 0, len(NYU40_CLASSES) - 1)], UNLABELED)

        instance_ids = np.full(points.shape[0], -1, dtype=np.int64)
        segs_json = self._find("_vh_clean_2.0.010000.segs.json")
        aggregation_json = self._find("_vh_clean.aggregation.json", ".aggregation.json")
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


def is_scannet_scene(path: str | Path) -> bool:
    return (Path(path) / "intrinsic" / "intrinsic_depth.txt").exists()
