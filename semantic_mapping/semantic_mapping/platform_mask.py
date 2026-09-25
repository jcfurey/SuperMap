"""The robot's own body in a camera image, from its URDF (``robot_description``).

A camera mounted on a vehicle sees parts of that vehicle: hood, blade, cab,
sensor mounts. An open-vocabulary detector labels them ("bulldozer", "truck"),
and LiDAR depth then lifts those masks onto whatever lies behind the body
(its own returns are usually cropped from the cloud), so an object travels
along with the robot. :class:`PlatformModel` renders the robot's link geometry
into a camera image at the links' current poses; pixels it covers show the
platform, not the scene.

No ROS dependency: the caller supplies each link's pose in the camera frame
(normally from TF, as ``robot_state_publisher`` broadcasts every link).
"""
from __future__ import annotations

import dataclasses
import math
import threading
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from semantic_mapping.geometry_utils import mask_bounds
from semantic_mapping.types import CameraIntrinsics, Detection2D

GEOMETRY_KINDS = ("visual", "collision")

_CYLINDER_SEGMENTS = 24
_SPHERE_RINGS = 12
_POSE_TOLERANCE = 1e-3
"""Reuse a link's rendering while its pose moved less than this (m, and rotation-matrix entries)."""


@dataclass(frozen=True)
class PlatformPart:
    """One ``<visual>`` or ``<collision>`` element of a link, as triangles in the link frame."""

    link: str
    kind: str
    """box | cylinder | sphere | mesh"""
    vertices: np.ndarray
    """(V, 3) vertices in the link frame (element origin and mesh scale applied)."""
    faces: np.ndarray
    """(F, 3) vertex indices."""
    T_geom_from_link: np.ndarray
    """Inverse of the element origin; places a point in the primitive's own frame."""
    size: tuple[float, ...] = ()
    """box: (x, y, z); cylinder: (radius, length); sphere: (radius,); mesh: ()."""

    def contains(self, point_link: np.ndarray) -> bool:
        """Whether a link-frame point lies inside this primitive (always False for meshes)."""
        p = self.T_geom_from_link[:3, :3] @ point_link + self.T_geom_from_link[:3, 3]
        if self.kind == "box":
            return bool(np.all(np.abs(p) <= np.asarray(self.size) / 2))
        if self.kind == "cylinder":
            return bool(math.hypot(p[0], p[1]) <= self.size[0] and abs(p[2]) <= self.size[1] / 2)
        if self.kind == "sphere":
            return bool(np.linalg.norm(p) <= self.size[0])
        return False


def urdf_origin(element) -> np.ndarray:
    """4x4 transform of a URDF ``<origin xyz rpy>`` (fixed-axis roll, pitch, yaw); identity when absent."""
    T = np.eye(4)
    if element is None:
        return T
    roll, pitch, yaw = (float(v) for v in element.get("rpy", "0 0 0").split())
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    T[:3, :3] = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                 [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                 [-sp, cp * sr, cp * cr]]
    T[:3, 3] = [float(v) for v in element.get("xyz", "0 0 0").split()]
    return T


def _box_mesh(size) -> tuple[np.ndarray, np.ndarray]:
    corners = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)]) * size
    faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                      [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return corners, faces


def _cylinder_mesh(radius: float, length: float) -> tuple[np.ndarray, np.ndarray]:
    # The polygon circumscribes the circle, so the silhouette never shrinks.
    n = _CYLINDER_SEGMENTS
    angle = 2 * np.pi * np.arange(n) / n
    r = radius / math.cos(math.pi / n)
    ring = np.column_stack([r * np.cos(angle), r * np.sin(angle)])
    vertices = np.vstack([np.column_stack([ring, np.full(n, -length / 2)]),
                          np.column_stack([ring, np.full(n, length / 2)]),
                          [[0, 0, -length / 2], [0, 0, length / 2]]])
    i, j = np.arange(n), (np.arange(n) + 1) % n
    faces = np.vstack([np.column_stack([i, j, n + j]), np.column_stack([i, n + j, n + i]),
                       np.column_stack([np.full(n, 2 * n), j, i]), np.column_stack([np.full(n, 2 * n + 1), n + i, n + j])])
    return vertices, faces


def _sphere_mesh(radius: float) -> tuple[np.ndarray, np.ndarray]:
    rings, segments = _SPHERE_RINGS, 2 * _SPHERE_RINGS
    # Scaled so every facet lies outside the sphere (a conservative silhouette).
    r = radius / (math.cos(math.pi / rings) * math.cos(math.pi / segments))
    polar = np.pi * np.arange(1, rings) / rings
    azimuth = 2 * np.pi * np.arange(segments) / segments
    body = np.array([[math.sin(p) * math.cos(a), math.sin(p) * math.sin(a), math.cos(p)] for p in polar for a in azimuth])
    vertices = r * np.vstack([[[0, 0, 1]], body, [[0, 0, -1]]])
    faces = []
    for a in range(segments):
        b = (a + 1) % segments
        faces.append([0, 1 + a, 1 + b])
        for ring in range(rings - 2):
            top, bottom = 1 + ring * segments, 1 + (ring + 1) * segments
            faces += [[top + a, bottom + a, bottom + b], [top + a, bottom + b, top + b]]
        last = 1 + (rings - 2) * segments
        faces.append([len(vertices) - 1, last + b, last + a])
    return vertices, np.array(faces)


def load_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (V, 3) and triangle faces (F, 3) of an STL (binary or ASCII) or OBJ file.

    Other formats (DAE, ...) load through ``trimesh`` when it is installed.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".stl":
        data = path.read_bytes()
        if len(data) >= 84:
            count = int(np.frombuffer(data, "<u4", count=1, offset=80)[0])
            if len(data) == 84 + 50 * count:
                record = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
                triangles = np.frombuffer(data, record, count=count, offset=84)["vertices"]
                return triangles.reshape(-1, 3).astype(np.float64), np.arange(3 * count).reshape(-1, 3)
        tokens = data.decode("ascii", errors="replace").split()
        vertices = np.array([[float(v) for v in tokens[i + 1:i + 4]] for i, token in enumerate(tokens) if token == "vertex"],
                            dtype=np.float64).reshape(-1, 3)
        if not len(vertices) or len(vertices) % 3:
            raise ValueError(f"{path}: not a triangle STL")
        return vertices, np.arange(len(vertices)).reshape(-1, 3)
    if suffix == ".obj":
        vertices, faces = [], []
        for line in path.read_text(errors="replace").splitlines():
            fields = line.split()
            if fields and fields[0] == "v":
                vertices.append([float(v) for v in fields[1:4]])
            elif fields and fields[0] == "f":
                index = [int(f.split("/")[0]) for f in fields[1:]]
                index = [i - 1 if i > 0 else len(vertices) + i for i in index]
                faces += [[index[0], index[k], index[k + 1]] for k in range(1, len(index) - 1)]
        if not faces:
            raise ValueError(f"{path}: no faces")
        return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int64)
    try:
        import trimesh
    except ImportError as exc:
        raise ValueError(f"{path}: {suffix} meshes need trimesh; STL and OBJ load without it") from exc
    mesh = trimesh.load(path, force="mesh")
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def simplify_mesh(vertices: np.ndarray, faces: np.ndarray, resolution: float) -> tuple[np.ndarray, np.ndarray]:
    """Vertex clustering: merge vertices within ``resolution`` cells and drop collapsed or repeated triangles.

    Keeps a silhouette to within about one cell while reducing a CAD mesh to
    what a camera mask needs; ``resolution <= 0`` returns the mesh unchanged.
    """
    if resolution <= 0 or not len(faces):
        return vertices, faces
    cells, inverse = np.unique(np.floor(vertices / resolution).astype(np.int64), axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    counts = np.bincount(inverse, minlength=len(cells))[:, None]
    merged = np.column_stack([np.bincount(inverse, vertices[:, k], len(cells)) for k in range(3)]) / counts
    faces = inverse[faces]
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
    faces = np.unique(np.sort(faces, axis=1), axis=0)
    return merged, faces


def resolve_mesh_path(uri: str, search_paths: Sequence[str | Path] = ()) -> Path:
    """Locate a URDF mesh ``filename``: ``package://pkg/rel``, ``file://abs`` or a plain path.

    ``package://`` looks in each search path for ``<path>/pkg/rel`` and
    ``<path>/share/pkg/rel`` (a source tree or an install prefix) before the
    ament index, so a robot's description package need not be built here.
    """
    if uri.startswith("file://"):
        path = Path(uri[len("file://"):])
        if path.is_file():
            return path
        raise FileNotFoundError(uri)
    if not uri.startswith("package://"):
        path = Path(uri)
        if path.is_file():
            return path
        raise FileNotFoundError(uri)
    package, _, relative = uri[len("package://"):].partition("/")
    for root in search_paths:
        for candidate in (Path(root) / package / relative, Path(root) / "share" / package / relative):
            if candidate.is_file():
                return candidate
    try:
        from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
    except ImportError:
        raise FileNotFoundError(f"{uri} (not under platform mesh paths; ament_index unavailable)") from None
    try:
        candidate = Path(get_package_share_directory(package)) / relative
    except PackageNotFoundError:
        raise FileNotFoundError(f"{uri} (package {package!r} is neither installed nor under the mesh paths)") from None
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(uri)


def _parse_geometry(link: str, element, resolve: Callable[[str], Path], mesh_resolution: float) -> PlatformPart:
    geometry = element.find("geometry")
    shape = next(iter(geometry), None) if geometry is not None else None
    if shape is None:
        raise ValueError("no geometry")
    T_link_from_geom = urdf_origin(element.find("origin"))
    size: tuple[float, ...] = ()
    if shape.tag == "box":
        size = tuple(float(v) for v in shape.get("size", "").split())
        if len(size) != 3:
            raise ValueError("box needs a 3-element size")
        vertices, faces = _box_mesh(np.asarray(size))
    elif shape.tag == "cylinder":
        size = (float(shape.get("radius")), float(shape.get("length")))
        vertices, faces = _cylinder_mesh(*size)
    elif shape.tag == "sphere":
        size = (float(shape.get("radius")),)
        vertices, faces = _sphere_mesh(*size)
    elif shape.tag == "mesh":
        vertices, faces = load_mesh(resolve(shape.get("filename", "")))
        scale = [float(v) for v in shape.get("scale", "1 1 1").split()]
        vertices = vertices * (scale if len(scale) == 3 else scale[:1] * 3)
        vertices, faces = simplify_mesh(vertices, faces, mesh_resolution)
    else:
        raise ValueError(f"unsupported geometry <{shape.tag}>")
    if any(not math.isfinite(v) or v <= 0 for v in size):
        raise ValueError(f"degenerate {shape.tag}")
    if not len(faces) or not np.isfinite(vertices).all():
        raise ValueError(f"empty or invalid {shape.tag}")
    vertices = vertices @ T_link_from_geom[:3, :3].T + T_link_from_geom[:3, 3]
    T_geom_from_link = np.eye(4)
    T_geom_from_link[:3, :3] = T_link_from_geom[:3, :3].T
    T_geom_from_link[:3, 3] = -T_link_from_geom[:3, :3].T @ T_link_from_geom[:3, 3]
    return PlatformPart(link, shape.tag, vertices, faces.astype(np.int64), T_geom_from_link, size)


def _clip_near(triangle: np.ndarray, near: float) -> np.ndarray:
    """Sutherland-Hodgman clip of one camera-frame polygon to z >= near."""
    out = []
    for a, b in zip(triangle, np.roll(triangle, -1, axis=0)):
        if a[2] >= near:
            out.append(a)
        if (a[2] >= near) != (b[2] >= near):
            out.append(a + (b - a) * ((near - a[2]) / (b[2] - a[2])))
    return np.array(out)


def _mask_bbox(mask: np.ndarray) -> np.ndarray:
    y1, y2, x1, x2 = mask_bounds(mask)
    return np.array([x1, y1, x2, y2], dtype=np.float64)


@dataclass
class _LinkRender:
    T_cam_from_link: np.ndarray
    window: tuple[int, int, int, int] | None
    """(y0, y1, x0, x1) of ``pixels`` in the image; None when nothing is visible."""
    pixels: np.ndarray | None
    encloses_camera: bool
    """A primitive of the link encloses the camera and was left out."""


class PlatformModel:
    """Link geometry of a robot and its camera-image mask.

    ``render`` caches each link's pixels per camera and reuses them while the
    link's pose in that camera does not change, so parts rigidly attached to
    the camera are drawn once and only articulated parts are redrawn.
    """

    def __init__(self, parts: Sequence[PlatformPart], skipped: Sequence[str] = (), *,
                 padding_px: int = 0, near_clip_m: float = 0.15) -> None:
        if padding_px < 0 or not math.isfinite(near_clip_m) or near_clip_m <= 0:
            raise ValueError("padding_px must be >= 0 and near_clip_m finite and positive")
        self.parts = list(parts)
        self.skipped = list(skipped)
        """One message per geometry that is not part of the mask, and why."""
        self.padding_px = int(padding_px)
        self.near_clip_m = float(near_clip_m)
        self._by_link: dict[str, list[PlatformPart]] = {}
        for part in self.parts:
            self._by_link.setdefault(part.link, []).append(part)
        self._lock = threading.Lock()
        self._cache: dict[tuple, dict[str, _LinkRender]] = {}
        self._union: dict[tuple, np.ndarray] = {}
        self.inside_warnings: set[tuple[str, str]] = set()
        """(camera key, link) pairs whose primitives enclose the camera and were left out."""

    @classmethod
    def from_urdf(cls, urdf: str, *, geometry: str = "visual", mesh_paths: Sequence[str | Path] = (),
                  mesh_resolution_m: float = 0.02, exclude_links: Iterable[str] = (), padding_px: int = 0,
                  near_clip_m: float = 0.15) -> PlatformModel:
        """Build from URDF XML. Each link uses its ``geometry`` elements (visual or collision);
        a link with none that load (missing mesh, unsupported format, zero size) falls back to the other kind.
        """
        if geometry not in GEOMETRY_KINDS:
            raise ValueError(f"geometry must be one of {GEOMETRY_KINDS}")
        root = ET.fromstring(urdf)
        if root.tag != "robot":
            raise ValueError("robot_description is not a URDF <robot>")
        excluded = set(exclude_links)
        fallback = GEOMETRY_KINDS[1 - GEOMETRY_KINDS.index(geometry)]

        def resolve(uri: str) -> Path:
            return resolve_mesh_path(uri, mesh_paths)

        parts, skipped = [], []
        for link in root.findall("link"):
            name = link.get("name", "")
            if name in excluded:
                continue
            for kind in (geometry, fallback):
                loaded, errors = [], []
                for element in link.findall(kind):
                    try:
                        loaded.append(_parse_geometry(name, element, resolve, mesh_resolution_m))
                    except (OSError, ValueError, TypeError, KeyError) as exc:
                        errors.append(f"{name} {kind}: {exc}")
                skipped += errors
                if loaded:
                    parts += loaded
                    break
        if not parts:
            raise ValueError("robot_description has no usable link geometry"
                             + (f" ({'; '.join(skipped[:3])})" if skipped else ""))
        return cls(parts, skipped, padding_px=padding_px, near_clip_m=near_clip_m)

    @property
    def links(self) -> list[str]:
        """Links that contribute geometry; the caller looks up each one's pose."""
        return sorted(self._by_link)

    @property
    def triangle_count(self) -> int:
        return sum(len(part.faces) for part in self.parts)

    def render(self, T_cam_from_link: Mapping[str, np.ndarray], intrinsics: CameraIntrinsics,
               camera_key: str = "") -> np.ndarray:
        """Read-only (H, W) boolean mask of platform pixels, dilated by ``padding_px``.

        ``T_cam_from_link`` maps every link in :attr:`links` into the optical
        frame (z forward). Geometry nearer than ``near_clip_m`` to the image
        plane is ignored, and a primitive that encloses the camera is left out
        (it would cover the whole image); meshes are drawn as surfaces.
        """
        import cv2

        shape = (int(intrinsics.height), int(intrinsics.width))
        key = (camera_key, shape, float(intrinsics.fx), float(intrinsics.fy), float(intrinsics.cx), float(intrinsics.cy))
        with self._lock:
            links = self._cache.setdefault(key, {})
            changed = False
            for link in self.links:
                T = np.asarray(T_cam_from_link[link], dtype=np.float64)
                cached = links.get(link)
                if cached is not None and np.abs(cached.T_cam_from_link - T).max() <= _POSE_TOLERANCE:
                    continue
                links[link] = self._render_link(link, T, intrinsics, shape, cv2)
                if links[link].encloses_camera:
                    self.inside_warnings.add((camera_key, link))
                changed = True
            if not changed and key in self._union:
                return self._union[key]
            mask = np.zeros(shape, np.uint8)
            for render in links.values():
                if render.window is not None:
                    y0, y1, x0, x1 = render.window
                    mask[y0:y1, x0:x1] |= render.pixels
            if self.padding_px and mask.any():
                size = 2 * self.padding_px + 1
                mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
            result = mask.astype(bool)
            result.flags.writeable = False
            self._union[key] = result
            return result

    def _render_link(self, link: str, T: np.ndarray, intr: CameraIntrinsics, shape, cv2) -> _LinkRender:
        height, width = shape
        near = self.near_clip_m
        camera_in_link = -T[:3, :3].T @ T[:3, 3]
        polygons, encloses = [], False
        for part in self._by_link[link]:
            if part.contains(camera_in_link):
                encloses = True
                continue
            triangles = (part.vertices @ T[:3, :3].T + T[:3, 3])[part.faces]
            front = triangles[:, :, 2] >= near
            whole = triangles[front.all(axis=1)]
            clipped = [_clip_near(t, near) for t in triangles[front.any(axis=1) & ~front.all(axis=1)]]
            for group in [whole, *[c[None] for c in clipped if len(c) >= 3]]:
                if not len(group):
                    continue
                u = intr.fx * group[:, :, 0] / group[:, :, 2] + intr.cx
                v = intr.fy * group[:, :, 1] / group[:, :, 2] + intr.cy
                seen = (u.max(axis=1) >= -0.5) & (u.min(axis=1) < width - 0.5) \
                    & (v.max(axis=1) >= -0.5) & (v.min(axis=1) < height - 0.5)
                if seen.any():
                    polygons.append(np.round(np.stack([u[seen], v[seen]], axis=-1)).astype(np.int64))
        if not polygons:
            return _LinkRender(T, None, None, encloses)
        points = np.concatenate([p.reshape(-1, 2) for p in polygons])
        x0, y0 = np.clip(points.min(axis=0), 0, [width, height])
        x1, y1 = np.clip(points.max(axis=0) + 1, 0, [width, height])
        pixels = np.zeros((y1 - y0, x1 - x0), np.uint8)
        offset = np.array([x0, y0])
        for group in polygons:
            for polygon in group - offset:
                cv2.fillConvexPoly(pixels, polygon.astype(np.int32), 1)
        return _LinkRender(T, (int(y0), int(y1), int(x0), int(x1)), pixels, encloses)


def exclude_platform(detections: Sequence[Detection2D], platform: np.ndarray,
                     max_overlap: float) -> tuple[list[Detection2D], int]:
    """Remove platform pixels from detections; drop those that are mostly platform.

    A detection whose mask (its box, when it has none) lies more than
    ``max_overlap`` on the platform is dropped. Others keep their off-platform
    mask pixels, with the box shrunk to them. Returns the kept detections and
    the number dropped; the inputs are not modified.
    """
    kept, dropped = [], 0
    height, width = platform.shape
    for detection in detections:
        if detection.mask is not None:
            # Only the mask's bounding crop can overlap: no whole-image passes per detection.
            bounds = mask_bounds(detection.mask)
            if bounds is None:
                dropped += 1
                continue
            y1, y2, x1, x2 = bounds
            crop = detection.mask[y1:y2, x1:x2]
            on_platform = crop & platform[y1:y2, x1:x2]
            area, on = int(np.count_nonzero(crop)), int(np.count_nonzero(on_platform))
            # A mask entirely on the platform is dropped even with max_overlap 1.
            if on > max_overlap * area or on == area:
                dropped += 1
                continue
            if on:
                mask = detection.mask.copy()
                mask[y1:y2, x1:x2] = crop & ~on_platform
                detection = dataclasses.replace(detection, mask=mask, bbox=_mask_bbox(mask))
        else:
            x1, y1, x2, y2 = np.round(np.asarray(detection.bbox, dtype=np.float64)).astype(np.int64)
            x1, x2 = np.clip([x1, x2], 0, width)
            y1, y2 = np.clip([y1, y2], 0, height)
            if x2 > x1 and y2 > y1 and platform[y1:y2, x1:x2].mean() > max_overlap:
                dropped += 1
                continue
        kept.append(detection)
    return kept, dropped
