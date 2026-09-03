#!/usr/bin/env python3
"""Generate a small synthetic RGB-D + odometry + detections demo sequence.

No public SuperMap dataset is bundled with this release, so this script
builds a deterministic, fully offline substitute: a robot orbits a small
room containing a handful of labeled axis-aligned "objects" while three of
them are scripted to disappear partway through and three new ones to
appear, mirroring the qualitative scenario in Sec. V-C of the paper (a
plant, trash can, and chair removed; a bucket, cart, and safety sign
added), plus a box that is moved to another spot and a backpack that is
taken away and brought back, the relocation and return cases whose
identities Sec. IV-B says must persist. Depth, RGB, camera pose, and
"boxer" pre-baked 2D detections are
rendered analytically (ray/box intersection) so ``examples/example.py`` can
run the full pipeline against real geometry without any external downloads
or GPU models.

Point this script's ``--out_dir`` at a real capture instead once one is
available; the on-disk layout is documented below and matches what
:class:`semantic_mapping.detectors.offline.OfflineDetector` and
``examples/example.py`` expect.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_mapping.geometry_utils import back_project_depth, rotation_matrix_to_quaternion  # noqa: E402
from semantic_mapping.segmentation_metrics import GroundTruthPoints  # noqa: E402

WORLD_UP = np.array([0.0, 0.0, 1.0])


@dataclass
class ScenePhase:
    """One contiguous presence interval of an object at one place."""

    appear_frac: float
    """Fraction of the sequence (0-1) at which the object becomes present here."""

    disappear_frac: float
    """Fraction of the sequence (0-1) at which it is removed from here (1.0 = never)."""

    center_xy: tuple[float, float]


@dataclass
class SceneObject:
    label: str
    half_extents: tuple[float, float, float]
    z_center: float
    color: tuple[int, int, int]
    phases: list[ScenePhase]
    """Where and when the object is present. One phase: static, removed, or
    added; two phases at different places: relocated; two at the same place:
    taken away and brought back."""

    def phase_at(self, frac: float) -> ScenePhase | None:
        return next((p for p in self.phases if p.appear_frac <= frac < p.disappear_frac), None)

    def is_present(self, frac: float) -> bool:
        return self.phase_at(frac) is not None

    def bbox_for(self, phase: ScenePhase) -> np.ndarray:
        cx, cy = phase.center_xy
        hx, hy, hz = self.half_extents
        return np.array([cx - hx, cy - hy, self.z_center - hz, cx + hx, cy + hy, self.z_center + hz])


def _obj(label, center_xy, half_extents, z_center, color, appear=0.0, disappear=1.0) -> SceneObject:
    return SceneObject(label, half_extents, z_center, color, [ScenePhase(appear, disappear, center_xy)])


# Mirrors the paper's qualitative long-horizon change scenario (Sec. V-C):
# three static fixtures, three objects removed partway through, three added,
# one relocated, one taken away and brought back.
SCENE_OBJECTS: list[SceneObject] = [
    _obj("table", (0.0, 1.6), (0.6, 0.4, 0.4), 0.4, (150, 110, 70)),
    _obj("sofa", (-2.2, -1.0), (0.9, 0.5, 0.4), 0.4, (90, 90, 160)),
    _obj("shelf", (2.4, -1.6), (0.4, 0.3, 1.0), 1.0, (120, 120, 120)),
    # Removed partway through:
    _obj("plant", (1.6, 1.4), (0.25, 0.25, 0.5), 0.5, (60, 150, 70), 0.0, 0.55),
    _obj("trash can", (-1.4, 1.7), (0.2, 0.2, 0.35), 0.35, (80, 80, 80), 0.0, 0.55),
    _obj("chair", (0.6, -1.8), (0.25, 0.25, 0.45), 0.45, (170, 130, 90), 0.0, 0.55),
    # Newly introduced partway through:
    _obj("bucket", (0.7, 0.3), (0.2, 0.2, 0.25), 0.25, (200, 60, 60), 0.45, 1.0),
    _obj("cart", (-2.0, 0.4), (0.35, 0.5, 0.5), 0.5, (60, 60, 200), 0.45, 1.0),
    _obj("safety sign", (2.0, 0.6), (0.05, 0.3, 0.5), 0.9, (230, 200, 40), 0.45, 1.0),
    # Relocated: same physical box, moved to the other side of the room.
    SceneObject("box", (0.2, 0.25, 0.2), 0.2, (210, 130, 40),
                [ScenePhase(0.0, 0.4, (-0.6, -0.6)), ScenePhase(0.6, 1.0, (1.2, -0.4))]),
    # Taken away and brought back to the same place.
    SceneObject("backpack", (0.2, 0.15, 0.25), 0.25, (160, 40, 160),
                [ScenePhase(0.0, 0.3, (-0.9, 0.9)), ScenePhase(0.65, 1.0, (-0.9, 0.9))]),
]


@dataclass
class PresentObject:
    """An object as it stands in one frame: its label, colour, and box."""

    label: str
    color: tuple[int, int, int]
    bbox3d: np.ndarray
    scene_index: int
    in_final_scene: bool
    """Whether this box is part of the scene at the end of the sequence (the
    segmentation ground truth scores the final state)."""

# Backdrop "room shell" the camera ray-casts against where it doesn't hit an
# object; must comfortably contain the whole camera orbit (radius=3.4 below)
# on every axis, or the shell's near face gets hit instead of its far face.
ROOM_BOUNDS = np.array([-5.5, -4.5, 0.0, 5.5, 4.5, 3.0])  # xmin,ymin,zmin,xmax,ymax,zmax

# Voxel size used to thin the accumulated ground-truth surface samples.
GT_VOXEL_M = 0.05

# An object with fewer rendered pixels than this is treated as not visible:
# no detection is emitted for it, and the frame is excluded from its
# evaluation "appearance interval" (see semantic_mapping/evaluation.py).
MIN_VISIBLE_PIXELS = 20


def _look_at_rotation(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation (world_from_cam) for a pinhole camera at ``position`` facing ``target``,
    using the standard optical convention (x right, y down, z forward)."""
    forward = target - position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(WORLD_UP, forward)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward], axis=1)


def _ray_box_hit(origin: np.ndarray, directions: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Vectorized ray/axis-aligned-box intersection distance (near-t); +inf if no hit."""
    box_min, box_max = box[:3], box[3:]
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (box_min[None, :] - origin[None, :]) / directions
        t2 = (box_max[None, :] - origin[None, :]) / directions
    t_near = np.max(np.minimum(t1, t2), axis=1)
    t_far = np.min(np.maximum(t1, t2), axis=1)
    hit = (t_far >= t_near) & (t_far >= 0)
    distance = np.where(t_near > 0, t_near, t_far)
    return np.where(hit, distance, np.inf)


def render_frame(
    position: np.ndarray,
    R_world_from_cam: np.ndarray,
    present_objects: list[PresentObject],
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytically ray-cast the synthetic room. Returns (depth, rgb, object_id_map)."""
    us, vs = np.meshgrid(np.arange(width), np.arange(height))
    dirs_cam = np.stack([
        (us.ravel() - cx) / fx,
        (vs.ravel() - cy) / fy,
        np.ones(us.size),
    ], axis=1)
    dirs_cam /= np.linalg.norm(dirs_cam, axis=1, keepdims=True)
    dirs_world = dirs_cam @ R_world_from_cam.T

    best_dist = _ray_box_hit(position, dirs_world, ROOM_BOUNDS)
    best_id = np.full(us.size, -1, dtype=np.int32)

    for idx, obj in enumerate(present_objects):
        dist = _ray_box_hit(position, dirs_world, obj.bbox3d)
        closer = dist < best_dist
        best_dist = np.where(closer, dist, best_dist)
        best_id = np.where(closer, idx, best_id)

    # _ray_box_hit returns range along the (unit) ray; a depth image stores
    # z-depth (distance along the optical axis), which is what real RGB-D
    # sensors emit and what the pipeline's pinhole back-projection assumes.
    # Storing range instead stretches peripheral points radially by 1/cos and
    # makes Eq. (9) see them as "in front of the surface", so the conversion
    # here is load-bearing, not cosmetic.
    depth = (best_dist * dirs_cam[:, 2]).reshape(height, width)
    depth[~np.isfinite(depth)] = 0.0
    object_id_map = best_id.reshape(height, width)

    rgb = np.full((height, width, 3), 235, dtype=np.uint8)
    shading = np.clip(1.0 - depth / 8.0, 0.35, 1.0)
    for idx, obj in enumerate(present_objects):
        mask = object_id_map == idx
        color = np.array(obj.color, dtype=np.float64)
        rgb[mask] = np.clip(color[None, :] * shading[mask, None], 0, 255).astype(np.uint8)

    return depth, rgb, object_id_map


def ground_truth_surfaces(
    depth: np.ndarray,
    object_id_map: np.ndarray,
    present_objects: list[PresentObject],
    K: np.ndarray,
    R_world_from_cam: np.ndarray,
    position: np.ndarray,
    sample_mask: np.ndarray,
    final_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World-frame surface samples with class labels and instance ids, the
    synthetic stand-in for ScanNet's annotated mesh (Sec. V-B benchmark).

    Scores the *final* scene: surfaces of objects removed before the end are
    dropped, and the room shell is labeled wall / floor / ceiling by the face
    the ray hit, so "with background" metrics have real background classes.
    """
    valid = sample_mask & (depth > 0)
    points_world = back_project_depth(K, depth, mask=valid) @ R_world_from_cam.T + position
    ids_local = object_id_map[valid]
    labels = np.full(points_world.shape[0], "", dtype=object)
    instance_ids = np.full(points_world.shape[0], -1, dtype=np.int64)

    shell = ids_local < 0
    z = points_world[shell, 2]
    labels[shell] = np.where(
        np.abs(z - ROOM_BOUNDS[2]) < 0.02, "floor", np.where(np.abs(z - ROOM_BOUNDS[5]) < 0.02, "ceiling", "wall"),
    )
    for local_idx, obj in enumerate(present_objects):
        if not obj.in_final_scene:
            continue
        member = ids_local == local_idx
        labels[member] = obj.label
        instance_ids[member] = obj.scene_index
    keep = labels != ""
    return points_world[keep], labels[keep].astype(str), instance_ids[keep]


def make_detections(
    present_objects: list[PresentObject],
    object_id_map: np.ndarray,
    rng: np.random.Generator,
    detections_dir: Path | None = None,
    frame_id: int = 0,
    with_masks: bool = True,
) -> list[dict]:
    """Pre-baked detections for one frame: a jittered box per visible object and,
    by default, its instance mask -- mirroring the paper's Grounding DINO box +
    SAM2 mask pairing. ``with_masks=False`` emits box-only records to exercise
    the harder fallback path.
    """
    detections = []
    for idx, obj in enumerate(present_objects):
        mask = object_id_map == idx
        ys, xs = np.nonzero(mask)
        if xs.size < MIN_VISIBLE_PIXELS:  # object not (meaningfully) visible this frame
            continue
        jitter = rng.normal(0.0, 1.5, size=4)
        bbox = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float64) + jitter
        record = {
            "bbox": bbox.tolist(),
            "label": obj.label,
            "score": float(np.clip(rng.normal(0.9, 0.05), 0.5, 0.99)),
        }
        if with_masks and detections_dir is not None:
            mask_name = f"{frame_id:06d}_{len(detections)}.npy"
            np.save(detections_dir / mask_name, mask)
            record["mask"] = mask_name
        detections.append(record)
    return detections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", default="data/example_scene", help="Output dataset directory.")
    parser.add_argument("--num_frames", type=int, default=60)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_masks", action="store_true",
                        help="Emit box-only detections (no instance masks) to exercise the fallback path.")
    parser.add_argument("--lidar_like", action="store_true",
                        help="Store sparse, noisy depth as a LiDAR scan rasterized into the camera would give "
                             "(RGB, detections, and ground truth stay dense).")
    parser.add_argument("--lidar_density", type=float, default=0.05,
                        help="Fraction of pixels carrying a depth reading with --lidar_like.")
    parser.add_argument("--lidar_noise_m", type=float, default=0.02, help="Range noise (1 sigma) with --lidar_like.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    detections_dir = out_dir / "detections"
    frames_dir.mkdir(parents=True, exist_ok=True)
    detections_dir.mkdir(parents=True, exist_ok=True)

    fx = fy = 0.9 * args.width
    cx, cy = args.width / 2.0, args.height / 2.0
    intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": args.width, "height": args.height}
    (out_dir / "intrinsics.json").write_text(json.dumps(intrinsics, indent=2))

    rng = np.random.default_rng(args.seed)
    radius, cam_height = 3.4, 1.2
    target = np.array([0.0, 0.0, 0.9])
    theta0, theta1 = np.deg2rad(200), np.deg2rad(-20)

    # Per (object, phase): frames in which it is present / meaningfully visible.
    visible_frames: dict[tuple[int, int], list[int]] = {}
    present_frames: dict[tuple[int, int], list[int]] = {}

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    final_frac = (args.num_frames - 1) / args.num_frames
    sample_mask = np.zeros((args.height, args.width), dtype=bool)
    sample_mask[::2, ::2] = True
    gt_chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for frame_id in range(args.num_frames):
        # Orbit progress spans the full arc (last frame lands on theta1);
        # presence uses a half-open fraction in [0, 1) so disappear_frac=1.0
        # really means "never removed" on the final frame too.
        orbit_frac = frame_id / max(args.num_frames - 1, 1)
        frac = frame_id / args.num_frames
        stamp = frame_id / args.fps

        theta = theta0 + (theta1 - theta0) * orbit_frac
        position = np.array([radius * np.cos(theta), radius * np.sin(theta), cam_height])
        assert np.all(position > ROOM_BOUNDS[:3]) and np.all(position < ROOM_BOUNDS[3:]), (
            "camera orbit must stay strictly inside ROOM_BOUNDS for the backdrop ray-cast to hit its far face"
        )
        R_world_from_cam = _look_at_rotation(position, target)

        present_objects: list[PresentObject] = []
        phase_keys: list[tuple[int, int]] = []
        for idx, obj in enumerate(SCENE_OBJECTS):
            phase = obj.phase_at(frac)
            if phase is None:
                continue
            phase_idx = obj.phases.index(phase)
            in_final_scene = phase_idx == len(obj.phases) - 1 and obj.is_present(final_frac)
            present_objects.append(PresentObject(obj.label, obj.color, obj.bbox_for(phase), idx, in_final_scene))
            phase_keys.append((idx, phase_idx))
        depth, rgb, object_id_map = render_frame(
            position, R_world_from_cam, present_objects, args.width, args.height, fx, fy, cx, cy,
        )

        gt_chunks.append(ground_truth_surfaces(
            depth, object_id_map, present_objects, K, R_world_from_cam, position, sample_mask, final_frac,
        ))

        stored_depth = depth
        if args.lidar_like:
            keep = rng.random(depth.shape) < args.lidar_density
            stored_depth = np.where(keep & (depth > 0), depth + rng.normal(0.0, args.lidar_noise_m, depth.shape), 0.0)
            stored_depth = np.maximum(stored_depth, 0.0)

        frame_prefix = frames_dir / f"{frame_id:06d}"
        np.save(f"{frame_prefix}_depth.npy", stored_depth.astype(np.float32))
        np.save(f"{frame_prefix}_rgb.npy", rgb)

        quaternion = rotation_matrix_to_quaternion(R_world_from_cam)
        pose = {"stamp": stamp, "translation": position.tolist(), "quaternion": quaternion.tolist()}
        (frames_dir / f"{frame_id:06d}_pose.json").write_text(json.dumps(pose))

        detections = make_detections(
            present_objects, object_id_map, rng, detections_dir, frame_id, with_masks=not args.no_masks,
        )
        (detections_dir / f"{frame_id:06d}.json").write_text(json.dumps({"detections": detections}))

        for local_idx, key in enumerate(phase_keys):
            present_frames.setdefault(key, []).append(frame_id)
            if int((object_id_map == local_idx).sum()) >= MIN_VISIBLE_PIXELS:
                visible_frames.setdefault(key, []).append(frame_id)

    # One ground-truth entry per presence phase; entries of the same physical
    # object share an identity, which is what the identity-consistency metric
    # scores (the same instance ID must serve every phase).
    scene_meta = {
        "num_frames": args.num_frames,
        "fps": args.fps,
        "objects": [
            {
                "label": o.label,
                "identity": f"{o.label}#{idx}",
                "phase": phase_idx,
                "bbox3d": o.bbox_for(phase).tolist(),
                "appear_frac": phase.appear_frac,
                "disappear_frac": phase.disappear_frac,
                # Each phase is a single contiguous interval by construction.
                "appear_frame": present_frames.get((idx, phase_idx), [args.num_frames])[0],
                "disappear_frame": (present_frames[(idx, phase_idx)][-1] + 1
                                    if (idx, phase_idx) in present_frames else args.num_frames),
                "visible_frames": visible_frames.get((idx, phase_idx), []),
            }
            for idx, o in enumerate(SCENE_OBJECTS)
            for phase_idx, phase in enumerate(o.phases)
        ],
    }
    (out_dir / "scene_ground_truth.json").write_text(json.dumps(scene_meta, indent=2))

    # Segmentation benchmark ground truth (Sec. V-B): every surface sample seen
    # over the sequence, thinned to one per (voxel, instance).
    points = np.concatenate([c[0] for c in gt_chunks], axis=0)
    labels = np.concatenate([c[1] for c in gt_chunks])
    instance_ids = np.concatenate([c[2] for c in gt_chunks])
    keys = np.concatenate([np.floor(points / GT_VOXEL_M).astype(np.int64), instance_ids[:, None]], axis=1)
    _unique, first = np.unique(keys, axis=0, return_index=True)
    first = np.sort(first)
    GroundTruthPoints(points[first], labels[first], instance_ids[first]).save(out_dir / "gt_points.npz")

    print(f"Wrote {args.num_frames} synthetic frames to {out_dir}/ "
          f"({first.size} labeled ground-truth surface points in gt_points.npz)")
    print("This is a deterministic offline substitute for a real capture -- "
          "point --data_dir at a real RGB-D/LiDAR sequence for actual deployment.")


if __name__ == "__main__":
    main()
