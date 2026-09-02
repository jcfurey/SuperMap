#!/usr/bin/env python3
"""Generate a small synthetic RGB-D + odometry + detections demo sequence.

No public SuperMap dataset is bundled with this release, so this script
builds a deterministic, fully offline substitute: a robot orbits a small
room containing a handful of labeled axis-aligned "objects" while three of
them are scripted to disappear partway through and three new ones to
appear, mirroring the qualitative scenario in Sec. V-C of the paper (a
plant, trash can, and chair removed; a bucket, cart, and safety sign
added). Depth, RGB, camera pose, and "boxer" pre-baked 2D detections are
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

from semantic_mapping.geometry_utils import rotation_matrix_to_quaternion  # noqa: E402

WORLD_UP = np.array([0.0, 0.0, 1.0])


@dataclass
class SceneObject:
    label: str
    center_xy: tuple[float, float]
    half_extents: tuple[float, float, float]
    z_center: float
    color: tuple[int, int, int]
    appear_frac: float
    """Fraction of the sequence (0-1) at which this object becomes present."""

    disappear_frac: float
    """Fraction of the sequence (0-1) at which this object is removed (1.0 = never)."""

    def bbox3d(self) -> np.ndarray:
        cx, cy = self.center_xy
        hx, hy, hz = self.half_extents
        return np.array([cx - hx, cy - hy, self.z_center - hz, cx + hx, cy + hy, self.z_center + hz])

    def is_present(self, frac: float) -> bool:
        return self.appear_frac <= frac < self.disappear_frac


# Mirrors the paper's qualitative long-horizon change scenario (Sec. V-C):
# three static fixtures, three objects removed partway through, three added.
SCENE_OBJECTS: list[SceneObject] = [
    SceneObject("table", (0.0, 1.6), (0.6, 0.4, 0.4), 0.4, (150, 110, 70), 0.0, 1.0),
    SceneObject("sofa", (-2.2, -1.0), (0.9, 0.5, 0.4), 0.4, (90, 90, 160), 0.0, 1.0),
    SceneObject("shelf", (2.4, -1.6), (0.4, 0.3, 1.0), 1.0, (120, 120, 120), 0.0, 1.0),
    # Removed partway through:
    SceneObject("plant", (1.6, 1.4), (0.25, 0.25, 0.5), 0.5, (60, 150, 70), 0.0, 0.55),
    SceneObject("trash can", (-1.4, 1.7), (0.2, 0.2, 0.35), 0.35, (80, 80, 80), 0.0, 0.55),
    SceneObject("chair", (0.6, -1.8), (0.25, 0.25, 0.45), 0.45, (170, 130, 90), 0.0, 0.55),
    # Newly introduced partway through:
    SceneObject("bucket", (0.7, 0.3), (0.2, 0.2, 0.25), 0.25, (200, 60, 60), 0.45, 1.0),
    SceneObject("cart", (-2.0, 0.4), (0.35, 0.5, 0.5), 0.5, (60, 60, 200), 0.45, 1.0),
    SceneObject("safety sign", (2.0, 0.6), (0.05, 0.3, 0.5), 0.9, (230, 200, 40), 0.45, 1.0),
]

# Backdrop "room shell" the camera ray-casts against where it doesn't hit an
# object; must comfortably contain the whole camera orbit (radius=3.4 below)
# on every axis, or the shell's near face gets hit instead of its far face.
ROOM_BOUNDS = np.array([-5.5, -4.5, 0.0, 5.5, 4.5, 3.0])  # xmin,ymin,zmin,xmax,ymax,zmax

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
    present_objects: list[SceneObject],
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
        dist = _ray_box_hit(position, dirs_world, obj.bbox3d())
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


def make_detections(
    present_objects: list[SceneObject],
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

    visible_frames: dict[str, list[int]] = {o.label: [] for o in SCENE_OBJECTS}
    present_frames: dict[str, list[int]] = {o.label: [] for o in SCENE_OBJECTS}

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

        present_objects = [o for o in SCENE_OBJECTS if o.is_present(frac)]
        depth, rgb, object_id_map = render_frame(
            position, R_world_from_cam, present_objects, args.width, args.height, fx, fy, cx, cy,
        )

        frame_prefix = frames_dir / f"{frame_id:06d}"
        np.save(f"{frame_prefix}_depth.npy", depth.astype(np.float32))
        np.save(f"{frame_prefix}_rgb.npy", rgb)

        quaternion = rotation_matrix_to_quaternion(R_world_from_cam)
        pose = {"stamp": stamp, "translation": position.tolist(), "quaternion": quaternion.tolist()}
        (frames_dir / f"{frame_id:06d}_pose.json").write_text(json.dumps(pose))

        detections = make_detections(
            present_objects, object_id_map, rng, detections_dir, frame_id, with_masks=not args.no_masks,
        )
        (detections_dir / f"{frame_id:06d}.json").write_text(json.dumps({"detections": detections}))

        for idx, obj in enumerate(present_objects):
            present_frames[obj.label].append(frame_id)
            if int((object_id_map == idx).sum()) >= MIN_VISIBLE_PIXELS:
                visible_frames[obj.label].append(frame_id)

    scene_meta = {
        "num_frames": args.num_frames,
        "fps": args.fps,
        "objects": [
            {
                "label": o.label,
                "bbox3d": o.bbox3d().tolist(),
                "appear_frac": o.appear_frac,
                "disappear_frac": o.disappear_frac,
                # Presence is a single contiguous interval by construction.
                "appear_frame": present_frames[o.label][0] if present_frames[o.label] else args.num_frames,
                "disappear_frame": present_frames[o.label][-1] + 1 if present_frames[o.label] else args.num_frames,
                "visible_frames": visible_frames[o.label],
            }
            for o in SCENE_OBJECTS
        ],
    }
    (out_dir / "scene_ground_truth.json").write_text(json.dumps(scene_meta, indent=2))

    print(f"Wrote {args.num_frames} synthetic frames to {out_dir}/")
    print("This is a deterministic offline substitute for a real capture -- "
          "point --data_dir at a real RGB-D/LiDAR sequence for actual deployment.")


if __name__ == "__main__":
    main()
