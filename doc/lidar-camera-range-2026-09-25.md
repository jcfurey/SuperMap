# Camera masks on LiDAR depth: boxes at ring gaps and moving labels

Reported behavior: object boxes sat at the start of gaps in the point cloud,
and the semantic cloud moved when the scene did not. Both came from how a
camera mask's distance was taken from the LiDAR. The object tracker and the
dense labeller each took it from whichever returns fell inside the 2D mask.
Neither checked, in 3D, that those returns belonged to the object.

## Causes

1. **Ground rings inside the mask.** A segmentation mask bleeds a few pixels
   onto the ground at an object's base. At 20 m, with the camera at 1.2 m,
   one image row spans about 0.7 m of ground. A 6 px bleed therefore reaches
   the LiDAR ring 2–3 m in front of the object. That ring is nearer than the
   object:
   - With `foreground_depth_gap_m` (nearest supported layer), the ring is the
     chosen layer. The whole box sits on the ring, just before the gap to the
     next ring.
   - Without it, the box stretches from the ring to the object.
   - The dense labeller's depth-relative layer gap (5 m at 20 m) keeps ring and
     object in one layer, so the ring takes the label.

   A spinning LiDAR's rings sit at fixed ranges from the sensor. As the
   vehicle drives, the ring slides with it, and so do the box (dynamic
   geometry replaces it each frame) and the labels. In scan mode the labels
   smear along the ground in front of the object.
2. **One-pixel z-buffer in the object tracker.** `node.py` rasterized the cloud
   with a per-pixel minimum. A sparse foreground leaves holes between its
   returns, and background behind it filled them. A LiDAR mounted above the
   camera also reaches surfaces the camera cannot see, e.g. a wall behind a
   barrier. Those returns were lifted into masks: a 10 m barrier's box
   reached the 30 m wall. They also looked like free space in front of mapped
   points, which the consistency update then pruned. The dense labeller already
   used a footprint z-buffer (review C5); the tracker did not.
3. **Half-pixel bias in the dense labeller.** `annotate()` truncated projected
   coordinates (`floor`). CameraInfo, and the rest of the code, put pixel
   centres at integer coordinates (`round`). Every label shifted half a pixel
   down and right: at 20 m, about 0.37 m further onto the ground in front of
   an object.
4. **Labels published on the wrong cloud.** The dense node's publish timer read
   the latest labels and the latest cloud without a common lock. It could
   publish one scan's labels on the next scan's bytes. An organized (Ouster)
   cloud of equal size raises no error: labels jumped for that message.

## Changes

- `geometry_utils.occlusion_visible` / `rasterize_depth(splat_radius_m=...)`:
  - This is the dense path's footprint z-buffer, now shared by both paths.
  - The live node (`pointcloud_splat_radius_m: 0.05`, `pointcloud_splat_max_px`,
    `pointcloud_occlusion_gap_m`, `pointcloud_occlusion_grid_px`) and
    `rosbag_to_sequence.py` use it by default.
  - The image still holds one real reading per pixel.
  - Points on a densely sampled surface (a neighbour reading at similar depth
    both vertically and horizontally) are exempt. The one-pixel z-buffer is
    already exact there. Without the exemption, a depth camera's cloud lost
    7–11% of its readings in bands beside near silhouettes (the synthetic
    scene); with it, 99%+ remain. A spinning LiDAR's rings are several
    pixels apart, so its returns never qualify.
  - Auto cells keep it exact up to VGA. Measured added cost: 14.5 ms at 640×480,
    20 ms at 1440×1080, 22 ms at 2448×2048.
- `ground_exclusion` (tracker and dense labeller, on by default):
  - A local ground plane is fitted in world z around the mask.
  - Masked readings less than the clearance (0.15 m) above it are dropped when
    enough readings stand above it (tracker: `ground_exclusion_min_returns: 3`;
    dense: `mask_min_layer_points`).
  - Masks keep every reading in three cases: `ground_surface_labels` (floor,
    road, rug, ...), masks lying flat on the ground, and masks with no ground
    fitted below the camera. The last case means an optical or other
    non-z-up "world" frame leaves masks untouched.
  - The existing `ground_removal_labels` remains the unconditional variant.
  - The plane fit now groups cells with a packed key instead of a row-wise
    `np.unique`, and fits at most 2048 readings.
- Dense labeller: pixel centres at integer coordinates.
- Dense node: labels and their cloud are stored and published as one pair.

## Reproduction

The analysis rig: a spinning 64-beam, 45° LiDAR 0.4 m above and 0.1 m behind a
640×480 camera (f = 450 px). It drives 2 m toward an object in front of a wall.
The camera mask is either exact or bled 6 px downward. Values are the
reported box's x-range (true range in brackets), sampled along the drive.

| Scene, mask | Setting | Before | After |
|---|---|---|---|
| bag 20 m [20.0, 21.0], bled | all returns | 16.9–20.0 … 16.9–21.0 | 20.0–20.0 … 20.0–21.0 |
| bag 20 m, bled | nearest layer, gap 0.75 m | 16.9–17.0 → 16.9–19.0 (on rings) | 20.0–20.0 … 20.0–20.8 |
| bag 20 m, bled | nearest layer + dynamic geometry | 17.0 → 18.8, moving with the vehicle | on the bag |
| barrier 10 m [10.0, 10.6], exact | all returns | 10.0–30.0 (wall through the barrier) | 10.0–10.6 |
| barrier 10 m, bled | all returns | 9.0–30.0 | 10.0–10.6 |
| person 12 m [12.0, 12.3], bled | nearest layer, gap 0.75 m | 11.3–12.0 | 12.0–12.0 |

The LiDAR sees the front face only, hence partial depth extents.

Dense labeller, scan mode, same drives (points labelled off the object):

| Scene, mask | Before | After |
|---|---|---|
| bag 20 m, bled | 220 ground points, 17.0–19.9 m | 0 |
| barrier 10 m, bled | 382 ground points, 9.0–9.9 m | 0 |
| person 12 m, bled | 127 ground and wall points, 10.9–25.0 m | 0 |

`test/test_lidar_camera_range.py` builds the same rig at 320×240. Its tests
cover:

- the old failure (a box on the ring that moves with the sensor);
- boxes that stay on the object across the drive;
- wall see-through hidden by the footprint test, with every barrier reading
  kept;
- dense labels that no longer smear along the ground;
- ground classes and flat objects that keep their returns;
- a non-z-up world frame, which disables the exclusion;
- the pixel convention.

## Validation

- 403 unit tests pass (ROS tests not run here: no ROS in this environment).
- `flake8` with the repository configuration is clean.
- Synthetic scene, CI metric gate (dense depth, generic defaults):

| Metric | Before | After | Gate |
|---|---:|---:|---|
| Mean detection recall | 0.976 | 0.971 | ≥ 0.90 |
| Mean change recall | 0.957 | 0.954 | ≥ 0.90 |
| Final-map F1 | 0.889 | 0.941 | ≥ 0.80 |
| Instance IDs | 12 | 12 | ≤ 13 |
| Identities kept | 2 | 2 | 2 |
| mIoU without background | 1.000 | 0.987 | ≥ 0.95 |
| mAP50 | 1.000 | 1.000 | ≥ 0.95 |

The mIoU loss is the lowest 0.15 m of box, bucket and table, now excluded as
ground. Pipeline time on that scene rose from 10.9 to about 17 ms per frame
(ground fit on dense depth, about 1 ms per masked detection). Set
`ground_exclusion: false` for indoor RGB-D if either matters.

These are simulation and regression results. They were not measured on the
Ouster/Lucid replays.

## Limits and checks on the real rig

- A physical footprint covers the gap between returns only while
  `radius / range` exceeds half the LiDAR's angular spacing. For a 64-beam
  unit (0.7° vertical), 5 cm stops closing ring gaps beyond about 8 m. A
  person at 12 m kept 7 of 14 wall readings at its top edge in the rig; 10 cm
  removed them but hid some genuinely visible surface. Classes with
  foreground layer selection are unaffected. For others, raise
  `pointcloud_splat_radius_m` or add them to `foreground_depth_labels`.
- Ground exclusion needs a z-up world frame and some ground in or around the
  mask. Sloped ground steeper than `ground_max_slope` falls back to "no
  ground" and keeps every reading.
- Moving labels have other possible causes outside this code. If TF puts the
  cloud or the image at the wrong time, the projection shifts with vehicle
  speed:
  - a deskewed scan whose header stamp is not its deskew reference time;
  - a camera stamp offset;
  - an inverted extrinsic (see the 0705 note).

  A stationary check should show no drift. A drift that grows with speed
  points at timing.
