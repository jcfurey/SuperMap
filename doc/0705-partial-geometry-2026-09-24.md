# Partial geometry and residual fragments

This follow-up starts from SuperMap `4e1137d` and workspace integration
`af0594f`. Both were committed before implementation began.

The prior full replay ended with an occluded person represented by 21 points
in a 0.14 × 0.81 × 0.16 m box. Two paths produced this behavior: a few
foreground foot returns could replace an entire body, and per-point pruning
could leave floor-contact geometry indefinitely occupied.

## Changes

- Optional image-span support requires a candidate depth layer to cover both
  dimensions of the visible silhouette before sparse depth filling. If a
  near layer contains only feet, another supported layer may be considered;
  no qualifying layer means unknown depth.
- An optional compact-object size limit rejects implausibly broad geometry,
  including continuously connected background returns. It never clips an
  invalid measurement into a plausible shape.
- Dynamic instances keep their last supported geometry when a detection has
  inadequate depth, and the geometry is marked occluded. `geometry_stamp`
  records the last accepted 3D observation independently from 2D detections.
  Save/load retains this field; old maps infer it from their latest observation.
  Live marker and VLM location ages use this timestamp. Query JSON exposes
  geometry age separately from the most recent 2D observation.
- Optional whole-body cleanup keeps the supported point set intact while
  updating occupancy/membership evidence. Sufficiently contradicted extent
  retires the instance rather than shrinking its box to a floor fragment.
  Unknown, occluded and out-of-view depth retain their prior evidence.
- Tracks without geometry use image tracking without an invented world-origin
  projection. They cannot become confirmed dynamic spatial objects without
  depth, and expired depthless tracks have no spatial scene-graph node.
- Merging dynamic duplicates selects the latest actual geometry, even when an
  older body has a more recent 2D-only detection.

The 0705 profile requires 50% of the visible mask span in each image axis,
rejects camera-axis extents above 3 m for `person`, and retires a contradicted
body when a measured axis retains less than 35% of its span. Axes smaller
than four voxel cells do not drive this retirement. These three thresholds
default to zero (disabled) in generic configuration. The existing foreground
and dynamic class lists limit the policies to the selected classes.

## Validation

The short reproduction starts playback at 480 seconds with a fresh deliriom
odometry session. It is not a continuation of the full run's odometry frame.
Its captured fixture contains 330 pre-fusion frames, including empty detection
results, and 51 person detections. Baseline and revised offline pipelines use
identical depth, camera pose and detector masks, with appearance embedding off.
Fixtures and results are under `results/supermap-quality/` in the workspace.

The baseline accepted 49 person depth updates, including six with less than
0.4 m optical height. The revised profile accepts four updates, none below
0.4 m; the rest are explicitly unknown 3D measurements. These close-up frames
often have inadequate common camera/LiDAR support or background-contaminated
layers, so fewer depth updates are intentional. The 2D detector still runs.

In the earlier, better-observed fixture, 161 of 166 person measurements remain
accepted. All have optical height above 0.4 m. This supports rejecting the
failure cases without broadly disabling person mapping, but it is not a
labelled accuracy or recall evaluation.

All 244 tests passed in 26.27 seconds, including ROS integration. New regressions cover spatial
support before filling, implausible extents, preservation of a body during
partial observations, unknown/occluded/out-of-view evidence, retirement despite
dense floor returns, depthless track continuity, duplicate geometry age,
geometry timestamp persistence, old-map loading, and location age during
newer 2D detections. Both workspace packages build successfully.

The revised live replay processed 682 frames over the final 72.61 seconds of
the bag. Excluding the initialization window, six 10-second runtime windows
averaged 9.97 Hz mapping and 9.92 Hz detection. Average detector execution was
29.45 ms; pipeline window averages ranged from 12.1 to 36.2 ms (the higher
window includes the close-up masks). These are short-run measurements, not a
new full-duration slowdown test.

A subscriber observed the final 41.3 seconds: 412 map/marker publications,
408 annotated images, and no published person box below 0.4 m world height.
The saved map retains one occluded person with 211 points and a
1.72 × 1.65 × 1.73 m world-axis box, last measured roughly 36 seconds before
playback ended. This verifies that its geometry no longer collapses to feet;
it does not establish body-boundary accuracy or continued physical presence.
The short replay still reports weak-geometry and duplicate/backward-IMU
warnings from deliriom, plus startup and occasional TF skips. It completes
with no error log entries. Original full-run artifacts remain untouched.

Outputs: `results/supermap-quality/fixed/semantic-map/`, `fixed.log`,
`probe_live.json`, `summary.json`, and the captured fixture evaluations.
The live run preceded the final location-age display change; that change is
covered by ROS marker, JSON serialization and VLM prompt regression tests.

## Limits

The size and coverage limits are deployment policies for compact objects.
Severe occlusion, limited LiDAR vertical coverage, image truncation, or poor
calibration may leave only a 2D track. Occluded geometry remains a last-known
location, not a motion prediction. This pass does not add camera-rate inference,
full dynamic 3D tracking, measured extrinsic calibration, or pose-graph updates
to previously mapped semantic objects.
