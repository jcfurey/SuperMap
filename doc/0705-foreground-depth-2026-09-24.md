# Sparse LiDAR foreground depth and moving geometry

The `07052026_4_an` Ouster/Lucid replay exposed two causes of growing person
boxes. A correct segmentation mask can contain LiDAR returns through or
beside a silhouette, including tunnel surfaces 15–25 m behind the person.
Unioning successive person point sets also retains earlier body positions
as the person walks.

## Opt-in policies

`foreground_depth_gap_m` splits sorted raw masked depths at discontinuities
and selects the nearest supported layer. The replay uses a 0.75 m gap,
at least five real depth pixels, and at least 10% of valid masked pixels.
Support is counted before sparse depth filling or subsampling. A lone
foreground return therefore cannot create its own supported layer by filling
neighbouring pixels. If no layer qualifies, there is no new 3D geometry.
`foreground_depth_labels` defaults to `[person]`; other classes can retain
long structures. A zero gap disables selection in the generic configuration.

`dynamic_geometry_enabled` replaces matched geometry for
`dynamic_geometry_labels` (default `[person]`) when new supported points arrive.
It preserves the instance ID, label belief, appearance, and trajectory.
Missing depth retains the last supported geometry. Duplicate merging keeps
one recent geometry rather than unioning earlier body positions. This policy
is disabled by default, so static object fusion retains its usual behavior.

The workspace's `supermap_bringup` Ouster/Lucid profile enables both policies.
Its camera TF was also corrected: the legacy calibration matrix maps LiDAR
points into the optical frame, despite its `cam2lidar` label. TF needs the
inverse rotation **and** inverse translation. Five image/cloud overlays
support this direction; neither the July bag nor the same-platform
`06042026_` reference bag records the missing Ouster-to-camera-rig bridge.

## Validation

Exact pre-fusion RGB, depth, camera pose and detector masks were captured from
the replay. The local fixture contains 166 person detections over 124 sampled
frames, roughly two frames per second through the failing segment. Both
pipeline runs used identical captured inputs, 0.2–40 m depth limits,
two-pixel depth filling, and 2nd–98th percentile box bounds. Appearance
embedding was disabled in this offline comparison.

| Measurement | Before | Foreground selection |
| --- | ---: | ---: |
| Median per-detection depth span, 2nd–98th percentiles | 4.82 m | 1.31 m |
| 95th percentile depth span | 24.16 m | 2.56 m |
| Maximum depth span | 32.05 m | 3.38 m |

With foreground selection and dynamic geometry enabled, the main active
person box at the end of the sampled sequence measured approximately
1.77 × 0.83 × 1.38 m, versus 33.88 × 10.56 × 9.28 m before. This is a
regression comparison, not ground-truth person-size accuracy. The sampling
also differs from the full 10 Hz replay.

The restarted live replay sustained approximately 10 Hz for detection,
mapping and publication. Through bag offset 132 seconds, 675 active person
marker samples had median world-axis box dimensions 1.62 × 0.61 × 1.33 m.
The per-axis 95th percentiles were 4.15 × 1.18 × 1.53 m; the largest length
was 5.88 m. Thus the extreme background extrusion is substantially reduced,
but occasional broad foreground layers and partial geometry remain. The
first observed person marker was about 1.6 degrees off the camera optical
axis, consistent with its image location after the TF direction correction.

All 227 tests passed, including ROS integration, on ROS 2 Lyrical/Python 3.14.
New regressions cover majority-background masks, isolated foreground noise,
insufficient sensor support, filling order, unaffected static classes,
moving-object identity/history, missing depth, and duplicate geometry.
The workspace packages `semantic_mapping` and `supermap_bringup` build.

Local inputs, comparison output and test logs are under
`results/supermap-0705/foreground-review/` in the containing workspace.

## Completed replay review

The 552.61-second replay finished cleanly with 5,475 semantic frames fused
and no logged errors. Excluding the first two ten-second startup windows,
mean detection was 9.98 Hz and mean mapping/publication was 9.98 Hz.
Comparing ten early steady windows against the final ten windows, mean
mapping pipeline cost was 16.50 versus 16.74 ms; inference was 29.44 versus
29.56 ms. This run showed no sustained slowdown. Detection remains paced by
the 10 Hz cloud synchronizer, below the 20 Hz camera input.

The final semantic save reloads successfully and contains nine historical
records: seven disappeared and two occluded. The occluded records contain
36 rock points and 21 person points. That person's remaining box is only
0.14 × 0.81 × 0.16 m, so compact-object cleanup and partial-depth support
remain important next work. These records are not nine current detections.
The one-hit backpack record was retired, but previously had a 13.44 m
extent; foreground filtering currently targets only the person class.

DELIRIOM's sealed final archive contains 584 selected mapping frames across
16 submaps, with 14,973,451 stored point samples and no reported source
sequence gaps. Point samples include repeated observations of surfaces.
The archive reloads and its keyframe point checksums validate. No loop
constraints or optimized pose revisions were applied. The selected-pose
trajectory travels about 115.15 m and ends 0.52 m from its first pose; this
is an endpoint separation, not an accuracy estimate without ground truth.
Tunnel degeneracy and duplicate/backward IMU timestamp warnings occurred.

Saved artifacts in the containing workspace:

- `results/supermap-0705/semantic-map/`: final semantic state.
- `results/supermap-0705/deliriom/final-0705.deliriommap`: sealed geometric archive.
- `results/supermap-0705/progress-review/`: measured summary, plots and archive inspection.

## Limits

This depth-layer heuristic is for compact foreground objects. A large depth
gap can split a genuinely deep object, while a continuous chain of returns
can join surfaces. Sparse returns, mask edges, and residual camera calibration
error still cause incomplete or jittering bounds. Refreshing moving geometry
trades accumulated surface coverage for a current location. The latest
observation is not a full dynamic 3D motion model; uncertain association can
still split an identity. DELIRIOM remains the odometry and geometric mapper.
