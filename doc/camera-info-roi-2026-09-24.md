# CameraInfo crop and binning support

SuperMap previously used CameraInfo's full calibration dimensions and principal
point for cropped images. The live decoder stretched the image to those
dimensions, while aligned cropped depth failed the size check. This affected
the calibrated Ouster pinhole panels when an ROI was selected.

The shared adapter now converts the full-resolution K into emitted-image
coordinates:

```
fx' = fx / binning_x       cx' = (cx - roi.x_offset) / binning_x
fy' = fy / binning_y       cy' = (cy - roi.y_offset) / binning_y
width' = roi.width // binning_x
height' = roi.height // binning_y
```

Zero binning means one; an all-zero ROI means the full image. The incoming
CameraInfo is not modified. Invalid calibration/ROI or an image-size mismatch
is rejected rather than repaired by resizing. Live input can recover on the
next valid frame. The bag converter uses the same intrinsics conversion and
counts mismatched RGB frames as skipped.

## Verification

- Full SuperMap suite: **261 passed in 32.02 seconds**, including ROS tests.
- Package build succeeded with `colcon build --packages-select semantic_mapping
  --symlink-install`.
- Synthetic cropped/binned RGB-D and cloud inputs preserve original pixels and
  their expected world coordinates, including compressed RGB transport.
- Cropped/binned bag conversion reproduces the original images, intrinsics,
  depth, poses and semantic evaluation.
- Actual `os_pinhole` from the local ouster-ros fork (`c27e8510`) decoded the
  `07052026_4_an` bag in isolated ROS domain 69. All four cardinal views passed
  for both default automatic cropping and an explicit ROI: 20 synchronized
  CameraInfo/reflectivity/depth triples per view, **160 triples total**.

| Configuration | Full calibration | Emitted image | ROI origin | Effective principal point |
| --- | --- | --- | --- | --- |
| Automatic | 256 × 141 | 256 × 141 | (0, 0) | (127.5, 70) |
| Explicit ROI | 256 × 141 | 192 × 48 | (32, 8) | (95.5, 62) |

The principal point may legitimately lie outside a crop. Each real message
pair had matching timestamps and optical frames. The live decoder preserved
the reflectivity pixels, and the corrected back-projection rays agreed with
their full-canvas counterparts. This validates crop handling, not the separate
nearest-neighbour approximation between pinhole pixels and raw LiDAR returns.

Artifacts are under workspace `results/supermap-camera-info/`: `tests.log`,
`build.log`, `probe_panels.py`, `panels.json`, captured panel arrays and driver
logs. The initial focused test expected unpadded boxes; it was corrected to
check actual mapped points independently of voxel padding before the full
passing run.

The probe skipped the expected partial startup revolution. After successful
capture, the cropped-panel driver emitted `Unexpected condition: string
capacity was zero for allocated data! Exiting.` during shutdown. That separate
race is now fixed in ouster-ros fork commit
[`8142673`](https://github.com/jcfurey/ouster-ros/commit/8142673): its standalone
executables join ROS's signal-handler shutdown before destroying their nodes. The driver
regression tests and repeated panel replay validate clean shutdown; diagnostic
artifacts are under workspace `results/ouster-shutdown/`.

## Scope

This preserves SuperMap's existing K-based pinhole model. It does not add image
undistortion, stereo rectification, panorama projection, monochrome detector
adaptation or fusion across cameras. The fork's pinhole K/P already describe
the generated rectified panel, so no new rectification is needed for these
messages. Use metric `depth_image`, not the radial `range_image`.
