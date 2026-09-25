# Full dense clouds with partial camera coverage

`DenseCloudPipeline` segments geometry over the complete incoming cloud and optionally adds camera labels to visible surfaces. Camera coverage never determines which points exist in the map. Both accumulated clouds and individual registered scans are supported. The geometry core runs on CPU, loads no pretrained models, and makes no network requests. Camera annotations accept manual masks by default. An explicitly enabled **YOLOE exception** adds automatic camera labels through a separate node; it does not qualify YOLOE as a US-created model.

The tunnel capture is **Ouster LiDAR plus a Lucid camera**. Tunnel geometry comes from the Ouster cloud; Lucid supplies optional camera observations. The RealSense D435i data used for the indoor checks are a separate capture. A depth camera can supply its full XYZ cloud through the same interface.

## Geometry and coverage

The pipeline assigns stable IDs to connected geometric regions using a voxelized, bounded-neighbor graph with local surface-normal and point-to-plane checks. Voxel representatives are measured points. Labels are expanded back to every original input point, in its original order. Invalid rows remain present with `valid_geometry=0`, `region_id=-1`, and unknown semantics. Small regions below `min_region_voxels` also receive `region_id=-1`; their points are retained.

| Input mode | Meaning | Map behavior |
|---|---|---|
| `snapshot` | Each message is an authoritative accumulated map or submap | Replace geometry with that message; retain label evidence and region identities where world voxels overlap |
| `scan` | Each message is a registered scan | Union world voxels; preserve geometry and labels absent from the current scan |

Input clouds may be in the world frame or a sensor frame with a valid timestamped TF. Region IDs persist by voxel overlap, with one ID surviving a merge or split. They identify geometric surfaces, **not recognized or tracked object instances**. Mixed class evidence is reported as per-region label counts rather than assigning one class to an entire region.

The default keeps unobserved geometry semantically unknown. An optional positive `label_propagation_radius` extends labels only to nearby voxels in the same geometric region, marks them as propagated, and never uses propagated labels as new seeds. This is a bounded inference that can still cross a boundary if geometry undersegments touching objects; it is disabled by default.

## ROS 2

After building this package in the intended deployment workspace:

```bash
# Full accumulated map/submap
ros2 launch semantic_mapping dense_cloud.launch.py \
  input_mode:=snapshot cloud_topic:=/your/dense_map world_frame:=map

# Registered scans, accumulated inside this node
ros2 launch semantic_mapping dense_cloud.launch.py \
  input_mode:=scan cloud_topic:=/your/registered_scan world_frame:=odom
```

The launch file accepts `config:=/path/to/dense_cloud.yaml`. Its `cloud_topic`, `world_frame`, and `input_mode` launch arguments override those YAML values. For a source-tree run with ROS already sourced, without rebuilding an active workspace:

```bash
cd src/SuperMap
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -m semantic_mapping.dense_cloud_node --ros-args \
  --params-file config/dense_cloud.yaml \
  -p input_mode:=snapshot -p cloud_topic:=/your/dense_map -p world_frame:=map
```

No RGB, CameraInfo, camera annotation, or detector result is required before cloud processing. Optional camera messages are handled independently. No detector, VLM, learned appearance encoder, or default model configuration from `semantic_mapping_node` is loaded by this entry point.

| Output | Content |
|---|---|
| `/supermap/dense_cloud` | Every point in the latest input message, preserving its frame, original fields, point bytes, organization and ordering; segmentation fields are appended |
| `/supermap/voxel_map` | Persistent map at the configured voxel resolution, in `world_frame`; this is separate from the complete dense input output |
| `/supermap/regions` | JSON label-ID vocabulary, region bounds and class counts, input/coverage statistics, and rejected-observation counters |

Appended fields are `region_id` (int32), `semantic_id` (int32), `semantic_confidence` (float32), `label_source` (uint8), `camera_visible` (uint8), and `valid_geometry` (uint8). Semantic ID 0 means unknown. Source 0 means unknown, 1 means a voxel has direct camera evidence, and 2 means bounded propagation. Confidence records annotation confidence, optionally reduced by propagation distance; it is not a calibrated recognition probability.

Original point padding is preserved; inter-row padding is repacked. Reflectivity, intensity, ring, RGB, and any other original fields survive unchanged. The separate voxel map contains XYZ and segmentation fields. In RViz, display the dense output and choose the intensity color transformer with `region_id` or `semantic_id` to inspect segmentation independently of any input RGB field.

## Optional camera labels

Publish a `std_msgs/String` JSON object to `/supermap/camera_annotations`. Use the **image capture timestamp**, calibrated rectified optical frame, and intrinsics for the exact mask resolution. The ROS node resolves world-from-camera TF at that timestamp; it does not accept a transform override from the message. Example schema:

```json
{
  "source": "manual",
  "rectified": true,
  "stamp": 123.5,
  "frame_id": "lucid_camera_1/optical_frame",
  "intrinsics": {
    "fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0,
    "width": 640, "height": 480
  },
  "detections": [
    {"label": "pipe", "score": 1.0, "mask_runs": [[153900, 20], [154540, 20]]}
  ]
}
```

The numbers above illustrate the schema and are **not Lucid calibration values**. A detection supplies either `mask_indices` (flattened row-major foreground pixel indices) or `mask_runs` (`[start, length]` foreground spans; not COCO RLE). Boxes alone are rejected. Optional `depth` is an image-sized array in metres on the same rectified grid; zero, negative, and nonfinite depth provide no visibility evidence.

Alternatively, configure `camera_info_topic` and omit `intrinsics`. CameraInfo must match the optical frame and capture time (stamp 0 is treated as static calibration). ROI and binning are accounted for. Distortion coefficients must be zero and any rectification rotation must be identity; otherwise supply the correct rectified optical TF and explicit pinhole intrinsics. Raw distorted-image masks are unsupported.

Projection uses the full latest dense cloud to build a nearest-surface depth buffer. Pixel centres are integer coordinates, as for CameraInfo; earlier versions truncated projections, shifting labels half a pixel down and right, onto the ground below an object's base. Only points inside the image, in front of the camera, and within `camera_depth_tolerance` of the nearest measured surface can receive labels. With `ground_exclusion` (default), points less than `ground_clearance` above the local ground (world z up, fitted to visible points in and `ground_context_px` around the mask) do not take a mask's label when at least `mask_min_layer_points` of its points stand above the ground: a mask bled onto the ground no longer paints the LiDAR ring in front of an object, which slid along with the sensor. `ground_surface_labels`, flat objects, and masks without a ground fitted below the camera keep their ground points. When depth is supplied, it must agree as well. Conflicting classes at one pixel or voxel are left unresolved for that observation. Existing positive evidence is not negated by an empty annotation or camera absence. Multiple camera observations can contribute positive labels to different visible portions of a cloud.

`max_camera_time_delta` bounds the difference between image capture and the latest cloud timestamp. A delayed result for an older cloud is rejected once it falls outside that window. Results that arrive before their matching cloud are held in a bounded `camera_annotation_queue_size` queue, then applied only to a matching cloud. Queue overflow and expired results increment `expired_annotations`; malformed or unusable matching results increment `rejected_annotations`. The queue holds compressed masks and resolves historical TF when a matching cloud is processed. At most the closest YOLOE observation per optical frame is applied to each cloud; manual observations can contribute separately. Historical clouds are not replayed for late inference. A non-world-frame observation with stamp 0 is rejected instead of requesting the latest dynamic TF. TF failure, invalid calibration, unapproved annotation source, and stale masks do not remove geometry.

## Explicit YOLOE labeling exception

[Ultralytics documents YOLOE](https://docs.ultralytics.com/models/yoloe/) as an image detection and segmentation model and credits its original Tsinghua authors. Enable this adapter only when that origin is an accepted exception. Other model providers remain rejected by the dense-cloud path.

Set `allow_yoloe_labels: true` in the cloud node configuration.

To preserve the existing tracker, object identities, 3D boxes and camera overlay, use the existing `semantic_mapping_node` with:

```yaml
publish_dense_camera_annotations: true
dense_camera_images_rectified: true
dense_camera_annotations_topic: /supermap/camera_annotations
detector: yoloe
yoloe:
  checkpoint: /absolute/path/to/yoloe-v8l-seg.pt
  text_encoder_path: /absolute/path/to/mobileclip_blt.ts
```

This shares the already running detector's masks with dense fusion; it does not start a second model. Keep its pointcloud input on the live deskewed Ouster topic and its RGB/CameraInfo inputs on the rectified Lucid stream. The tracker retains its normal `/obj_points`, `/obj_boxes` and annotated-image outputs, including existing foreground filtering and dynamic object handling. Dense geometry runs separately and does not pace the tracker. Model export requires local assets, the YOLOE backend without an extra refinement model, and explicit rectified-image configuration. Masks carry the original image timestamp.

For an independent camera-only label producer when object tracking is not needed, copy `config/dense_yoloe.yaml` to a deployment configuration and set:

- `allow_yoloe: true` and `images_rectified: true`.
- Absolute paths to an existing YOLOE segmentation `checkpoint` and matching local MobileCLIP TorchScript `text_encoder_path` (the tunnel deployment uses YOLOE-v8l and `mobileclip_blt.ts`). Missing files fail before model construction; no download is requested by this adapter.
- Rectified `rgb_topic` and matching `camera_info_topic`. Camera inference needs no cloud or accumulated-map subscription.
- The desired `prompts`. Keep `confidence_threshold` and the cloud node's `min_camera_score` consistent.

Run the adapter in an environment that already contains Ultralytics, Torch, OpenCV, the CLIP tokenizer and ROS:

```bash
ros2 run semantic_mapping dense_yoloe_labels_node --ros-args \
  --params-file /path/to/deployment-dense-yoloe.yaml
# Or, from this repository with ROS sourced and its Python dependencies available:
python3 -m semantic_mapping.dense_yoloe_node --ros-args \
  --params-file /path/to/deployment-dense-yoloe.yaml
```

The independent camera adapter runs directly on synchronized RGB/CameraInfo pairs, up to `max_rate_hz` (20 Hz by default). Neither live clouds nor accumulated maps gate image inference or the annotated camera display. The image capture timestamp and optical frame accompany every mask. CameraInfo must describe the actual rectified image dimensions with zero distortion and identity rectification rotation.

For live Ouster processing, feed the dense node the complete deskewed scan, e.g. `/deliriom/odom_node/pointcloud/deskewed`, in `odom`. Use `input_mode: snapshot` for independent complete live scans, or `scan` to retain a world-voxel union (whose segmentation cost grows with the union). To keep the live path responsive while also labeling an accumulated map, run a second dense node with the map as its snapshot input and distinct output topics. Both consume the same live camera annotations. Delayed map snapshots choose matching historical masks from a bounded queue (512 by default), with 30 seconds of TF history. They never hold up the camera or live-cloud process. Size the queue for the observed map delay and inference rate.

The tunnel replay uses `/supermap/dense_cloud` for the live Ouster output, `/supermap/live_regions` for its label vocabulary, and `/supermap/dense_map` plus `/supermap/regions` for the separate accumulated-map output. Its upstream mapper publishes a resident submap window and archives complete observations on disk; this should not be confused with publishing the entire trajectory's raw returns in each message.

Output `/supermap/camera_annotations` contains `source: "yoloe"`, mask spans, class names, confidence, and checkpoint/text-encoder SHA-256 hashes. `/supermap/annotated_image` displays the camera masks and names. The cloud node appends semantic IDs to **every original cloud point**, preserves unseen geometry, and reports the class vocabulary, `annotation_sources`, applied/rejected annotation counts and model metadata on `/supermap/regions`. Its `pretrained_models` list stays empty because model inference runs in the separate adapter. In RViz choose `semantic_id` for the dense cloud's intensity channel, and display the annotated image for class names.

This is camera-derived labeling of visible cloud surfaces, not learned point-cloud recognition. Unseen geometry remains unknown unless it retains previous direct evidence or bounded propagation is explicitly enabled. Camera labels do not turn geometric region IDs into tracked object instances. Sparse geometry can miss camera occluders, and accumulated maps can retain moving-object trails; the new dense path does not inherit the separate object tracker's dynamic-object filtering.

## Offline use

Only NumPy, SciPy, and PyYAML are needed for this path. Do not install a detector stack to use it.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 examples/dense_cloud.py /path/to/full_cloud.npz \
  --input-mode snapshot --out_dir /path/to/new_output

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 examples/dense_cloud.py /path/to/scan_00.npz /path/to/scan_01.npz \
  --input-mode scan --out_dir /path/to/new_scan_output
```

NPZ files contain `xyz` (Nx3 or HxWx3) and optional scalar `stamp` and rigid `T_world_from_cloud` (4x4). NPY files contain XYZ and are assigned sequential timestamps. If no transform is supplied, coordinates are assumed to already be in the same world frame. Never accumulate unrelated sensor-frame scans without their poses. Inputs are processed in the given order with nondecreasing timestamps. `--xyz-key`, `--config`, and `--voxel-size` are available.

`--annotations manual.json` accepts a list of the camera objects above, each with `cloud` equal to an input filename and a `T_world_from_camera` matrix (offline only). Each result NPZ contains complete input `xyz`, validity, region/semantic IDs, confidence, source and visibility arrays, plus voxel-map arrays. Sidecar JSON contains the vocabulary and region metadata; `run.json` records configuration, source hashes and CPU timing. Output directories and files must be new.

## Practical limits and verification

The core performs geometric segmentation and optional semantic fusion. Automatic class names require the explicitly enabled YOLOE camera adapter; the core itself runs no learned segmenter and makes no semantic object-instance accuracy claim. The existing image-driven object tracker and scene-graph path can run alongside dense fusion and share inference; surface regions are not silently inserted as recognized object instances.

Normal and distance thresholds depend on sensor noise, density and scene scale. A surface can fragment, and touching objects can merge. Point labels inherit voxel-level evidence, so boundaries are limited by voxel resolution. Cloud-derived visibility cannot account for an occluder missing from the cloud; independent synchronized depth improves that check.

Scan mode is a static accumulation path, with no free-space carving, moving-object removal, or correction of historical poses after loop closure. Use authoritative map snapshots for an upstream SLAM map that revises geometry. Changed world frames or clock rewinds require a new pipeline. Label evidence persists by exact world-voxel overlap; `label_ttl_sec` optionally expires labels without deleting geometry. Reused spatial locations can otherwise retain old labels until new evidence or a snapshot removes them.

Memory scales with the dense incoming cloud and the voxel graph. Neighbor searches use a bounded count and chunks, not an NxN distance matrix, on `kdtree_workers` threads (default -1, every core; results are identical for any value, so lower it to leave cores for SLAM). On a 4-core Xeon a 1M-point snapshot (630k voxels) segments in about 8 s from scratch and 3 s when only a local part changed. `max_map_voxels` raises an explicit error and preserves the last map at capacity; it never truncates the input to fit. Dense snapshots are computationally heavier than scans, and geometry processing is serialized, while a separate callback group receives compressed masks concurrently so a heavy cloud cannot block camera-annotation reception. At an unsustainable input rate, ROS subscription queues may drop messages; deployment rate must be measured for the target cloud size. No real-time guarantee is made.

Validation includes complete point/field preservation, endian and row-padding handling, partial FOV and occlusion, depth evidence, stale observations, mask conflicts, scan/snapshot behavior, persistent region IDs, explicit capacity failures, and actual ROS messages. A saved-data check preserved all 1,863,653 points in an overlapping stationary D435i snapshot and all 65,536 rows of each of five raw Ouster scans from the Lucid/Ouster tunnel bag. Ouster scans were evaluated individually because those fixtures do not store registration poses. These checks establish data handling and geometry execution, not semantic accuracy or performance on a complete moving room scan.
