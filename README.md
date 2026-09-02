<p align="center">
  <h1 align="center"> SuperMap: A  Living Spatial Memory for Embodied AI </h1>
  <h3 align="center">RSS 2026 </h3>
  <p align="center">
  </p>
  <p align="center"><strong>AirLab, Carnegie Mellon University</strong><br/>
  <p align="center">
    <a href="doc/paper.pdf"><img src="https://img.shields.io/badge/Paper-RSS%202026-b31b1b.svg" alt="Paper"></a>
    <a href="https://superodometry.com/supermap"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page"></a>
    <a href="https://www.youtube.com/watch?v=TQjTTqEewNQ"><img src="https://img.shields.io/badge/Video-YouTube-red.svg" alt="Video"></a>
    <a href="https://discord.gg/Huf2GJx32y"><img src="https://img.shields.io/badge/Discord-Join%20Us-5865F2.svg?logo=discord&logoColor=white" alt="Discord"></a>
  </p>
</p>

<p align="center">
  <a href="https://www.youtube.com/watch?v=TQjTTqEewNQ">
    <img src="doc/teaser_cropped.gif" width="90%" alt="SuperMap teaser — click for the full video"/>
  </a>
</p>

## News

- 2026-09-02: Initial `semantic_mapping` ROS2 package published: instance-level spatio-temporal tracking (Sec. IV-B), 4D scene graph construction (Sec. IV-C), and VLN grounding (Sec. IV-D), with an offline demo and unit tests. See [Setup](#setup) below.
- 2026-07-15: Project website and teaser video are now live.
- 2026-07-15: Initial README and project overview published.

**SuperMap is a living spatial memory for embodied AI**. It perceives the world, remembers its evolution, and supports reasoning and action. It is a training-free spatio-temporal SLAM system that builds a persistent semantic world model. It fuses high-frequency geometric SLAM with asynchronous open-vocabulary perception, producing a 4D scene graph: a queryable map carrying spatial *and* temporal information for every object, enabling visual-language navigation and long-horizon reasoning on real robots.

## Highlights

- 👀 **Perceive** — Stable identities across occlusions and scene changes via 3D-aware instance association.
- 🧠 **Remember** — persistent object identities capture long-term scene evolution.                                           
- 💡 **Reason** — a queryable 4D scene graph supports spatial and tempora reasoning.                                         
- 🤖 **Act** — spatial memory naturally grounds VLN, VLA, and future embodied AI systems.  
- 🔌 **Model-agnostic** — works with Grounding DINO, YOLOE, boxer pre-baked detections; SAM2 mask refinement wraps any of them.
- ⚡ **Fully online** — real-time on robot hardware; runs **offline** on datasets or **live** via ROS2 from one codebase.
- 🏫 **Field-proven** — continuous 2-hour deployment across the CMU campus ([interactive demo](https://superodometry.com/supermap)).

## Requirements

- Ubuntu 22.04 / 24.04, NVIDIA GPU with ≥ 16 GB VRAM, Python ≥ 3.10
- ROS2 Jazzy workspace for live mode

## Setup

```bash
git clone https://github.com/superxslam/SuperMap.git
cd SuperMap
conda create -n supermap python=3.11 && conda activate supermap

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
conda install cuda -c nvidia/label/cuda-12.4.0
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run (offline)

```bash
python examples/prepare_example_dataset.py   # generate a synthetic demo sequence (one-time)
python examples/example.py                   # run the mapping pipeline
python examples/evaluate.py                  # score it with the paper's metrics (Sec. V-D / V-E)
```

No public SuperMap dataset is bundled yet, so `prepare_example_dataset.py` synthesizes a small deterministic RGB-D + odometry + detections sequence (ray-cast against a scripted scene with objects appearing/disappearing, mirroring Sec. V-C) so the pipeline is runnable end-to-end offline. Detections come with instance masks, as the paper's Grounding DINO + SAM2 pairing produces; `--no_masks` emits box-only detections to exercise the harder fallback path. Point `--data_dir` at a real capture once one is available (layout documented in `semantic_mapping/datasets.py`).

`evaluate.py` implements the paper's true-positive criterion (3D IoU > 0.1, centroid < 0.3 m, correct label) and reports per-object detection recall and change-detection recall (Sec. V-D), identity fragmentation, and a final-map precision/recall/F1 in the spirit of the Sec. V-E ablation. On the synthetic sequence (9 objects, 3 removed and 3 introduced mid-sequence):

| detections | detection recall | change recall | instance IDs / objects | final-map F1 |
|---|---|---|---|---|
| boxes + masks | 0.97 | 0.96 | 9 / 9 | 0.86 |
| boxes only (`--no_masks`) | 0.47 | 0.52 | 14 / 9 | 0.53 |

`evaluate.py` also runs the paper's segmentation benchmark (Sec. V-B): map instances are transferred onto annotated points by nearest neighbour and scored with class-level mIoU / f-mIoU / accuracy, with and without background classes (Table II), and instance-level AP25 / AP50 per class (Table III). The synthetic scene ships annotated surfaces (`gt_points.npz`), so this runs offline too; the mask run above scores:

| setting | mIoU | f-mIoU | Acc | mAP25 | mAP50 |
|---|---|---|---|---|---|
| without background | 0.996 | 0.996 | 0.996 | 1.000 | 1.000 |
| with background (wall / floor / ceiling) | 0.609 | 0.090 | 0.096 | | |

The "with background" gap is the map never predicting wall / floor / ceiling, the same effect visible in the paper's Table II. ScanNet scenes are read straight from the standard `SensReader` export plus the annotated mesh (`semantic_mapping/datasets_scannet.py`); vocabulary-to-class aliases and the background set live in `config/segmentation_eval.yaml`:

```bash
python examples/evaluate.py --data_dir scans/scene0000_00 --frame_skip 10 --detector groundingdino
```

Options: `--detector yoloe|offline|groundingdino`, `--data_dir <path>`, `--config <yaml>`, `--prompts <yaml>`, `--live` (rerun window).

## Run (live ROS2)

```bash
# Clone into your workspace src/ as `semantic_mapping`, then:
colcon build --packages-select semantic_mapping && source install/setup.bash
ros2 launch semantic_mapping semantic_mapping.launch.py
```

In live mode, the system subscribes to RGB, CameraInfo, PointCloud2, and Odometry topics published by an upstream geometric SLAM backbone (Sec. IV-A) and publishes per-object voxels (`/obj_points`), labeled boxes (`/obj_boxes`), and annotated images. Perception is asynchronous, as in the paper (Sec. V-H): the detector runs in its own thread at `detector_rate_hz` (default 1 Hz) while geometric updates continue at the sensor rate; a frame handed to the detector is fused once, under its own pose and depth, when its detections return. Map outputs are published at `publish_rate_hz`. Topic and detector settings are configured in `config/semantic_mapping.yaml`, and the detection vocabulary is defined in `config/prompts.yaml`. The world-from-camera pose (Eq. 3) is resolved through TF2 rather than a fixed parameter, so a `sensor_frame -> camera_frame` extrinsic must be in the TF tree (via a URDF/`robot_state_publisher`, or the `static_transform_publisher` the launch file includes by default — override its `camera_x`/`camera_y`/.../`camera_qw` arguments with your calibration).

## Docker

```bash
docker build -f docker/Dockerfile -t supermap/semantic_mapping .                              # full image (torch, YOLOE)
docker build -f docker/Dockerfile --build-arg INSTALL_DETECTORS=0 -t supermap/semantic_mapping:lite .  # offline/CI-sized, no GPU stack

docker run --rm -it supermap/semantic_mapping:lite \
  bash -lc "python3 examples/prepare_example_dataset.py && python3 examples/example.py"       # offline demo, no GPU needed

docker run --rm -it --gpus all --network host supermap/semantic_mapping                       # live ROS2 mode
# or: docker compose up --build
```

### Query the map (visual-language navigation, Sec. IV-D)

The 4D scene graph is the interface between the map and a VLM: an instruction is answered by serializing a (local) subgraph -- instance IDs, labels, centroids, spatial and temporal relations -- into a prompt, asking the model for target instance IDs inside `<answer>` tags, and resolving those IDs to 3D waypoints from the map.

```bash
python examples/query.py "go to the chair next to the table"                        # offline, on the demo sequence
python examples/query.py --client openai_compatible --model gpt-4o "return to where the trash can was"
python examples/query.py --client anthropic --model claude-opus-5 "..."

ros2 topic pub --once /semantic_mapping/query std_msgs/String "{data: 'go to the whiteboard next to the painting'}"
ros2 topic echo /semantic_mapping/goal           # geometry_msgs/PoseStamped, first target
ros2 topic echo /semantic_mapping/waypoints      # nav_msgs/Path, all targets in order
ros2 topic echo /semantic_mapping/answer         # JSON: target_ids, waypoints, raw response
```

The model backend is provider-agnostic (`semantic_mapping/vln/clients.py`): `openai_compatible` talks to any `/chat/completions` endpoint (OpenAI, Gemini's OpenAI-compatible endpoint, vLLM, Ollama, ...), `anthropic` to the Messages API, both via the standard library with keys read from the environment. The default `keyword` client is a deterministic no-network stand-in that matches labels named in the instruction, so everything runs without credentials -- it cannot do the relational or temporal reasoning that a real model does over the graph. Configure under `vlm:` in `config/semantic_mapping.yaml`.

### Real captures: rosbag2 to sequence

Record RGB, CameraInfo, a point cloud (or an aligned depth image), odometry, and TF from any robot, then convert the bag into the offline layout so `example.py`, `evaluate.py`, `benchmark.py`, and `query.py` run on it unchanged:

```bash
python examples/rosbag_to_sequence.py my_capture.bag --out_dir data/my_capture \
    --rgb_topic /camera/color/image_raw --camera_info_topic /camera/color/camera_info \
    --pointcloud_topic /lidar/points --odometry_topic /odometry \
    --world_frame map --camera_frame camera_color_optical_frame --detector yoloe   # bake detections for replay
```

The camera pose is resolved through the bag's TF tree exactly as the live node does (odometry is injected as its `header.frame_id -> child_frame_id` transform), the point cloud is rasterized into the camera for depth, and RGB frames are paired with the nearest depth source within `--sync_slop`. Use `--depth_topic` for RGB-D cameras whose depth is already aligned to color. Needs a sourced ROS 2 environment (the Docker image works).

### Runtime and memory (Sec. V-H)

Every map update records per-stage timings (`FrameResult.timings`), the live node logs module rates every `stats_log_period_sec`, and `examples/benchmark.py` turns a sequence into the paper's runtime table. On the synthetic scene rendered at 640x480 with pre-baked detections, CPU only (4-core Xeon, no GPU):

| module | mean latency | sustainable rate | paper (onboard, Sec. V-H) |
|---|---|---|---|
| 3D mapping (tracklet prediction, back-projection, association, map update) | 38 ms | 26 Hz | 3 Hz |
| 4D scene graph construction | 0.3 ms | > 3 kHz | 5 Hz |
| 2D detector | model-bound (YOLOE / Grounding DINO + SAM2 on GPU) | | 1 Hz |

The geometric-consistency update over all instance points dominates (27 ms of the 38 ms); memory is 0.5 MiB of point arrays for 9 instances and a 91 MiB process. Latency scales with image resolution and map size, so measure your own sequence:

```bash
python examples/benchmark.py --data_dir <sequence> --detector yoloe --json runtime.json
```

### Persist the map (living memory across sessions)

The complete map state (per-instance points with their geometric and membership evidence, label beliefs, lifecycle status, timestamps, trajectories, and the ID counter) can be written to disk and restored, so a robot starts a session knowing what it mapped last time:

```bash
python examples/example.py --save_map data/office_map                       # offline: write the final map
python examples/example.py --load_map data/office_map --save_map data/office_map   # continue from it

ros2 launch semantic_mapping semantic_mapping.launch.py map_load_path:=/maps/office map_save_path:=/maps/office
ros2 service call /semantic_mapping_node/save_map std_srvs/srv/Trigger     # or set map_autosave_sec
```

Restored instances resume as *occluded* with a reset 2D tracklet (a tracklet is camera-relative and meaningless after a restart); re-observation runs through the 3D re-activation stage like any object that left the field of view, and the geometric-consistency update retires objects that are gone. New instances keep counting from the saved ID counter, so IDs recorded by downstream consumers stay unique. Format: `map.json` + `map_arrays.npz` (`semantic_mapping/persistence.py`).

Both offline and live modes emit the same per-frame JSON schema (`semantic_mapping.serialization`) with `bbox3d`, `label`, `id`, `center`, `spatial_relations`, `status`, and `latest_stamp`, so downstream consumers can use one shared interface.

## Citation

```bibtex
@inproceedings{zhao2026supermap,
  title     = {SuperMap: A Spatio-Temporal SLAM System for Visual-Language Navigation},
  author    = {Zhao, Shibo and Chen, Guofei and Zhu, Honghao and Li, Zhiheng and Yao, Changwei and Zantout, Nader and Kim, Seungchan and Wang, Wenshan and Zhang, Ji and Scherer, Sebastian},
  booktitle = {Proceedings of Robotics: Science and Systems (RSS)},
  year      = {2026}
}
```

## Related work from SuperX SLAM

- [SuperOdom](https://github.com/superxslam/SuperOdom) — robust LiDAR-only / LiDAR-inertial odometry
- [Robustness_Metric](https://github.com/superxslam/Robustness_Metric) — robustness metric for odometry and SLAM

## Acknowledgments

Special thanks to Professor Wenshan and Ji Zhang for suggestions and extensively testing SuperOdom and SuperMap on VLN projects.

Built on [SAM2](https://github.com/facebookresearch/sam2), [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO), [YOLOE](https://docs.ultralytics.com/models/yoloe/), and [ByteTrack](https://github.com/ifzhang/ByteTrack).

Questions? Join our [Discord](https://discord.gg/Huf2GJx32y), open an issue, or contact **guofei@cmu.edu** / **shiboz@andrew.cmu.edu**.
