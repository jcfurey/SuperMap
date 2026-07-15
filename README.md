<p align="center">
  <h1 align="center">SuperMap: A Spatio-Temporal SLAM System for Visual-Language Navigation</h1>
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

> Announcement: The code will be released after RSS.

## News

- 2026-07-15: Project website and teaser video are now live.
- 2026-07-15: Initial README and project overview published.

**SuperMap is a living spatial memory for embodied AI**. It perceives the world, remembers its evolution, and supports reasoning and action. It is a training-free spatio-temporal SLAM system that builds a persistent semantic world model. It fuses high-frequency geometric SLAM with asynchronous open-vocabulary perception, producing a 4D scene graph: a queryable map carrying spatial *and* temporal information for every object, enabling visual-language navigation and long-horizon reasoning on real robots.

## Highlights

- 👀 **Perceive** — Stable identities across occlusions and scene changes via 3D-aware instance association.
- 🧠 **Remember** — persistent object identities capture long-term scene evolution.                                           
- 💡 **Reason** — a queryable 4D scene graph supports spatial and tempora reasoning.                                         
- 🤖 **Act** — spatial memory naturally grounds VLN, VLA, and future embodied AI systems.  
- 🔌 **Model-agnostic** — works with Grounding DINO, YOLOE, boxer pre-baked detections.
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
python examples/prepare_example_dataset.py   # download example dataset (one-time)
python examples/example.py                   # run the mapping pipeline
```

Options: `--detector yoloe|offline`, `--data_dir <path>`, `--config <yaml>`, `--live` (rerun window).

## Run (live ROS2)

```bash
# Clone into your workspace src/ as `semantic_mapping`, then:
colcon build --packages-select semantic_mapping && source install/setup.bash
ros2 launch semantic_mapping semantic_mapping.launch.py
```

In live mode, the system subscribes to RGB, CameraInfo, PointCloud2, and Odometry topics and publishes per-object voxels (`/obj_points`), labeled boxes (`/obj_boxes`), and annotated images. Topic, extrinsic, and detector settings are configured in `config/semantic_mapping.yaml`, and the detection vocabulary is defined in `config/prompts.yaml`.

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
