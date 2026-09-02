# syntax=docker/dockerfile:1
#
# semantic_mapping (SuperMap) container: ROS2 Jazzy on Ubuntu 24.04, per the
# project README's stated requirements. Builds this repo as a colcon
# workspace and installs its Python dependencies.
#
# The heavy open-vocabulary detector stack (torch, ultralytics/YOLOE,
# GroundingDINO, SAM2) is behind the INSTALL_DETECTORS build arg since it is
# multi-gigabyte and only needed for the yoloe/groundingdino backends; the
# offline (pre-baked detection replay) backend, the offline example, and the
# test suite all work without it. GPU access at runtime comes from the NVIDIA
# Container Toolkit (`docker run --gpus all ...`), not from CUDA baked into
# this image; building GroundingDINO's optional custom CUDA op requires an
# nvcc devel image and is out of scope here (it has a slower pure-Python
# fallback when the extension isn't compiled).
#
# Build:
#   docker build -f docker/Dockerfile -t supermap/semantic_mapping .
#   docker build -f docker/Dockerfile --build-arg INSTALL_DETECTORS=0 -t supermap/semantic_mapping:lite .
#
# Run (offline example, no GPU needed):
#   docker run --rm -it supermap/semantic_mapping \
#     bash -lc "python3 examples/prepare_example_dataset.py && python3 examples/example.py"
#
# Run (live ROS2, GPU + host networking so DDS discovery reaches other
# nodes/containers on the host, e.g. an upstream SLAM backbone):
#   docker run --rm -it --gpus all --network host supermap/semantic_mapping

ARG ROS_DISTRO=jazzy
FROM ros:${ROS_DISTRO}-ros-base

ARG ROS_DISTRO
ARG INSTALL_DETECTORS=1
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    ROS_DISTRO=${ROS_DISTRO}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-colcon-common-extensions \
        python3-rosdep \
        python3-numpy \
        python3-scipy \
        python3-yaml \
        python3-opencv \
        ros-${ROS_DISTRO}-tf2-ros \
        ros-${ROS_DISTRO}-cv-bridge \
        ros-${ROS_DISTRO}-message-filters \
        ros-${ROS_DISTRO}-sensor-msgs-py \
        ros-${ROS_DISTRO}-vision-opencv \
        ros-${ROS_DISTRO}-launch \
        ros-${ROS_DISTRO}-launch-ros \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/src/semantic_mapping
COPY package.xml requirements.txt ./

# package.xml's ROS exec_depends are satisfied directly by the apt packages
# installed above (no rosdep resolve step: this pins the exact ros-${ROS_DISTRO}-*
# packages rather than depending on network access to raw.githubusercontent.com
# at build time, which is both fragile in restricted CI environments and
# unnecessary once the mapping from package.xml keys to apt packages is known).
#
# numpy/scipy/opencv/pyyaml come from apt (above), not pip: ros-${ROS_DISTRO}-cv-bridge
# is a compiled extension linked against the numpy ABI its own apt-provided
# python3-numpy build was compiled with, and a newer pip-installed numpy
# (numpy 2.x's C-API is not backwards compatible with 1.x-compiled extensions)
# would shadow it and silently break image conversion at runtime.
RUN pip3 install --no-cache-dir --break-system-packages "pytest>=7.4,<8"

# Detector backends (large: torch + CUDA runtime wheels, ultralytics, spacy).
# GroundingDINO/SAM2 are not on PyPI as installable packages by their
# canonical names -- install them from source per their own READMEs
# (https://github.com/IDEA-Research/GroundingDINO,
#  https://github.com/facebookresearch/sam2) if you need that backend.
# "numpy<2" is pinned on both installs below (not just left to whatever
# torch/ultralytics/spacy would pull in) for the same cv_bridge ABI reason
# noted above -- pip resolves a whole command's requirements together, so
# this constrains any transitive numpy pull, not just a direct one.
RUN if [ "$INSTALL_DETECTORS" = "1" ]; then \
        pip3 install --no-cache-dir --break-system-packages \
            "numpy<2" torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
            --index-url https://download.pytorch.org/whl/cu124 \
        && pip3 install --no-cache-dir --break-system-packages \
            "numpy<2" ultralytics rerun-sdk spacy \
        && python3 -m spacy download en_core_web_sm; \
    fi

WORKDIR /workspace/src/semantic_mapping
COPY . .

WORKDIR /workspace
RUN source /opt/ros/${ROS_DISTRO}/setup.bash \
    && colcon build --packages-select semantic_mapping --symlink-install

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace/src/semantic_mapping
ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "semantic_mapping", "semantic_mapping.launch.py"]
