# syntax=docker/dockerfile:1
#
# SuperMap container: ROS 2 Jazzy on Ubuntu 24.04. The repository root holds
# two colcon packages -- semantic_mapping (ament_python) and supermap_msgs
# (interfaces) -- which are built together in /workspace.
#
# The heavy open-vocabulary detector stack (torch, ultralytics/YOLOE) is
# behind the INSTALL_DETECTORS build arg since it is multi-gigabyte and only
# needed for the yoloe/groundingdino backends; the offline (pre-baked
# detection replay) backend, the offline example, and the test suite all work
# without it. GPU access at runtime comes from the NVIDIA Container Toolkit
# (`docker run --gpus all ...`), not from CUDA baked into this image.
# GroundingDINO/SAM2 are installed from source per their READMEs if needed.
#
# Build:
#   docker build -f docker/Dockerfile -t supermap/semantic_mapping .
#   docker build -f docker/Dockerfile --build-arg INSTALL_DETECTORS=0 -t supermap/semantic_mapping:lite .
#
# Run (offline example, no GPU needed; the working directory is the package):
#   docker run --rm -it supermap/semantic_mapping:lite \
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

# package.xml exec/test dependencies, installed directly as apt packages (no
# rosdep resolve step: this pins the exact ros-${ROS_DISTRO}-* packages and
# needs no access to raw.githubusercontent.com at build time).
#
# numpy/scipy/opencv/pyyaml come from apt, not pip, so the package shares one
# numpy and one cv2 with the distro's ROS Python bindings. Image messages are
# decoded with plain numpy (semantic_mapping/ros_msgs.py); cv_bridge is not needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-colcon-common-extensions \
        python3-numpy \
        python3-scipy \
        python3-yaml \
        python3-opencv \
        python3-pytest \
        ros-${ROS_DISTRO}-tf2-ros \
        ros-${ROS_DISTRO}-message-filters \
        ros-${ROS_DISTRO}-sensor-msgs-py \
        ros-${ROS_DISTRO}-vision-msgs \
        ros-${ROS_DISTRO}-diagnostic-updater \
        ros-${ROS_DISTRO}-diagnostic-msgs \
        ros-${ROS_DISTRO}-nav2-msgs \
        ros-${ROS_DISTRO}-rosbag2-py \
        ros-${ROS_DISTRO}-rosidl-default-generators \
        ros-${ROS_DISTRO}-launch \
        ros-${ROS_DISTRO}-launch-ros \
        ros-${ROS_DISTRO}-ament-flake8 \
        ros-${ROS_DISTRO}-ament-pep257 \
        ros-${ROS_DISTRO}-ament-copyright \
    && rm -rf /var/lib/apt/lists/*

# Detector backends (large: torch + CUDA runtime wheels, ultralytics).
# "numpy<2" is pinned on both installs so the apt numpy stays the one in use
# (pip resolves a whole command's requirements together, so this constrains
# any transitive numpy pull, not just a direct one). ultralytics depends on
# pip's opencv-python, which would shadow apt's python3-opencv with a second
# cv2 built against a different numpy; it is removed again so `import cv2`
# resolves to the apt build.
RUN if [ "$INSTALL_DETECTORS" = "1" ]; then \
        pip3 install --no-cache-dir --break-system-packages \
            "numpy<2" torch==2.5.1 torchvision==0.20.1 \
            --index-url https://download.pytorch.org/whl/cu124 \
        && pip3 install --no-cache-dir --break-system-packages \
            "numpy<2" ultralytics rerun-sdk \
        && pip3 uninstall -y --break-system-packages opencv-python opencv-python-headless || true; \
    fi \
    && python3 -c "import cv2, numpy; assert cv2.__file__.startswith('/usr/lib/python3/'), cv2.__file__; print('cv2', cv2.__version__, 'numpy', numpy.__version__)"

WORKDIR /workspace/src/SuperMap
COPY . .

WORKDIR /workspace
RUN source /opt/ros/${ROS_DISTRO}/setup.bash \
    && colcon build --symlink-install --packages-select supermap_msgs semantic_mapping

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace/src/SuperMap/semantic_mapping
ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "semantic_mapping", "semantic_mapping.launch.py"]
