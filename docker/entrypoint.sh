#!/bin/bash
set -e

source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
source "/workspace/install/setup.bash"

exec "$@"
