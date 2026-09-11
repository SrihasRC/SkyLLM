#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${DOCKER_IMAGE:-monemati/px4_ros2_gz_yolov8_image:latest}"
CONTAINER_NAME="px4_ros2_gazebo_yolov8_container"

echo "========================================================"
echo " Starting PX4-ROS2-Gazebo-YOLOv8 via Docker"
echo "========================================================"

# Check docker command and permissions
DOCKER_CMD="docker"
if ! docker ps >/dev/null 2>&1; then
    if sudo docker ps >/dev/null 2>&1; then
        echo "[INFO] Running docker with sudo (permission required)"
        DOCKER_CMD="sudo docker"
    else
        echo "[ERROR] Cannot access Docker daemon."
        echo "Please ensure docker service is running and your user is in docker group:"
        echo "  sudo usermod -aG docker \$USER && newgrp docker"
        exit 1
    fi
fi

# Setup X11 display permissions
echo "[INFO] Configuring X11 forwarding for display: $DISPLAY"
xhost +local:root >/dev/null 2>&1 || true
xhost +local:docker >/dev/null 2>&1 || true

XAUTH=/tmp/.docker.xauth
touch "$XAUTH"
if command -v xauth >/dev/null 2>&1 && [ -n "$DISPLAY" ]; then
    xauth nlist "$DISPLAY" 2>/dev/null | sed -e 's/^..../ffff/' | xauth -f "$XAUTH" nmerge - 2>/dev/null || true
fi

# Check if image exists locally or needs to be pulled
if ! $DOCKER_CMD image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "[INFO] Image $IMAGE_NAME not found locally. Pulling from Docker Hub..."
    $DOCKER_CMD pull "$IMAGE_NAME"
fi

# Clean up any existing container with the same name
$DOCKER_CMD rm -f "$CONTAINER_NAME" 2>/dev/null || true

# Check GPU support
GPU_FLAGS=()
if $DOCKER_CMD info 2>/dev/null | grep -qi nvidia || which nvidia-container-runtime >/dev/null 2>&1; then
    echo "[INFO] NVIDIA container toolkit detected. Enabling GPU passthrough."
    GPU_FLAGS=(
        "--gpus" "all"
        "-e" "NVIDIA_DRIVER_CAPABILITIES=all"
        "-e" "NVIDIA_VISIBLE_DEVICES=all"
    )
else
    echo "[WARN] NVIDIA container toolkit not detected. Running in standard/CPU mode."
fi

echo "[INFO] Launching container: $CONTAINER_NAME"
echo "Inside the container, tmuxinator will launch all simulation panes."
echo "Keyboard controls will be active in the controller pane."
echo "--------------------------------------------------------"

exec $DOCKER_CMD run --privileged -it \
    "${GPU_FLAGS[@]}" \
    -e MESA_GL_VERSION_OVERRIDE=3.3 \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --env="XAUTHORITY=$XAUTH" \
    --volume="$XAUTH:$XAUTH" \
    --network=host --ipc=host --shm-size=2gb \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --rm --name "$CONTAINER_NAME" \
    "$IMAGE_NAME"
