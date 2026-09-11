#!/usr/bin/env bash
# run_local.sh: Orchestrate local execution of PX4-ROS2-Gazebo-YOLOv8

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$DIR/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
PX4_DIR="${PX4_PATH:-$HOME/PX4-Autopilot}"

echo "========================================================"
echo " Starting PX4-ROS2-Gazebo-YOLOv8 Locally"
echo "========================================================"

# Source ROS 2 Humble
if [ -f "/opt/ros/humble/setup.bash" ]; then
    echo "[INFO] Sourcing ROS 2 Humble..."
    source /opt/ros/humble/setup.bash
else
    echo "[ERROR] ROS 2 Humble setup.bash not found at /opt/ros/humble/setup.bash"
    exit 1
fi

# Activate Python Virtualenv
if [ -f "$VENV_DIR/bin/activate" ]; then
    echo "[INFO] Activating virtualenv: $VENV_DIR"
    source "$VENV_DIR/bin/activate"
else
    echo "[ERROR] Virtualenv activate script not found at $VENV_DIR/bin/activate"
    exit 1
fi

# Clean up any previous leftover simulation processes
pkill -9 -f mavsdk_server 2>/dev/null || true
pkill -9 -f ros_gz_bridge 2>/dev/null || true
pkill -9 -f move_car.py 2>/dev/null || true
pkill -9 -f uav_camera_det.py 2>/dev/null || true
pkill -9 -f keyboard-mavsdk-test.py 2>/dev/null || true
pkill -9 -f px4 2>/dev/null || true

# Set resource paths for Gazebo Harmonic and PX4
export GZ_SIM_RESOURCE_PATH="$PX4_DIR/Tools/simulation/gz/models:$HOME/.gz/models:$DIR/models:$GZ_SIM_RESOURCE_PATH"
export PX4_GZ_MODELS="$PX4_DIR/Tools/simulation/gz/models"
export PX4_GZ_WORLDS="$PX4_DIR/Tools/simulation/gz/worlds"
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH"

# Track background process PIDs for clean teardown
PIDS=()
cleanup() {
    echo ""
    echo "[INFO] Stopping all background processes..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    pkill -9 -f mavsdk_server 2>/dev/null || true
    pkill -9 -f move_car.py 2>/dev/null || true
    pkill -9 -f uav_camera_det.py 2>/dev/null || true
    pkill -9 -f px4 2>/dev/null || true
    pkill -9 -f "gz sim" 2>/dev/null || true
    wait 2>/dev/null || true
    echo "[INFO] All processes stopped."
}
trap cleanup EXIT INT TERM

# 1. Start Micro-XRCE-DDS-Agent
if command -v MicroXRCEAgent >/dev/null 2>&1; then
    echo "[INFO] Starting MicroXRCEAgent on port 8888..."
    MicroXRCEAgent udp4 -p 8888 >/dev/null 2>&1 &
    PIDS+=($!)
else
    echo "[WARN] MicroXRCEAgent not found. Run ./setup_local.sh to build it."
fi

# 2. Start PX4 Autopilot SITL
if [ -f "$PX4_DIR/build/px4_sitl_default/bin/px4" ]; then
    echo "[INFO] Starting PX4 Autopilot SITL (logging to /tmp/px4_sitl.log)..."
    (cd "$PX4_DIR" && PX4_SYS_AUTOSTART=4002 PX4_GZ_MODEL_POSE="268.08,-128.22,3.86,0.00,0,-0.7" PX4_GZ_MODEL=x500_depth ./build/px4_sitl_default/bin/px4 > /tmp/px4_sitl.log 2>&1) &
    PIDS+=($!)
    echo "[INFO] Waiting for Gazebo world initialization..."
    sleep 4
    sleep 4
else
    echo ""
    echo "[ERROR] PX4 SITL binary not found at $PX4_DIR/build/px4_sitl_default/bin/px4."
    echo "=========================================================================="
    echo "Gazebo and the drone camera are spawned directly by PX4 SITL."
    echo "Without PX4 SITL, Gazebo will not open and the camera feed will be blank."
    echo ""
    echo "Options:"
    echo "  1) Run via Docker (Recommended, has PX4 & Gazebo pre-built):"
    echo "     ./run_docker.sh"
    echo ""
    echo "  2) Build PX4 SITL locally on your host:"
    echo "     git clone https://github.com/PX4/PX4-Autopilot.git --recursive ~/PX4-Autopilot"
    echo "     bash ~/PX4-Autopilot/Tools/setup/ubuntu.sh"
    echo "     cd ~/PX4-Autopilot && make px4_sitl"
    echo "=========================================================================="
    exit 1
fi

# 3. Start YOLO26n Camera Detection Node (Direct Gazebo transport)
if [ "$1" != "--no-yolo" ] && [ "$NO_YOLO" != "1" ]; then
    echo "[INFO] Starting YOLO26n Camera Detection node..."
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python "$DIR/uav_camera_det.py" &
    PIDS+=($!)
    sleep 2
fi

# 6. Start Keyboard Controller or wait for external controllers (e.g. TypeFly)
if [ "$1" = "--sim-only" ] || [ "$1" = "--no-keyboard" ] || [ "$NO_KEYBOARD" = "1" ]; then
    echo "========================================================"
    echo " PX4 SITL & Gazebo simulation running in background!"
    echo " Ready for SkyLLM (skyllm-run)."
    echo " Press Ctrl+C at any time to exit and stop all services."
    echo "========================================================"
    wait
else
    echo "========================================================"
    echo " All services running!"
    echo " Drone keyboard controls starting in foreground..."
    echo " Press Ctrl+C at any time to exit and stop all services."
    echo "========================================================"
    python "$DIR/keyboard-mavsdk-test.py"
fi
