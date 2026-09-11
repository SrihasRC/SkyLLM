#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TYPEFLY_DIR="$(cd "$DIR/../.." && pwd)"
VENV_DIR="$TYPEFLY_DIR/.venv"
INSTALL_PREFIX="$HOME/.local"

echo "========================================================"
echo " Setting up PX4-ROS2-Gazebo-YOLOv8 Local Environment"
echo "========================================================"

# 1. Verify ROS 2
if [ ! -f "/opt/ros/humble/setup.bash" ]; then
    echo "[ERROR] ROS 2 Humble not found at /opt/ros/humble/setup.bash"
    exit 1
fi
echo "[OK] ROS 2 Humble found."

# 2. Verify Python virtual environment
if [ ! -f "$VENV_DIR/bin/python" ]; then
    echo "[ERROR] Virtualenv not found at $VENV_DIR"
    exit 1
fi
echo "[OK] Python environment found at $VENV_DIR."

# Ensure ROS packages are accessible in venv
mkdir -p "$VENV_DIR/lib/python3.10/site-packages"
cat << 'EOF' > "$VENV_DIR/lib/python3.10/site-packages/ros_humble.pth"
/opt/ros/humble/lib/python3.10/site-packages
/opt/ros/humble/local/lib/python3.10/dist-packages
EOF
echo "[OK] Configured ROS 2 Python package bridge in .venv."

# 3. Setup Gazebo model paths
echo "[INFO] Copying simulation models to ~/.gz/models and ~/.ignition/models..."
mkdir -p "$HOME/.gz/models" "$HOME/.ignition/models" "$HOME/.gz/fuel/fuel.ignitionrobotics.org/openrobotics/models"
cp -r "$DIR"/models/* "$HOME/.gz/models/" 2>/dev/null || true
cp -r "$DIR"/models/* "$HOME/.ignition/models/" 2>/dev/null || true
cp -r "$DIR"/models_docker/* "$HOME/.gz/fuel/fuel.ignitionrobotics.org/openrobotics/models/" 2>/dev/null || true
echo "[OK] Gazebo models installed."

# 4. Check / Build Micro-XRCE-DDS-Agent
if command -v MicroXRCEAgent >/dev/null 2>&1; then
    echo "[OK] MicroXRCEAgent is already installed."
else
    echo "[INFO] MicroXRCEAgent not found. Building to $INSTALL_PREFIX..."
    BUILD_TMP="/tmp/micro_xrce_agent_build"
    rm -rf "$BUILD_TMP"
    git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git "$BUILD_TMP"
    mkdir -p "$BUILD_TMP/build"
    cd "$BUILD_TMP/build"
    cmake .. -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"
    make -j"$(nproc)"
    make install
    rm -rf "$BUILD_TMP"
    echo "[OK] MicroXRCEAgent installed to $INSTALL_PREFIX/bin."
fi

# 5. Check PX4-Autopilot
PX4_DIR="${PX4_PATH:-$HOME/PX4-Autopilot}"
if [ -d "$PX4_DIR" ]; then
    echo "[OK] PX4-Autopilot found at $PX4_DIR."
else
    echo "[WARN] PX4-Autopilot not found at $PX4_DIR."
    echo "To install PX4-Autopilot for local simulation, run:"
    echo "  git clone https://github.com/PX4/PX4-Autopilot.git --recursive $HOME/PX4-Autopilot"
    echo "  bash $HOME/PX4-Autopilot/Tools/setup/ubuntu.sh"
    echo "  cd $HOME/PX4-Autopilot && make px4_sitl"
fi

if [ -d "$PX4_DIR" ]; then
    echo "[INFO] Configuring PX4 simulation world and gimbal..."
    GZ_WORLDS_DIR="$PX4_DIR/Tools/simulation/gz/worlds"
    mkdir -p "$GZ_WORLDS_DIR"
    cp "$DIR/worlds/default_docker.sdf" "$GZ_WORLDS_DIR/default.sdf"
    
    # Run gimbal setup
    "$VENV_DIR/bin/python" "$DIR/setup_gimbal.py" || true
    echo "[OK] PX4 configured."
fi

echo "========================================================"
echo " Local setup complete!"
echo " Use ./run_local.sh to launch the simulation."
echo "========================================================"
