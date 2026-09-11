# SkyLLM: Autonomous Vision-Language-Action (VLA) UAV Flight Control System

[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Autopilot: PX4 SITL](https://img.shields.io/badge/Autopilot-PX4_SITL-orange.svg)](https://px4.io/)
[![Simulator: Gazebo Harmonic](https://img.shields.io/badge/Simulator-Gazebo_Harmonic-blueviolet.svg)](https://gazebosim.org/)
[![Vision: YOLO26n](https://img.shields.io/badge/Vision-YOLO26n-green.svg)](https://github.com/ultralytics/ultralytics)

**SkyLLM** is an end-to-end Vision-Language-Action (VLA) autonomous flight control system. Operators type natural language instructions, and an LLM planner reasons over real-time computer vision from the drone's camera to synthesize, validate, and execute closed-loop Python flight plans.

All flight control is executed through **MAVLink / MAVSDK** using native **PX4 Offboard Position Setpoints (`PositionNedYaw`)** with 24-state **EKF2 sensor fusion**, eliminating aerodynamic drift and achieving $< 6\text{ cm}$ hover accuracy.

---

## Key Features

- **Natural Language Flight Planning:** LLM code-synthesis engine (powered by Groq / LLaMA-3 / OpenAI) translates intent into executable, sandboxed Python scripts.
- **Closed-Loop Offboard Precision:** Cartesian North-East-Down (NED) position setpoints with compass-aligned body frame transformation, eliminating hover drift.
- **Acrobatic 360° Flips:** High-speed angular rate control (`AttitudeRate` at $720^\circ/\text{s}$) with automated pre-flip altitude verification, vertical thrust burst, and recovery catch lock.
- **Autonomous GPS Navigation:** Global WGS84 GPS coordinate waypoint flight (`goto_gps`), return-to-launch (`return_home` / `return_to_launch`), and circular loiter orbiting (`orbit`).
- **FlySky Controller Integration:** Full transmitter channel emulation including motor arm/disarm, 3-position flight mode switching, camera gimbal tilt/pan/center, and battery monitoring.
- **Real-Time YOLO26n Vision:** Lightweight neural vision inference directly streaming from Gazebo Harmonic transport topics with built-in object tracking and false-positive suppression.
- **Minimalist Monochrome Ground Control Station:** Custom zero-border high-contrast web GCS with real-time video feed, live telemetry status bar, and Server-Sent Events (SSE) chat logs.
- **AST Plan Validation & Safety Sandbox:** Abstract Syntax Tree parsing enforcing timeout budgets, loop bounds, and maximum displacement/velocity clamps before any command touches the autopilot.

---

## System Architecture

```
                               +-------------------------------------+
                               |         Operator Web GCS            |
                               | (Minimalist Monochrome Interface)   |
                               +------------------+------------------+
                                                  |
                                                  v
                               +-------------------------------------+
                               |         LLM Planner Engine          |
                               |  - Natural Language Understanding   |
                               |  - YOLO26n Visual Context Feed      |
                               |  - Python Plan Code Generation      |
                               +------------------+------------------+
                                                  |
                                                  v
                               +-------------------------------------+
                               |     AST Validator & Plan Policy     |
                               |  - Syntax & AST Tree Security       |
                               |  - Bounded Loops & Timeouts         |
                               |  - Safety Clamps (Speed / Distance) |
                               +------------------+------------------+
                                                  |
                                                  v
                               +-------------------------------------+
                               |       SkyLLM PX4 Robot Wrapper      |
                               |  - Body Frame -> NED Transformations|
                               |  - Dynamic Skillset (51 Skills)     |
                               +------------------+------------------+
                                                  |
                                                  v
                               +-------------------------------------+
                               |       MAVSDK (Asyncio Loop)         |
                               |  - MAVLink Protocol Serialization   |
                               |  - UDP Socket Communication         |
                               +------------------+------------------+
                                                  |
                                                  v
         +-------------------------------------------------------------------------+
         |                       PX4 Autopilot SITL / Hardware                     |
         |  - EKF2 State Estimation (IMU, Barometer, GPS, Optical Flow)            |
         |  - Offboard Position Setpoints (PositionNedYaw)                         |
         |  - Offboard Attitude Rate Setpoints (AttitudeRate for 360 Flips)        |
         |  - Gazebo Harmonic Simulator / Physical Drone (Pixhawk)                 |
         +-------------------------------------------------------------------------+
```

---

## Repository Structure

```
SkyLLM/
├── skyllm/                       # Core Python package
│   ├── assets/                   # Web GCS templates, prompts, guidelines
│   ├── config/                   # Robot and sensor configuration
│   ├── platforms/                # Hardware/simulation wrappers (PX4Wrapper)
│   ├── proto/                    # Protocol buffers for vision service
│   ├── serving/                  # YOLO vision microservice (YOLO26n)
│   ├── launcher.py               # Single-command launcher orchestrator
│   ├── llm_controller.py         # Controller loop & SSE chat logs
│   ├── llm_planner.py            # LLM prompt synthesis & code generation
│   ├── llm_wrapper.py            # LLM API client (Groq / OpenAI)
│   ├── plan_execution.py         # AST validator & safety sandbox
│   ├── robot_info.py             # Robot configuration data models
│   ├── robot_wrapper.py          # Robot abstraction base class & skills
│   ├── skillset.py               # Dynamic skillset registry
│   ├── utils.py                  # Logging & utilities
│   ├── webui.py                  # Ground Control Station Flask server
│   └── yolo_client.py            # Client-side YOLO tracking node
├── simulation/                   # Unified PX4 Gazebo simulation suite
│   ├── models/                   # Gazebo models (x500_depth drone, camera, gimbal)
│   ├── worlds/                   # Gazebo simulation worlds (default.sdf)
│   ├── run_local.sh              # Local simulation launcher (--sim-only supported)
│   ├── run_docker.sh             # Docker simulation launcher
│   ├── setup_local.sh            # Local simulation setup and build script
│   ├── setup_gimbal.py           # Gimbal setup utility
│   ├── uav_camera_det.py         # Standalone vision node
│   ├── yolo26n.pt                # 5.5MB lightweight vision weights
│   └── README.md                 # Simulation technical details
├── tests/                        # Automated unit test suite (115 passing tests)
├── docker/                       # Containerized vision deployment configs
├── .env.example                  # Environment configuration template
├── .gitignore                    # Version control ignore rules
├── pyproject.toml                # Project packaging, metadata & CLI entrypoints
└── README.md                     # Project documentation
```

---

## Quick Start

### 1. Prerequisites

- **OS:** Linux (Ubuntu 22.04 LTS recommended)
- **Python:** 3.10+
- **PX4 Autopilot SITL & Gazebo Harmonic** (pre-installed or built locally in `~/PX4-Autopilot`)
- **ROS 2 Humble** (optional for fallback transport, Gazebo Transport is used directly)
- **Groq API Key** (free at [console.groq.com](https://console.groq.com/))

### 2. Installation

Clone the repository and install the Python package in editable mode:

```bash
git clone https://github.com/<your-username>/SkyLLM.git
cd SkyLLM

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package dependencies
pip install -e .

# Configure API Key
cp .env.example .env
# Edit .env and set your GROQ_API_KEY
```

---

## Running the System

### Step 1: Start the Simulation (Terminal 1)

Launch the PX4 SITL simulation and Gazebo Harmonic:

```bash
cd simulation
./run_local.sh --sim-only
```

*Gazebo will launch with the x500 depth quadcopter initialized in the virtual flight environment.*

### Step 2: Start the Ground Control Station (Terminal 2)

In a new terminal, launch the SkyLLM Ground Control Station:

```bash
cd SkyLLM
source .venv/bin/activate
skyllm-run
```

Open your browser at **`http://localhost:50000`** to access the Ground Control Station.

---

## Example Natural Language Commands

Type any of the following natural language instructions into the command console:

### Flight & Movement
- `"takeoff"` - Arms motors and takes off to stable 2.5m hover.
- `"move forward 5 meters"` - Precision closed-loop forward movement using NED position setpoints.
- `"climb 3 meters"` / `"descend 1.5 meters"` - Closed-loop altitude adjustment.
- `"rotate left 90 degrees"` - Closed-loop yaw rotation.
- `"hold"` - Locks position loiter hover.
- `"land"` - Safely descends and lands.

### Acrobatic 360° Flips
- `"flip"` / `"front flip"` - Performs a full 360° front aerial flip.
- `"back flip"` - Performs a full 360° back flip.
- `"flip left"` - Performs a full 360° left roll flip.
- `"flip right"` - Performs a full 360° right roll flip.

### GPS & Autonomous Navigation
- `"flight telemetry"` - Retrieves and displays real-time battery, GPS coordinates, satellites, and flight mode.
- `"get gps"` - Displays current WGS84 GPS latitude, longitude, and altitude.
- `"return home"` - Navigates back to the launch GPS coordinates and lands safely.
- `"orbit 5 meters"` - Flies a circular loiter circle while keeping the camera locked onto center.

### FlySky & Gimbal Controls
- `"disarm"` - Disarms motors safely.
- `"tilt camera -45 degrees"` - Tilts camera gimbal downwards.
- `"center camera"` - Resets gimbal pitch and yaw to forward 0°.
- `"set flight mode hold"` - Changes autopilot flight mode.

---

## Running Automated Tests

Run the full test suite verifying plan validation, security sandbox policies, and PX4 flight controls:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/
```

Expected result:
```
============================= 115 passed in 2.01s ==============================
```

---

## Acknowledgements & Credits

This project builds upon the foundational concept and architecture of [TypeFly](https://github.com/typefly/TypeFly). While TypeFly provided the initial inspiration and groundwork for language-driven robotics, **SkyLLM** significantly redesigns and expands the platform for autonomous unmanned aerial vehicles (UAVs):
- Native **PX4 Autopilot SITL & Gazebo Harmonic** physics simulation integration.
- Closed-loop **Offboard Position Setpoints (`PositionNedYaw`)** with compass-aligned body frame transformation eliminating aerodynamic hover drift.
- Full **GPS waypoint navigation**, Return-To-Launch (RTL), and circular orbit loitering.
- 4-stage acrobatic **360° aerial flips** (`AttitudeRate` at $720^\circ/\text{s}$).
- Real-time **YOLO26n** lightweight vision inference directly over Gazebo Transport.
- Emulated **FlySky transmitter controls** (arm/disarm, mode switches, gimbal tilt/pan/center).
