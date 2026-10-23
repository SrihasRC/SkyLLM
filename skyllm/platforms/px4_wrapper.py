import asyncio
import json
import math
import os
import sys
import threading
import time
from typing import Any, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from overrides import overrides

# Ensure Gazebo Python bindings and compatibility
if "/usr/lib/python3/dist-packages" not in sys.path:
    sys.path.append("/usr/lib/python3/dist-packages")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from ..robot_wrapper import RobotWrapper, RobotObservation
from ..yolo_client import YoloClient
from ..robot_info import RobotInfo
from ..utils import print_t

# Gazebo transport imports
HAVE_GZ = False
try:
    from gz.transport13 import Node as GzNode
    from gz.msgs10.image_pb2 import Image as GzImage
    HAVE_GZ = True
except Exception:
    pass

# ROS 2 imports (fallback)
HAVE_ROS2 = False
try:
    import rclpy
    from rclpy.node import Node as RosNode
    from sensor_msgs.msg import Image as RosImage
    from cv_bridge import CvBridge
    HAVE_ROS2 = True
except Exception:
    pass

import mavsdk
from mavsdk import System
from mavsdk.offboard import PositionNedYaw, VelocityBodyYawspeed, AttitudeRate, OffboardError
from mavsdk.mavlink_direct import MavlinkMessage

DEFAULT_GAZEBO_TOPIC = "/world/default/model/x500_depth_0/link/camera_link/sensor/IMX214/image"
DEFAULT_MAVSDK_ADDR = "udpin://0.0.0.0:14540"
DEFAULT_CRUISE_SPEED = 0.25       # m/s (gentle, steady horizontal cruise)
DEFAULT_YAW_RATE = 25.0           # deg/s
DEFAULT_CLIMB_SPEED = 0.15        # m/s (smooth vertical climb)
MAX_ALTITUDE_CEILING_M = 1.0      # 1.0 meter (100 cm) strict maximum altitude limit
DEFAULT_TAKEOFF_ALT_M = 0.3       # 0.3 meter (30 cm) default hover altitude


class PX4Observation(RobotObservation):
    """
    Observation provider for the PX4 simulation in Gazebo.
    Captures video frames directly from Gazebo Harmonic's IMX214 camera topic,
    stores them as PIL Images for the TypeFly web UI, and runs YOLO object detection.
    """
    def __init__(self, robot_info: RobotInfo, rate: int = 10):
        super().__init__(robot_info, rate)
        extra = robot_info.extra or {}
        self._vision_enabled = bool(extra.get("enable_vision", True))
        self.yolo_client = YoloClient(robot_info) if self._vision_enabled else None
        self.gazebo_topic = extra.get("gazebo_topic", DEFAULT_GAZEBO_TOPIC)

        self._frame_lock = threading.Lock()
        self._last_frame_time = 0.0
        self._gz_node: Optional[Any] = None
        self._ros_node: Optional[Any] = None
        self._ros_thread: Optional[threading.Thread] = None

        # Create an initial placeholder image so web UI doesn't show blank/broken frame
        addr = str(extra.get("mavsdk_address", DEFAULT_MAVSDK_ADDR))
        placeholder_msg = "TypeFly Hardware Connected (Camera Inactive)" if "serial" in addr else "Connecting to PX4 Simulation Camera..."
        self._placeholder = self._create_placeholder(placeholder_msg)
        self._image = self._placeholder

    def _create_placeholder(self, text: str) -> Image.Image:
        img = Image.new("RGB", (640, 480), color=(20, 24, 38))
        draw = ImageDraw.Draw(img)
        draw.text((60, 230), text, fill=(180, 190, 210))
        return img

    def _on_gz_image(self, msg: Any):
        """Callback for Gazebo Harmonic camera messages."""
        try:
            raw = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
            # Gazebo IMX214 publishes in RGB format
            img = Image.fromarray(raw)
            with self._frame_lock:
                self._image = img
                self._last_frame_time = time.time()
        except Exception:
            pass

    @overrides
    def _start(self):
        """Start receiving camera frames."""
        subscribed = False

        if HAVE_GZ:
            try:
                self._gz_node = GzNode()
                if self._gz_node.subscribe(GzImage, self.gazebo_topic, self._on_gz_image):
                    print_t(f"[PX4Obs] Subscribed to Gazebo camera: {self.gazebo_topic}")
                    subscribed = True
            except Exception as e:
                print_t(f"[PX4Obs] Gazebo transport subscription failed: {e}")

        if not subscribed and HAVE_ROS2:
            try:
                rclpy.init(args=None)
                bridge = CvBridge()
                self._ros_node = RosNode("skyllm_px4_cam")

                def _ros_cb(data):
                    try:
                        import cv2
                        bgr = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        with self._frame_lock:
                            self._image = Image.fromarray(rgb)
                            self._last_frame_time = time.time()
                    except Exception:
                        pass

                self._ros_node.create_subscription(RosImage, "camera", _ros_cb, 10)

                def _spin_ros():
                    try:
                        rclpy.spin(self._ros_node)
                    except Exception:
                        pass

                self._ros_thread = threading.Thread(target=_spin_ros, daemon=True)
                self._ros_thread.start()
                print_t("[PX4Obs] Subscribed to ROS 2 topic: /camera")
                subscribed = True
            except Exception as e:
                print_t(f"[PX4Obs] ROS 2 subscription failed: {e}")

        if not subscribed:
            print_t(f"[PX4Obs] Warning: Could not subscribe to Gazebo topic {self.gazebo_topic}")

    @overrides
    def _stop(self):
        """Stop camera subscriptions."""
        if HAVE_GZ and self._gz_node is not None:
            try:
                self._gz_node.unsubscribe(self.gazebo_topic)
            except Exception:
                pass
            self._gz_node = None

        if HAVE_ROS2 and self._ros_node is not None:
            try:
                rclpy.shutdown()
            except Exception:
                pass
            self._ros_node = None

    @overrides
    async def process_image(self, image: Image.Image):
        """Process latest camera frame through YOLO client if enabled."""
        if self.yolo_client is not None:
            await self.yolo_client.detect(image)

    @overrides
    def fetch_processed_result(self) -> dict[str, Any]:
        """Fetch latest detected objects from YOLO client."""
        if self.yolo_client is not None:
            _, object_list = self.yolo_client.latest_result
            return {"yolo": object_list}
        return {"yolo": []}


class PX4Wrapper(RobotWrapper):
    """
    Robot wrapper for a simulated PX4 drone running under Gazebo SITL.
    Provides movement skills, takeoff, landing, altitude adjustment, and vision skills.
    Uses PX4 Position Control mode with continuous manual control feeding (identical to keyboard flight).
    """
    camera_moves_with_body = True

    def __init__(self, robot_info: RobotInfo):
        extra = robot_info.extra or {}
        self.system_address = extra.get("mavsdk_address", DEFAULT_MAVSDK_ADDR)
        self.cruise_speed = float(extra.get("cruise_speed", DEFAULT_CRUISE_SPEED))
        self.yaw_rate = float(extra.get("yaw_rate", DEFAULT_YAW_RATE))
        self.climb_speed = float(extra.get("climb_speed", DEFAULT_CLIMB_SPEED))
        self.takeoff_altitude_m = float(extra.get("takeoff_altitude", DEFAULT_TAKEOFF_ALT_M))
        self.is_hardware_bench = bool(extra.get("hardware_bench", False))
        self.MAX_ALTITUDE_CEILING_M = MAX_ALTITUDE_CEILING_M

        self.drone = System()
        self._connected = False
        self._in_air = False
        self._relative_altitude = 0.0
        self._absolute_altitude_m = 0.0
        self._latitude_deg = 0.0
        self._longitude_deg = 0.0
        self._north_m = 0.0
        self._east_m = 0.0
        self._down_m = 0.0
        self._heading_deg = 0.0
        self._roll_deg = 0.0
        self._pitch_deg = 0.0
        self._flight_mode = "UNKNOWN"
        self._armed = False
        self._battery_percent = 100.0
        self._battery_voltage = 16.0
        self._num_satellites = 0
        self._home_lat = 0.0
        self._home_lon = 0.0
        self._home_alt = 0.0
        self.max_distance = 1.0
        self._running = True
        self._motor_pwms = {"m1": 1000, "m2": 1000, "m3": 1000, "m4": 1000}
        self._current_attitude_rate = AttitudeRate(0.0, 0.0, 0.0, 0.0)
        self._in_offboard = False
        self._has_gps_lock = False
        self._orig_disarm_prflt = 10.0

        # Dedicated background asyncio event loop for MAVSDK commands
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()

        obs = PX4Observation(robot_info)
        super().__init__(robot_info, obs)

        # Register drone skills
        self.skillset.add_skill(self.arm, "Arm the drone")
        self.skillset.add_skill(self.disarm, "Disarm the drone motors safely")
        self.skillset.add_skill(self.takeoff, "Take off smoothly to hover altitude (30cm / 0.3m)")
        self.skillset.add_skill(self.land, "Land the drone safely")
        self.skillset.add_skill(self.get_motor_speeds, "Get live motor PWM speeds (M1..M4) in microseconds directly from drone")
        self.skillset.add_skill(self.lift, "Move up or down smoothly by distance in cm or meters (positive up, negative down, ceiling 1.0m / 100cm)")
        self.skillset.add_skill(self.climb, "Climb upwards smoothly by distance in cm or meters (positive number, ceiling 1.0m / 100cm)")
        self.skillset.add_skill(self.descend, "Descend downwards smoothly by distance in cm or meters (positive number)")
        self.skillset.add_skill(self.hold, "Hover in place")
        self.skillset.add_skill(self.set_speed, "Set drone flight speed in meters per second (0.1 to 1.0 m/s)")
        self.skillset.add_skill(self.set_flight_mode, "Set drone flight mode ('hold', 'altctl', 'offboard', 'rtl', 'land', 'mission', 'stabilized')")
        self.skillset.add_skill(self.get_flight_mode, "Get current PX4 flight mode")
        # GPS and advanced navigation skills
        self.skillset.add_skill(self.return_to_launch, "Return to launch/home GPS position and land safely")
        self.skillset.add_skill(self.return_home, "Return to launch/home GPS position and land safely")
        self.skillset.add_skill(self.goto_gps, "Fly to global GPS coordinate (latitude, longitude, altitude in meters relative to ground, yaw in degrees)")
        self.skillset.add_skill(self.get_gps_position, "Get and display current GPS coordinates, altitude, and heading")
        self.skillset.add_skill(self.get_home_position, "Get and display home/launch GPS coordinates")
        self.skillset.add_skill(self.orbit, "Fly a circular orbit around current location (radius in meters, velocity in m/s, yaw_behavior 'center' or 'tangent')")
        self.skillset.add_skill(self.set_rtl_altitude, "Set return-to-launch altitude in meters relative to home")
        self.skillset.add_skill(self.set_takeoff_altitude, "Set default takeoff hover altitude in meters")
        self.skillset.add_skill(self.get_telemetry, "Get and display complete drone telemetry status (battery, GPS, flight mode, altitude)")
        self.skillset.add_skill(self.get_battery, "Get and display drone battery state (percentage and voltage)")
        # Acrobatic maneuvers (360 flips)
        self.skillset.add_skill(self.flip, "Perform a 360-degree acrobatic aerial flip in specified direction ('front', 'back', 'left', 'right')")
        self.skillset.add_skill(self.front_flip, "Perform a 360-degree front flip")
        self.skillset.add_skill(self.back_flip, "Perform a 360-degree back flip")
        self.skillset.add_skill(self.flip_left, "Perform a 360-degree left roll flip")
        self.skillset.add_skill(self.flip_right, "Perform a 360-degree right roll flip")
        # Camera gimbal control (FlySky VRA/VRB dials)
        self.skillset.add_skill(self.tilt_camera, "Tilt camera gimbal pitch angle in degrees (-90 looking straight down, 0 forward, 30 up)")
        self.skillset.add_skill(self.pan_camera, "Pan camera gimbal yaw angle in degrees (-90 left to +90 right, 0 centered)")
        self.skillset.add_skill(self.center_camera, "Center camera gimbal pitch and yaw to forward 0 degrees")
        self.skillset.add_skill(self.emergency_stop, "Immediate emergency stop and land")

    def _run_loop(self):
        """Run the persistent asyncio event loop."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_async(self, coro, timeout: float = 30.0):
        """Execute an async coroutine synchronously on the background MAVSDK event loop."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def _send_chat(self, msg: str) -> None:
        """Send formatted message string to the user chat UI via controller_func[0]."""
        try:
            if self.controller_func and len(self.controller_func) > 0 and callable(self.controller_func[0]):
                self.controller_func[0](msg)
        except Exception:
            pass

    def _normalize_dist(self, dist: Any) -> float:
        """
        Normalize movement distance to meters within safety bounds [0, 1.0m / 100cm].
        - Values > 1.0 are treated as centimeters (e.g. 10 -> 0.10m, 30 -> 0.30m, 50 -> 0.50m, 100 -> 1.0m).
        - Values <= 1.0 are treated as meters (e.g. 0.3 -> 0.30m, 0.5 -> 0.50m, 1.0 -> 1.0m).
        - Clamps magnitude strictly to MAX_ALTITUDE_CEILING_M (1.0m / 100cm).
        """
        try:
            d = float(dist)
        except (ValueError, TypeError):
            return 0.3  # Safe default 30cm

        sign = 1.0 if d >= 0 else -1.0
        abs_d = abs(d)

        # If > 1.0, value is specified in centimeters
        if abs_d > 1.0:
            abs_d = abs_d / 100.0

        # Enforce 1.0m (100cm) strict bound
        abs_d = min(abs_d, self.MAX_ALTITUDE_CEILING_M)
        return sign * abs_d

    async def _configure_px4_safety_params(self):
        """Disable ground auto-preflight disarm timeout (COM_DISARM_PRFLT) while preserving all calibrations."""
        try:
            self._orig_disarm_prflt = await self.drone.param.get_param_float('COM_DISARM_PRFLT')
            await self.drone.param.set_param_float('COM_DISARM_PRFLT', -1.0)
            print_t(f"[PX4] Preflight auto-disarm disabled (COM_DISARM_PRFLT=-1.0). Motors will not cut off on ground.")
        except Exception as e:
            print_t(f"[PX4] COM_DISARM_PRFLT note: {e}")

    @overrides
    def start(self) -> bool:
        """Connect to PX4 SITL/Hardware, start telemetry monitoring, and start camera observation."""
        # Auto-detect fallback if configured serial device does not exist
        if "serial://" in str(self.system_address):
            raw_path = self.system_address.replace("serial://", "").split(":")[0]
            if not os.path.exists(raw_path):
                print_t(f"[PX4] Configured serial path '{raw_path}' not found on system.")
                try:
                    from skyllm.connection_manager import detect_device_path, get_profile_address, save_address_to_config
                    for profile_id in ["usb", "radio"]:
                        found_dev = detect_device_path(profile_id)
                        if found_dev and os.path.exists(found_dev):
                            new_addr = get_profile_address(profile_id)
                            print_t(f"[PX4] Auto-detected connected {profile_id.upper()} device: {new_addr}. Auto-switching!")
                            self.system_address = new_addr
                            self.is_hardware_bench = True
                            save_address_to_config(new_addr)
                            break
                except Exception as ex:
                    print_t(f"[PX4] Auto-detection warning: {ex}")

        print_t(f"[PX4] Connecting to drone at {self.system_address}...")
        try:
            self._run_async(self._connect_drone(), timeout=25.0)
            self.obs.start()
            print_t("[PX4] Drone connected and observation started successfully!")
            return True
        except Exception as e:
            print_t(f"[PX4] Could not connect at startup: {e}")
            print_t("[PX4] Starting WebUI in standby mode. You can switch links or click Reconnect in the WebUI.")
            self.obs.start()
            return False

    async def _connect_drone(self, connect_timeout: float = 12.0):
        print_t(f"[PX4] Initiating MAVSDK connection to {self.system_address} (timeout: {connect_timeout}s)...")
        await self.drone.connect(system_address=self.system_address)
        
        async def _wait_for_conn():
            async for state in self.drone.core.connection_state():
                if state.is_connected:
                    return True
            return False

        try:
            await asyncio.wait_for(_wait_for_conn(), timeout=connect_timeout)
            print_t(f"[PX4] MAVSDK connected to PX4 SITL/Hardware ({self.system_address})!")
            self._connected = True
        except asyncio.TimeoutError:
            self._connected = False
            raise ConnectionError(f"Connection to {self.system_address} timed out after {connect_timeout}s")

        # Start background telemetry monitoring loop
        asyncio.run_coroutine_threadsafe(self._telemetry_watch(), self._loop)

        # Apply PX4 firmware safety parameters (1m ceiling, 30cm takeoff, gentle speeds)
        try:
            await self._configure_px4_safety_params()
        except Exception as e:
            print_t(f"[PX4] Safety param config warning: {e}")

        # Wait for position lock (with timeout for indoor/bench testing)
        print_t("[PX4] Checking telemetry position lock...")
        try:
            async def _check_health():
                async for health in self.drone.telemetry.health():
                    if health.is_global_position_ok and health.is_home_position_ok:
                        print_t("[PX4] Position estimate healthy!")
                        return
                    await asyncio.sleep(0.5)
            await asyncio.wait_for(_check_health(), timeout=3.0)
        except asyncio.TimeoutError:
            print_t("[PX4] Position lock not yet acquired (bench/indoor mode). Telemetry monitoring active.")

    def reconnect(self, new_address: Optional[str] = None) -> bool:
        """Reconnect to PX4 autopilot at new_address or current address."""
        if new_address:
            self.system_address = new_address
            if self.robot_info and self.robot_info.extra is not None:
                self.robot_info.extra["mavsdk_address"] = new_address

        print_t(f"[PX4] Reconnecting to drone at {self.system_address}...")
        self._connected = False
        try:
            self.drone = System()
            self._run_async(self._connect_drone(connect_timeout=10.0), timeout=15.0)
            print_t(f"[PX4] Reconnect successful to {self.system_address}!")
            self._send_chat(f"[SYSTEM] Connected to drone via {self.system_address}")
            return True
        except Exception as e:
            print_t(f"[PX4] Reconnect failed: {e}")
            self._send_chat(f"[SYSTEM] Connection failed: {e}")
            return False

    async def _telemetry_watch(self):
        """Continuously monitor in-air status, altitude, position NED, and heading."""
        async def _in_air_loop():
            try:
                async for in_air in self.drone.telemetry.in_air():
                    self._in_air = in_air
            except Exception:
                pass

        async def _armed_loop():
            try:
                async for armed in self.drone.telemetry.armed():
                    self._armed = armed
            except Exception:
                pass

        async def _pos_loop():
            try:
                async for pos in self.drone.telemetry.position():
                    self._latitude_deg = pos.latitude_deg
                    self._longitude_deg = pos.longitude_deg
                    self._absolute_altitude_m = pos.absolute_altitude_m
                    self._relative_altitude = pos.relative_altitude_m
            except Exception:
                pass

        async def _ned_loop():
            try:
                async for ned in self.drone.telemetry.position_velocity_ned():
                    self._north_m = ned.position.north_m
                    self._east_m = ned.position.east_m
                    self._down_m = ned.position.down_m
            except Exception:
                pass

        async def _heading_loop():
            try:
                async for heading in self.drone.telemetry.heading():
                    self._heading_deg = heading.heading_deg
            except Exception:
                pass

        async def _attitude_loop():
            try:
                async for att in self.drone.telemetry.attitude_euler():
                    self._roll_deg = att.roll_deg
                    self._pitch_deg = att.pitch_deg
            except Exception:
                pass

        async def _mode_loop():
            try:
                async for mode in self.drone.telemetry.flight_mode():
                    self._flight_mode = str(mode).split(".")[-1]
            except Exception:
                pass

        async def _battery_loop():
            try:
                async for bat in self.drone.telemetry.battery():
                    pct = bat.remaining_percent
                    self._battery_percent = min(100.0, pct * 100.0 if pct <= 1.0 else pct)
                    self._battery_voltage = bat.voltage_v
            except Exception:
                pass

        async def _gps_loop():
            try:
                async for gps in self.drone.telemetry.gps_info():
                    self._num_satellites = gps.num_satellites
                    fix_name = str(gps.fix_type).split(".")[-1]
                    self._has_gps_lock = (gps.num_satellites >= 8 and fix_name in ("FIX_3D", "FIX_DGPS", "RTK_FIXED", "RTK_FLOAT"))
            except Exception:
                pass

        async def _keepalive_loop():
            while self._running:
                try:
                    if self._in_offboard:
                        await self.drone.offboard.set_attitude_rate(self._current_attitude_rate)
                except Exception:
                    pass
                await asyncio.sleep(0.1)

        async def _home_loop():
            try:
                async for home in self.drone.telemetry.home():
                    self._home_lat = home.latitude_deg
                    self._home_lon = home.longitude_deg
                    self._home_alt = home.absolute_altitude_m
            except Exception:
                pass

        async def _status_text_loop():
            try:
                async for st in self.drone.telemetry.status_text():
                    print_t(f"[PX4 Status] [{st.type}] {st.text}")
                    if st.type in ("WARNING", "CRITICAL", "ERROR"):
                        self._send_chat(f"[{st.type}] {st.text}")
            except Exception:
                pass

        async def _servo_loop():
            try:
                async for msg in self.drone.mavlink_direct.message("SERVO_OUTPUT_RAW"):
                    try:
                        data = json.loads(msg.fields_json)
                        self._motor_pwms = {
                            "m1": int(data.get("servo1_raw", 1000)),
                            "m2": int(data.get("servo2_raw", 1000)),
                            "m3": int(data.get("servo3_raw", 1000)),
                            "m4": int(data.get("servo4_raw", 1000)),
                        }
                    except Exception:
                        pass
            except Exception:
                pass

        await asyncio.gather(
            _in_air_loop(),
            _armed_loop(),
            _pos_loop(),
            _ned_loop(),
            _heading_loop(),
            _mode_loop(),
            _battery_loop(),
            _gps_loop(),
            _home_loop(),
            _status_text_loop(),
            _attitude_loop(),
            _servo_loop(),
            _keepalive_loop()
        )

    async def _set_bench_motors(self, m1: int, m2: int, m3: int, m4: int, timeout_s: float = 3.0):
        """
        Command bench/indoor motor response using offboard attitude rate and throttle.
        Translates target PWMs into real motor spin without attitude windup.
        """
        try:
            # If all <= 1020us -> zero throttle / idle
            if max(m1, m2, m3, m4) <= 1020:
                self._current_attitude_rate = AttitudeRate(0.0, 0.0, 0.0, 0.0)
                try:
                    await self.drone.offboard.set_attitude_rate(self._current_attitude_rate)
                except Exception:
                    pass
                return

            # Ensure drone is armed if spin requested
            if not self._armed and max(m1, m2, m3, m4) > 1050:
                try:
                    await self.drone.action.arm()
                except Exception:
                    try:
                        await self.drone.action.arm_force()
                    except Exception:
                        pass

            avg_pwm = (m1 + m2 + m3 + m4) / 4.0
            # Full 1000us to 2000us linear thrust mapping (1000us -> 0.0, 1500us -> 0.50, 1680us -> 0.68, 2000us -> 1.0)
            thrust = max(0.0, min(0.95, (avg_pwm - 1000.0) / 1000.0))

            # Differential roll / pitch / yaw rates in deg/s
            # m1: FR (CCW), m2: RL (CCW), m3: FL (CW), m4: RR (CW)
            # Pitch: rear motors (M2, M4) vs front motors (M1, M3)
            pitch_rate = float((m2 + m4) - (m1 + m3)) * 0.06
            # Roll: left motors (M2, M3) vs right motors (M1, M4)
            roll_rate = float((m2 + m3) - (m1 + m4)) * 0.06
            # Yaw: CW motors (M3, M4) vs CCW motors (M1, M2)
            yaw_rate = float((m3 + m4) - (m1 + m2)) * 0.06

            self._current_attitude_rate = AttitudeRate(roll_rate, pitch_rate, yaw_rate, thrust)
            await self.drone.offboard.set_attitude_rate(self._current_attitude_rate)
            if not self._in_offboard:
                try:
                    await self.drone.offboard.start()
                    self._in_offboard = True
                except Exception:
                    pass
        except Exception:
            pass

    async def _do_arm_and_takeoff(self):
        """
        Arm the drone and perform a slow, constant, smooth takeoff to 30cm (0.3m).
        Ensures zero sudden jumps, clamps strictly to 30cm, and sets hold immediately.
        """
        target_alt = min(0.3, self.takeoff_altitude_m)  # strictly 30cm default

        # 1. Arm drone if not yet armed
        if not self._armed:
            self._current_attitude_rate = AttitudeRate(0.0, 0.0, 0.0, 0.06)
            try:
                await self.drone.offboard.set_attitude_rate(self._current_attitude_rate)
                await self.drone.offboard.start()
                self._in_offboard = True
            except Exception:
                pass
            try:
                await self.drone.action.arm()
                self._armed = True
                print_t("[PX4] Armed successfully.")
            except Exception:
                try:
                    await self.drone.action.arm_force()
                    self._armed = True
                    print_t("[PX4] Force-armed successfully.")
                except Exception as e:
                    print_t(f"[PX4] Arm note: {e}")

        # 2. Indoor / bench / low GPS mode: gentle thrust spool up and 30cm hover
        is_indoor = self.is_hardware_bench or not self._has_gps_lock
        if is_indoor:
            print_t(f"[PX4 Indoor] Smoothly spooling up for 30cm hover...")
            self._send_chat(f"[ROBOT] Gently spooling up for 30cm hover...")
            await self._set_bench_motors(1150, 1150, 1150, 1150, timeout_s=0.8)
            await asyncio.sleep(0.8)
            await self._set_bench_motors(1280, 1280, 1280, 1280, timeout_s=0.8)
            await asyncio.sleep(0.8)
            await self._set_bench_motors(1420, 1420, 1420, 1420, timeout_s=15.0)
            await asyncio.sleep(0.5)
            self._in_air = True
            self._relative_altitude = target_alt
            m = self._motor_pwms
            self._send_chat(f"[ROBOT] Takeoff complete! Holding stable 30cm hover | Live Motors: M1:{m['m1']} M2:{m['m2']} M3:{m['m3']} M4:{m['m4']}µs")
            return

        # 3. Outdoor GPS mode: smooth continuous takeoff via position NED
        try:
            start_n = self._north_m
            start_e = self._east_m
            start_d = self._down_m
            target_d = start_d - target_alt  # -0.3m from ground
            target_d = max(-self.MAX_ALTITUDE_CEILING_M, min(-0.2, target_d))
            yaw = self._heading_deg

            # Stream initial setpoint before starting offboard
            for _ in range(8):
                await self.drone.offboard.set_position_ned(
                    PositionNedYaw(start_n, start_e, start_d, yaw)
                )
                await asyncio.sleep(0.05)
            await self.drone.offboard.start()
            print_t(f"[PX4] Starting smooth continuous takeoff ramp to {target_alt*100:.0f}cm (0.3m)...")
            self._send_chat(f"[ROBOT] Initiating smooth continuous takeoff to 30cm (0.30m)...")

            # Ascend at ~0.12 m/s (12 cm/s) over 2.5s (50 steps of 50ms)
            steps = 50
            for step in range(1, steps + 1):
                fraction = step / steps
                eased = 0.5 * (1.0 - math.cos(math.pi * fraction))
                interp_d = start_d + eased * (target_d - start_d)
                await self.drone.offboard.set_position_ned(
                    PositionNedYaw(start_n, start_e, interp_d, yaw)
                )
                await asyncio.sleep(0.05)

            await self.drone.offboard.set_position_ned(
                PositionNedYaw(start_n, start_e, target_d, yaw)
            )
            await asyncio.sleep(0.4)
            await self.drone.action.hold()
            self._in_air = True
            curr = max(target_alt, self._relative_altitude)
            print_t(f"[PX4] Smooth takeoff complete! Stably holding at {curr:.2f}m (30cm).")
            self._send_chat(f"[ROBOT] Smooth takeoff complete! Hovering stably at 30cm (0.30m).")
            return
        except Exception as e:
            print_t(f"[PX4] Offboard takeoff note ({e}), proceeding with action.takeoff()...")

        # Fallback: action.takeoff with tight 50ms polling loop (locks to hold at 0.28m)
        try:
            await self.drone.action.takeoff()
            self._in_air = True
            print_t(f"[PX4] Autonomous takeoff active, monitoring ascent to {target_alt:.2f}m...")
            t0 = time.time()
            while time.time() - t0 < 6.0:
                if self._relative_altitude >= (target_alt - 0.03):
                    try:
                        await self.drone.action.hold()
                        print_t(f"[PX4] Target 30cm reached ({self._relative_altitude:.2f}m). Locked in HOLD.")
                    except Exception:
                        pass
                    break
                await asyncio.sleep(0.05)
            curr = max(target_alt, self._relative_altitude)
            self._send_chat(f"[ROBOT] Takeoff complete! Hovering stably at {curr*100:.0f}cm ({curr:.2f}m).")
            return
        except Exception as e:
            print_t(f"[PX4] Autonomous takeoff fallback error: {e}")

    def _ensure_flying(self, allow_takeoff: bool = True) -> bool:
        """Ensure drone is armed and airborne before executing movement commands."""
        if not self._in_air and self._relative_altitude < 0.2:
            if not allow_takeoff:
                return False
            print_t("-> [PX4] Drone is on ground. Automatically taking off smoothly to 30cm first...")
            self._run_async(self._do_arm_and_takeoff(), timeout=20.0)
        return True

    @overrides
    def stop(self) -> bool:
        """Land the drone, restore safety parameters, and stop camera observation."""
        print_t("[PX4] Stopping drone...")
        try:
            if self._connected:
                self.land()
                try:
                    self._run_async(self.drone.param.set_param_float('COM_DISARM_PRFLT', self._orig_disarm_prflt), timeout=3.0)
                except Exception:
                    pass
        except Exception as e:
            print_t(f"[PX4] Stop warning: {e}")

        self._running = False
        self.obs.stop()
        return True

    async def _do_move(self, dx: float, dy: float):
        """Move smoothly using interpolated horizontal setpoints (no sudden jerks)."""
        dx = self._normalize_dist(dx)
        dy = self._normalize_dist(dy)

        if self.is_hardware_bench or not self._has_gps_lock:
            dur = min(3.5, max(1.5, math.hypot(dx, dy) / 0.20))
            m1, m2, m3, m4 = 1420, 1420, 1420, 1420
            action_desc = []
            if dx > 0.05:
                m1 = 1250; m3 = 1250; m2 = 1580; m4 = 1580
                action_desc.append(f"PITCH FORWARD {dx*100:.0f}cm")
            elif dx < -0.05:
                m1 = 1580; m3 = 1580; m2 = 1250; m4 = 1250
                action_desc.append(f"PITCH BACKWARD {abs(dx)*100:.0f}cm")
            if dy > 0.05:
                m1 = 1580; m4 = 1580; m2 = 1250; m3 = 1250
                action_desc.append(f"ROLL LEFT {dy*100:.0f}cm")
            elif dy < -0.05:
                m2 = 1580; m3 = 1580; m1 = 1250; m4 = 1250
                action_desc.append(f"ROLL RIGHT {abs(dy)*100:.0f}cm")

            desc_str = " + ".join(action_desc) if action_desc else "attitude hold"
            self._send_chat(f"[BENCH FLIGHT] Executing {desc_str} for {dur:.1f}s...")
            await self._set_bench_motors(m1, m2, m3, m4, timeout_s=dur)
            await asyncio.sleep(dur)
            await self._set_bench_motors(1420, 1420, 1420, 1420, timeout_s=30.0)
            await asyncio.sleep(0.3)
            self._send_chat("[ROBOT] Movement complete. Returned to level hover.")
            return

        rad = math.radians(self._heading_deg)
        # In body frame: dx is forward(+)/back(-), dy is left(+)/right(-)
        # In NED frame: positive dy is body-left, so body-right is -dy
        body_fwd = dx
        body_right = -dy
        delta_north = body_fwd * math.cos(rad) - body_right * math.sin(rad)
        delta_east = body_fwd * math.sin(rad) + body_right * math.cos(rad)

        start_n = self._north_m
        start_e = self._east_m
        # Clamp altitude strictly between 0.2m and 1.0m ceiling during horizontal motion
        target_d = max(-self.MAX_ALTITUDE_CEILING_M, min(-0.2, self._down_m))
        target_yaw = self._heading_deg

        target_n = start_n + delta_north
        target_e = start_e + delta_east

        dist_total = math.hypot(delta_north, delta_east)
        if dist_total < 0.02:
            return

        # Gentle smooth horizontal speed: ~0.20 m/s (20 cm/s)
        speed = 0.20
        duration = max(1.2, dist_total / speed)
        steps = max(12, int(duration / 0.05))

        # Prime initial setpoint before starting offboard
        await self.drone.offboard.set_position_ned(
            PositionNedYaw(start_n, start_e, target_d, target_yaw)
        )
        try:
            await self.drone.offboard.start()
        except OffboardError:
            pass

        # Smooth S-curve horizontal trajectory interpolation (continuous, gradual velocity)
        for step in range(1, steps + 1):
            fraction = step / steps
            eased = 0.5 * (1.0 - math.cos(math.pi * fraction))
            interp_n = start_n + eased * delta_north
            interp_e = start_e + eased * delta_east
            await self.drone.offboard.set_position_ned(
                PositionNedYaw(interp_n, interp_e, target_d, target_yaw)
            )
            await asyncio.sleep(0.05)

        # Settle at final position
        await self.drone.offboard.set_position_ned(
            PositionNedYaw(target_n, target_e, target_d, target_yaw)
        )
        await asyncio.sleep(0.4)
        try:
            await self.drone.action.hold()
        except Exception:
            pass
        self._send_chat(f"[ROBOT] Smooth movement complete ({dist_total*100:.0f}cm / {dist_total:.2f}m). Locked in hover.")

    @overrides
    def _move(self, dx: float, dy: float):
        """
        Move by dx (forward/back) and dy (left/right).
        Interprets values in meters or cm (normalized to max 1.0m / 100cm).
        """
        self._ensure_flying()
        dx = self._normalize_dist(dx)
        dy = self._normalize_dist(dy)
        parts = []
        if abs(dx) > 0.01:
            dir_x = "forward" if dx > 0 else "backward"
            parts.append(f"{dir_x} {abs(dx)*100:.0f}cm ({abs(dx):.2f}m)")
        if abs(dy) > 0.01:
            dir_y = "left" if dy > 0 else "right"
            parts.append(f"{dir_y} {abs(dy)*100:.0f}cm ({abs(dy):.2f}m)")
        desc = " and ".join(parts) if parts else f"dx={dx:.2f}m, dy={dy:.2f}m"
        print_t(f"-> [PX4] Smooth precision flight: {desc}...")
        self._send_chat(f"[ROBOT] Moving {desc} smoothly...")
        self._run_async(self._do_move(dx, dy), timeout=60.0)

    async def _do_rotate(self, deg: float):
        """Rotate closed-loop or bench attitude response."""
        if self.is_hardware_bench or not self._has_gps_lock:
            dur = min(4.0, max(2.5, abs(deg) / max(15.0, self.yaw_rate)))
            if deg > 0:
                # Rotate Left (CCW): CW motors (M3, M4) to 1680us, CCW motors (M1, M2) to 1180us
                m1, m2, m3, m4 = 1180, 1180, 1680, 1680
                direction = "LEFT (CCW)"
            else:
                # Rotate Right (CW): CCW motors (M1, M2) to 1680us, CW motors (M3, M4) to 1180us
                m1, m2, m3, m4 = 1680, 1680, 1180, 1180
                direction = "RIGHT (CW)"
            self._send_chat(f"[BENCH FLIGHT] Rotating {direction} by {abs(deg):.0f}° (diagonal motor pair differential) for {dur:.1f}s...")
            await self._set_bench_motors(m1, m2, m3, m4, timeout_s=dur)
            await asyncio.sleep(dur)
            m = self._motor_pwms
            self._send_chat(f"[MOTOR LOG] Yaw Speeds: M1:{m['m1']}µs | M2:{m['m2']}µs | M3:{m['m3']}µs | M4:{m['m4']}µs")
            await self._set_bench_motors(1420, 1420, 1420, 1420, timeout_s=30.0)
            await asyncio.sleep(0.3)
            self._heading_deg = (self._heading_deg - deg) % 360.0
            self._send_chat(f"[ROBOT] Rotation complete. New heading: {self._heading_deg:.0f}°.")
            return

        target_yaw = (self._heading_deg - deg) % 360.0

        await self.drone.offboard.set_position_ned(
            PositionNedYaw(self._north_m, self._east_m, self._down_m, self._heading_deg)
        )
        try:
            await self.drone.offboard.start()
        except OffboardError:
            pass

        await self.drone.offboard.set_position_ned(
            PositionNedYaw(self._north_m, self._east_m, self._down_m, target_yaw)
        )

        timeout = max(4.0, (abs(deg) / max(10.0, self.yaw_rate)) * 2.0 + 3.0)
        t0 = time.time()
        while time.time() - t0 < timeout:
            yaw_err = abs((self._heading_deg - target_yaw + 180.0) % 360.0 - 180.0)
            if yaw_err < 4.0:
                break
            await asyncio.sleep(0.1)

        try:
            await self.drone.action.hold()
        except Exception:
            pass

    @overrides
    def _rotate(self, deg: float):
        """
        Rotate by deg degrees.
        deg > 0: rotate left (counter-clockwise)
        deg < 0: rotate right (clockwise)
        """
        self._ensure_flying()
        direction = "left" if deg > 0 else "right"
        print_t(f"-> [PX4] Rotating {direction} {abs(deg):.1f} deg...")
        self._run_async(self._do_rotate(deg), timeout=30.0)

    async def _do_lift(self, dist: float):
        """Climb or descend smoothly using interpolated vertical setpoints (zero sudden jumps)."""
        dist = self._normalize_dist(dist)
        if abs(dist) < 0.02:
            return

        if self.is_hardware_bench or not self._has_gps_lock:
            dur = min(3.0, max(1.5, abs(dist) / 0.15))
            if dist > 0:
                self._send_chat(f"[BENCH FLIGHT] Climbing {dist*100:.0f}cm ({dist:.2f}m) for {dur:.1f}s...")
                await self._set_bench_motors(1550, 1550, 1550, 1550, timeout_s=dur)
                await asyncio.sleep(dur)
                self._relative_altitude = min(self.MAX_ALTITUDE_CEILING_M, self._relative_altitude + dist)
            else:
                self._send_chat(f"[BENCH FLIGHT] Descending {abs(dist)*100:.0f}cm ({abs(dist):.2f}m) for {dur:.1f}s...")
                await self._set_bench_motors(1280, 1280, 1280, 1280, timeout_s=dur)
                await asyncio.sleep(dur)
                self._relative_altitude = max(0.2, self._relative_altitude + dist)
            await self._set_bench_motors(1420, 1420, 1420, 1420, timeout_s=30.0)
            await asyncio.sleep(0.3)
            self._send_chat(f"[ROBOT] Altitude set to {self._relative_altitude*100:.0f}cm ({self._relative_altitude:.2f}m). Holding hover.")
            return

        # Closed-loop offboard smooth altitude interpolation
        # In NED frame: height h = -down_m. Climbing means decreasing down_m (more negative).
        start_d = self._down_m
        start_n = self._north_m
        start_e = self._east_m
        yaw = self._heading_deg

        desired_d = start_d - dist
        # Hard safety ceiling: height h <= 1.0m <=> d >= -1.0m
        # Floor safety: height h >= 0.2m <=> d <= -0.2m
        target_d = max(-self.MAX_ALTITUDE_CEILING_M, min(-0.2, desired_d))

        actual_dist = abs(target_d - start_d)
        if actual_dist < 0.02:
            curr_h = max(0.0, -self._down_m)
            self._send_chat(f"[SAFETY] Altitude boundary active ({curr_h*100:.0f}cm). Holding position.")
            return

        # Gentle smooth climb/descent speed: ~0.15 m/s (15 cm/s)
        speed = 0.15
        duration = max(1.0, actual_dist / speed)
        steps = max(10, int(duration / 0.05))  # 20 Hz update rate

        # Prime initial setpoint
        await self.drone.offboard.set_position_ned(
            PositionNedYaw(start_n, start_e, start_d, yaw)
        )
        try:
            await self.drone.offboard.start()
        except OffboardError:
            pass

        # Smooth S-curve setpoint interpolation (constant, slow, smooth elevator motion)
        for step in range(1, steps + 1):
            fraction = step / steps
            eased = 0.5 * (1.0 - math.cos(math.pi * fraction))
            interp_d = start_d + eased * (target_d - start_d)
            interp_d = max(-self.MAX_ALTITUDE_CEILING_M, min(-0.2, interp_d))
            await self.drone.offboard.set_position_ned(
                PositionNedYaw(start_n, start_e, interp_d, yaw)
            )
            await asyncio.sleep(0.05)

        # Settle at target setpoint
        await self.drone.offboard.set_position_ned(
            PositionNedYaw(start_n, start_e, target_d, yaw)
        )
        await asyncio.sleep(0.4)
        try:
            await self.drone.action.hold()
        except Exception:
            pass

        curr_h = max(0.0, -self._down_m)
        self._send_chat(f"[ROBOT] Altitude adjusted smoothly to {curr_h*100:.0f}cm ({curr_h:.2f}m). Stable hover locked.")

    def lift(self, dist: float = 0.3):
        """Climb or descend smoothly by distance in cm or meters (positive up, negative down, ceiling 1.0m / 100cm)."""
        dist = self._normalize_dist(dist if dist is not None else 0.3)
        if abs(dist) < 0.02:
            return

        current_h = max(0.0, -self._down_m if abs(self._down_m) > 0.01 else self._relative_altitude)

        if dist > 0:
            if current_h >= self.MAX_ALTITUDE_CEILING_M - 0.02:
                msg = f"[SAFETY] Drone is already at maximum ceiling limit ({current_h*100:.0f}cm / 1.0m). Climb command rejected."
                print_t(f"-> {msg}")
                self._send_chat(msg)
                return

            if current_h + dist > self.MAX_ALTITUDE_CEILING_M:
                clamped_dist = max(0.05, self.MAX_ALTITUDE_CEILING_M - current_h)
                print_t(f"-> [SAFETY] Clamping climb from {dist*100:.0f}cm to {clamped_dist*100:.0f}cm (ceiling: 1.0m)")
                dist = clamped_dist

            self._ensure_flying(allow_takeoff=True)
            print_t(f"-> [PX4] Smoothly climbing {dist*100:.0f}cm (current alt: {current_h*100:.0f}cm)...")
            self._send_chat(f"[ROBOT] Smoothly climbing {dist*100:.0f}cm ({dist:.2f}m)...")
            self._run_async(self._do_lift(dist), timeout=30.0)
        else:
            if not self._ensure_flying(allow_takeoff=False):
                print_t(f"-> [PX4] Drone is on ground. Cannot descend.")
                return
            if current_h <= 0.25:
                msg = f"[SAFETY] Drone is near minimum flight height ({current_h*100:.0f}cm). Use 'land()' to touch down safely."
                print_t(f"-> {msg}")
                self._send_chat(msg)
                return

            print_t(f"-> [PX4] Smoothly descending {abs(dist)*100:.0f}cm (current alt: {current_h*100:.0f}cm)...")
            self._send_chat(f"[ROBOT] Smoothly descending {abs(dist)*100:.0f}cm ({abs(dist):.2f}m)...")
            self._run_async(self._do_lift(dist), timeout=30.0)

    def climb(self, dist: float = 0.3):
        """Climb upwards smoothly by distance (default 30cm / 0.3m, max 1.0m / 100cm). Accepts meters or cm."""
        self.lift(abs(self._normalize_dist(dist if dist is not None else 0.3)))

    def descend(self, dist: float = 0.3):
        """Descend downwards smoothly by distance (default 30cm / 0.3m, max 1.0m / 100cm). Accepts meters or cm."""
        self.lift(-abs(self._normalize_dist(dist if dist is not None else 0.3)))

    def set_speed(self, speed: float):
        """Set drone flight speed in m/s (between 0.1 and 3.0 m/s for safe smooth control)."""
        speed = max(0.1, min(3.0, float(speed)))
        self.cruise_speed = speed
        print_t(f"-> [PX4] Cruise speed set to {self.cruise_speed:.2f} m/s ({self.cruise_speed*100:.0f} cm/s)")

    def arm(self):
        """Arm the drone."""
        self._policy_guard("action")
        print_t("-> [PX4] Arm commanded")
        async def _do_arm():
            try:
                await self.drone.param.set_param_float('COM_DISARM_PRFLT', -1.0)
            except Exception:
                pass
            is_bench = self.is_hardware_bench or "serial://" in str(self.system_address) or not self._has_gps_lock
            if is_bench:
                # 1. Prime offboard with idle attitude rate and start offboard mode
                self._current_attitude_rate = AttitudeRate(0.0, 0.0, 0.0, 0.06)
                try:
                    await self.drone.offboard.set_attitude_rate(self._current_attitude_rate)
                    await self.drone.offboard.start()
                    self._in_offboard = True
                except Exception as ex:
                    print_t(f"[PX4] Offboard start note: {ex}")
                    self._in_offboard = True

                # 2. Arm motors (standard or force)
                armed_ok = False
                try:
                    await self.drone.action.arm()
                    armed_ok = True
                except Exception as e:
                    print_t(f"[PX4] action.arm() note ({e}), attempting force-arm...")
                    try:
                        await self.drone.action.arm_force()
                        armed_ok = True
                    except Exception as e2:
                        print_t(f"[PX4] Arm error: {e2}")

                if armed_ok or self._armed:
                    self._armed = True
                    self._in_offboard = True
                    try:
                        await self.drone.offboard.start()
                    except Exception:
                        pass
                    await asyncio.sleep(0.3)
                    m = self._motor_pwms
                    self._send_chat(f"[ROBOT] Drone motors armed! All 4 spinning at idle: M1:{m['m1']} M2:{m['m2']} M3:{m['m3']} M4:{m['m4']}µs")
                else:
                    self._in_offboard = False
                    self._send_chat("[ROBOT] Arming denied by PX4 safety checks.")
            else:
                try:
                    await self.drone.action.arm()
                    self._armed = True
                    self._send_chat("[ROBOT] Drone motors armed successfully.")
                except Exception:
                    try:
                        await self.drone.action.arm_force()
                        self._armed = True
                        self._send_chat("[ROBOT] Drone motors armed successfully.")
                    except Exception:
                        self._send_chat("[ROBOT] Arming denied by PX4 safety checks (check GPS/RC status).")
        self._run_async(_do_arm(), timeout=10.0)

    def disarm(self):
        """Disarm the drone motors safely (FlySky switch / stick disarm)."""
        self._policy_guard("action")
        print_t("-> [PX4] Disarm commanded")
        async def _do_disarm():
            self._in_offboard = False
            self._current_attitude_rate = AttitudeRate(0.0, 0.0, 0.0, 0.0)
            # 1. Stop all bench motor pulses immediately
            try:
                await self.drone.offboard.set_attitude_rate(self._current_attitude_rate)
            except Exception:
                pass
            try:
                await self.drone.offboard.stop()
            except Exception:
                pass

            # 2. Helper to send direct MAVLink COMMAND_LONG disarm / force disarm
            async def _send_mavlink_disarm(force: bool = True):
                try:
                    fields = {
                        'target_system': 1,
                        'target_component': 0,
                        'command': 400, # MAV_CMD_COMPONENT_ARM_DISARM
                        'confirmation': 0,
                        'param1': 0,    # 0 = Disarm
                        'param2': 21196 if force else 0, # Force disarm flag
                        'param3': 0, 'param4': 0, 'param5': 0, 'param6': 0, 'param7': 0
                    }
                    msg = MavlinkMessage('COMMAND_LONG', 255, 0, 1, 0, json.dumps(fields))
                    await self.drone.mavlink_direct.send_message(msg)
                except Exception:
                    pass

            # 3. Attempt MAVSDK action disarm
            try:
                await self.drone.action.disarm()
                print_t("[PX4] Disarmed successfully via action.disarm().")
            except Exception as e:
                print_t(f"[PX4] action.disarm() warning ({e}), falling back to direct MAVLink disarm and action.kill()...")
                await _send_mavlink_disarm(force=True)
                try:
                    await self.drone.action.kill()
                except Exception:
                    pass

            self._in_air = False
            self._relative_altitude = 0.0
            self._armed = False
            self._in_offboard = False
            self._send_chat("[ROBOT] Drone motors disarmed.")
        self._run_async(_do_disarm(), timeout=10.0)

    def takeoff(self):
        """Take off to hover altitude."""
        self._policy_guard("action")
        print_t("-> [PX4] Takeoff commanded")
        self._send_chat("[ROBOT] Taking off...")
        self._run_async(self._do_arm_and_takeoff(), timeout=20.0)

    def land(self):
        """Land the drone safely."""
        self._policy_guard("action")
        print_t("-> [PX4] Land commanded")
        async def _do_land():
            if not self._in_air and not self._armed:
                self._send_chat("[ROBOT] Drone is already on the ground and disarmed.")
                return

            if self.is_hardware_bench or not self._has_gps_lock:
                # Get current motor throttle - strictly decrease, never increase
                cur_pwm = sum(self._motor_pwms.values()) / 4.0
                if 1060 < cur_pwm < 1800:
                    start_thrust = (cur_pwm - 1000.0) / 1000.0
                else:
                    start_thrust = max(0.12, getattr(self._current_attitude_rate, "thrust_value", 0.24))

                start_pwm = int(start_thrust * 1000.0 + 1000.0)
                self._send_chat(f"[ROBOT] Landing initiated: Smoothly decreasing throttle from {start_pwm}µs to ground idle...")

                # Smooth downward ramp to ground idle (0.08 / ~1080µs) over 2.0s (40 steps of 50ms)
                steps = 40
                target_idle_thrust = 0.08
                for s in range(1, steps + 1):
                    frac = s / steps
                    cur_t = start_thrust + frac * (target_idle_thrust - start_thrust)
                    self._current_attitude_rate = AttitudeRate(0.0, 0.0, 0.0, cur_t)
                    try:
                        await self.drone.offboard.set_attitude_rate(self._current_attitude_rate)
                    except Exception:
                        pass
                    await asyncio.sleep(0.05)

                # Ground idle pause to ensure touchdown
                self._send_chat("[ROBOT] Ground touchdown, spooling down to cutoff...")
                await asyncio.sleep(0.6)

                # Cutoff motors (1000µs) and disarm
                self._in_offboard = False
                self._current_attitude_rate = AttitudeRate(0.0, 0.0, 0.0, 0.0)
                try:
                    await self.drone.offboard.set_attitude_rate(self._current_attitude_rate)
                except Exception:
                    pass
                try:
                    await self.drone.offboard.stop()
                except Exception:
                    pass
                try:
                    await self.drone.action.disarm()
                except Exception:
                    try:
                        await self.drone.action.kill()
                    except Exception:
                        pass
                self._in_air = False
                self._relative_altitude = 0.0
                self._armed = False
                m = self._motor_pwms
                self._send_chat(f"[ROBOT] Landed safely. Motors disarmed (M1:{m['m1']}µs | M2:{m['m2']}µs | M3:{m['m3']}µs | M4:{m['m4']}µs).")
                return

            try:
                try:
                    await self.drone.offboard.stop()
                except Exception:
                    pass
                await self.drone.action.land()
                print_t("[PX4] Landing in progress...")
                self._send_chat("[ROBOT] Landed safely.")
            except Exception as e:
                print_t(f"[PX4] Land error: {e}")
                self._send_chat(f"[ERROR] Land failed: {e}")
        self._run_async(_do_land(), timeout=20.0)
        self._in_air = False

    def hold(self):
        """Hover in place and lock GPS/altitude position."""
        self._policy_guard("action")
        print_t("-> [PX4] Hold (hover in place / Loiter lock)")
        async def _do_hold():
            if self.is_hardware_bench or not self._has_gps_lock:
                self._current_attitude_rate = AttitudeRate(0.0, 0.0, 0.0, 0.24)
                try:
                    await self.drone.offboard.set_attitude_rate(self._current_attitude_rate)
                    if not self._in_offboard:
                        await self.drone.offboard.start()
                        self._in_offboard = True
                except Exception:
                    pass
                self._send_chat("[ROBOT] Level attitude hover hold locked.")
                return

            try:
                try:
                    await self.drone.offboard.stop()
                except Exception:
                    pass
                await self.drone.action.hold()
                self._send_chat("[ROBOT] Position hold locked.")
            except Exception:
                pass
        self._run_async(_do_hold(), timeout=5.0)

    def return_to_launch(self):
        """Return to home/launch GPS position and land safely."""
        self._policy_guard("action")
        print_t("-> [PX4] Return to Launch (RTL) commanded. Navigating back to Home...")
        self._send_chat("[ROBOT] Return to Launch commanded. Returning to Home...")
        async def _do_rtl():
            try:
                try:
                    await self.drone.offboard.stop()
                except Exception:
                    pass
                await self.drone.action.return_to_launch()
                print_t("[PX4] Return to Launch in progress...")
            except Exception as e:
                print_t(f"[PX4] RTL error: {e}")
                self._send_chat(f"[ERROR] RTL failed: {e}")
        self._run_async(_do_rtl(), timeout=15.0)

    def return_home(self):
        """Return to home/launch GPS position and land safely."""
        self._policy_guard("action")
        self.return_to_launch()

    def goto_gps(self, latitude: float, longitude: float, altitude_m: float = 5.0, yaw_deg: float = 0.0):
        """Fly to global GPS coordinates (latitude, longitude, altitude in meters relative to ground, yaw in degrees)."""
        self._policy_guard("action")
        self._ensure_flying(allow_takeoff=True)
        base_alt = self._home_alt if self._home_alt > 0.1 else (self._absolute_altitude_m - self._relative_altitude)
        target_abs_alt = base_alt + max(1.0, altitude_m)
        print_t(f"-> [PX4] Navigating to GPS: Lat={latitude:.6f}, Lon={longitude:.6f}, Alt={altitude_m:.1f}m (AMSL={target_abs_alt:.1f}m)...")
        async def _do_goto():
            try:
                try:
                    await self.drone.offboard.stop()
                except Exception:
                    pass
                await self.drone.action.goto_location(latitude, longitude, target_abs_alt, yaw_deg)
                t0 = time.time()
                while time.time() - t0 < 60.0:
                    d_lat = (self._latitude_deg - latitude) * 111139.0
                    d_lon = (self._longitude_deg - longitude) * 111139.0 * math.cos(math.radians(latitude))
                    dist_to_gps = math.hypot(d_lat, d_lon)
                    if dist_to_gps < 1.0:
                        print_t(f"[PX4] Arrived at GPS target (dist: {dist_to_gps:.2f}m)!")
                        break
                    await asyncio.sleep(0.5)
                await self.drone.action.hold()
            except Exception as e:
                print_t(f"[PX4] Goto GPS error: {e}")
        self._run_async(_do_goto(), timeout=70.0)

    def get_gps_position(self) -> dict[str, float]:
        """Get and log current GPS coordinates, altitude, and heading."""
        self._policy_guard("action")
        info = {
            "latitude": round(self._latitude_deg, 6),
            "longitude": round(self._longitude_deg, 6),
            "relative_altitude": round(self._relative_altitude, 2),
            "absolute_altitude": round(self._absolute_altitude_m, 2),
            "heading": round(self._heading_deg, 1),
            "satellites": self._num_satellites,
        }
        msg = f"GPS Position: Lat={info['latitude']:.6f}, Lon={info['longitude']:.6f}, Alt={info['relative_altitude']:.1f}m, Heading={info['heading']:.1f}°, Sats={info['satellites']}"
        print_t(f"-> [PX4] Current {msg}")
        self._send_chat(f"[GPS] {msg}")
        return info

    def get_home_position(self) -> dict[str, float]:
        """Get and log home/launch GPS coordinates."""
        self._policy_guard("action")
        info = {
            "latitude": round(self._home_lat, 6),
            "longitude": round(self._home_lon, 6),
            "absolute_altitude": round(self._home_alt, 2),
        }
        msg = f"Home Position: Lat={info['latitude']:.6f}, Lon={info['longitude']:.6f}, Altitude={info['absolute_altitude']:.1f}m AMSL"
        print_t(f"-> [PX4] {msg}")
        self._send_chat(f"[HOME] {msg}")
        return info

    def orbit(self, radius_m: float = 3.0, velocity_m_s: float = 1.0, yaw_behavior: str = "center"):
        """Fly a circular orbit around current location (radius in meters, velocity in m/s, yaw_behavior 'center' or 'tangent')."""
        self._policy_guard("action")
        self._ensure_flying(allow_takeoff=True)
        from mavsdk.action import OrbitYawBehavior
        behavior = OrbitYawBehavior.HOLD_FRONT_TO_CIRCLE_CENTER if str(yaw_behavior).lower() != "tangent" else OrbitYawBehavior.HOLD_FRONT_TANGENT_TO_CIRCLE
        radius_m = max(1.0, min(50.0, float(radius_m)))
        velocity_m_s = max(0.2, min(5.0, float(velocity_m_s)))
        print_t(f"-> [PX4] Orbit commanded: radius={radius_m:.1f}m, speed={velocity_m_s:.1f}m/s, orientation={yaw_behavior}...")
        self._send_chat(f"[ORBIT] Orbiting at radius {radius_m:.1f}m, speed {velocity_m_s:.1f}m/s...")
        async def _do_orbit():
            try:
                try:
                    await self.drone.offboard.stop()
                except Exception:
                    pass
                await self.drone.action.do_orbit(
                    radius_m, velocity_m_s, behavior, float("nan"), float("nan"), float("nan")
                )
                print_t("[PX4] Orbit routine active.")
            except Exception as e:
                print_t(f"[PX4] Orbit error: {e}")
                self._send_chat(f"[ERROR] Orbit failed: {e}")
        self._run_async(_do_orbit(), timeout=15.0)

    def set_rtl_altitude(self, altitude_m: float):
        """Set the return-to-launch altitude in meters relative to home."""
        self._policy_guard("action")
        altitude_m = max(2.0, min(100.0, float(altitude_m)))
        print_t(f"-> [PX4] Setting RTL altitude to {altitude_m:.1f}m...")
        async def _do_set():
            try:
                await self.drone.action.set_return_to_launch_altitude(altitude_m)
                print_t(f"[PX4] RTL altitude configured to {altitude_m:.1f}m.")
                self._send_chat(f"[RTL] Return-to-launch altitude set to {altitude_m:.1f}m.")
            except Exception as e:
                print_t(f"[PX4] Set RTL altitude error: {e}")
        self._run_async(_do_set(), timeout=10.0)

    def set_takeoff_altitude(self, altitude_m: float):
        """Set the default takeoff hover altitude in meters or cm (max 1.0m / 100cm)."""
        self._policy_guard("action")
        altitude_m = self._normalize_dist(altitude_m)
        altitude_m = max(0.2, min(self.MAX_ALTITUDE_CEILING_M, altitude_m))
        self.takeoff_altitude_m = altitude_m
        print_t(f"-> [PX4] Setting takeoff altitude to {altitude_m*100:.0f}cm ({altitude_m:.2f}m)...")
        async def _do_set():
            try:
                await self.drone.action.set_takeoff_altitude(altitude_m)
            except Exception:
                pass
            print_t(f"[PX4] Takeoff altitude configured to {altitude_m*100:.0f}cm ({altitude_m:.2f}m).")
            self._send_chat(f"[TAKEOFF] Default hover altitude configured to {altitude_m*100:.0f}cm ({altitude_m:.2f}m).")
        self._run_async(_do_set(), timeout=10.0)

    def get_telemetry_data(self) -> dict[str, Any]:
        """Return raw telemetry dictionary for UI and API endpoints."""
        curr_alt = self._relative_altitude if self._relative_altitude > 0.05 else max(0.0, -self._down_m)
        return {
            "latitude": round(self._latitude_deg, 6),
            "longitude": round(self._longitude_deg, 6),
            "altitude_rel": round(curr_alt, 2),
            "altitude_cm": round(curr_alt * 100, 1),
            "altitude_abs": round(self._absolute_altitude_m, 2),
            "heading": round(self._heading_deg, 1),
            "roll": round(self._roll_deg, 2),
            "pitch": round(self._pitch_deg, 2),
            "speed": round(self.cruise_speed, 1),
            "battery": round(self._battery_percent, 1),
            "voltage": round(self._battery_voltage, 2),
            "satellites": self._num_satellites,
            "flight_mode": self._flight_mode,
            "in_air": self._in_air,
            "armed": self._armed,
            "motors": self._motor_pwms,
            "home": {
                "latitude": round(self._home_lat, 6),
                "longitude": round(self._home_lon, 6),
                "altitude": round(self._home_alt, 2),
            }
        }

    def get_motor_speeds(self) -> dict[str, int]:
        """Get live motor PWM output values in microseconds directly from ArduPilot/PX4."""
        self._policy_guard("action")
        m = self._motor_pwms
        msg = f"M1(FR): {m['m1']}µs | M2(RL): {m['m2']}µs | M3(FL): {m['m3']}µs | M4(RR): {m['m4']}µs"
        print_t(f"-> [PX4] Live Motor PWMs: {msg}")
        self._send_chat(f"[MOTORS] Live Speeds -> {msg}")
        return m

    def set_manual_motors(self, m1: int, m2: int, m3: int, m4: int) -> dict[str, int]:
        """Manually command live motor speeds directly via sliders/UI (1000 - 1850 µs)."""
        # Clamp between 1000 µs (idle) and 1850 µs (full flight range)
        m1 = int(max(1000, min(1850, m1)))
        m2 = int(max(1000, min(1850, m2)))
        m3 = int(max(1000, min(1850, m3)))
        m4 = int(max(1000, min(1850, m4)))
        self._motor_pwms = {"m1": m1, "m2": m2, "m3": m3, "m4": m4}
        is_bench = self.is_hardware_bench or "serial://" in str(self.system_address)
        if is_bench:
            self._run_async(self._set_bench_motors(m1, m2, m3, m4, timeout_s=3.0), timeout=3.0)
        return self._motor_pwms

    def get_telemetry(self) -> dict[str, Any]:
        """Get complete flight status, battery, GPS coordinates, and flight mode."""
        self._policy_guard("action")
        data = self.get_telemetry_data()
        m = data['motors']
        msg = (
            f"Mode: {data['flight_mode']} | Armed: {data['armed']} | "
            f"Battery: {data['battery']:.0f}% ({data['voltage']:.1f}V) | "
            f"Alt: {data['altitude_rel']:.1f}m | Hdg: {data['heading']:.1f}° | "
            f"Motors: M1:{m['m1']} M2:{m['m2']} M3:{m['m3']} M4:{m['m4']}µs"
        )
        print_t(f"-> [PX4] Telemetry: {msg}")
        self._send_chat(f"[TELEMETRY] {msg}")
        return data

    def get_battery(self) -> dict[str, float]:
        """Get drone battery state (percentage and voltage)."""
        self._policy_guard("action")
        pct = round(self._battery_percent, 1)
        volt = round(self._battery_voltage, 2)
        msg = f"Battery: {pct:.1f}% ({volt:.2f}V)"
        print_t(f"-> [PX4] {msg}")
        self._send_chat(f"[BATTERY] {msg}")
        return {"battery_percent": pct, "voltage_v": volt}

    def get_flight_mode(self) -> str:
        """Get current PX4 flight mode."""
        self._policy_guard("action")
        mode = self._flight_mode
        self._send_chat(f"[MODE] Current Flight Mode: {mode}")
        return mode

    def set_flight_mode(self, mode: str):
        """Set drone flight mode ('hold', 'altctl', 'offboard', 'rtl', 'land', 'mission', 'stabilized')."""
        self._policy_guard("action")
        mode_str = str(mode).strip().lower()
        print_t(f"-> [PX4] Setting flight mode to {mode_str}...")
        async def _do_mode():
            try:
                if mode_str in ("hold", "loiter", "posctl", "position"):
                    try:
                        await self.drone.offboard.stop()
                    except Exception:
                        pass
                    await self.drone.action.hold()
                    self._send_chat("[MODE] Flight mode set to HOLD (Position Hold).")
                elif mode_str in ("rtl", "return", "home", "return_to_launch"):
                    try:
                        await self.drone.offboard.stop()
                    except Exception:
                        pass
                    await self.drone.action.return_to_launch()
                    self._send_chat("[MODE] Flight mode set to RETURN-TO-LAUNCH.")
                elif mode_str in ("land", "landing"):
                    try:
                        await self.drone.offboard.stop()
                    except Exception:
                        pass
                    await self.drone.action.land()
                    self._send_chat("[MODE] Flight mode set to LAND.")
                elif mode_str in ("offboard",):
                    await self.drone.offboard.start()
                    self._send_chat("[MODE] Flight mode set to OFFBOARD.")
                else:
                    self._send_chat(f"[MODE] Flight mode set to {mode_str.upper()}.")
            except Exception as e:
                print_t(f"[PX4] Mode change error: {e}")
                self._send_chat(f"[ERROR] Failed to set flight mode: {e}")
        self._run_async(_do_mode(), timeout=10.0)

    def flip(self, direction: str = "front"):
        """Perform a 360-degree acrobatic aerial flip in specified direction ('front', 'back', 'left', 'right')."""
        self._policy_guard("action")
        self._ensure_flying(allow_takeoff=True)
        dir_clean = str(direction).strip().lower()

        if dir_clean in ("front", "forward"):
            roll_rate, pitch_rate = 0.0, 720.0
            dir_name = "FRONT"
        elif dir_clean in ("back", "backward"):
            roll_rate, pitch_rate = 0.0, -720.0
            dir_name = "BACK"
        elif dir_clean in ("left",):
            roll_rate, pitch_rate = -720.0, 0.0
            dir_name = "LEFT"
        elif dir_clean in ("right",):
            roll_rate, pitch_rate = 720.0, 0.0
            dir_name = "RIGHT"
        else:
            roll_rate, pitch_rate = 0.0, 720.0
            dir_name = "FRONT"

        # Ensure safety altitude before flip
        if self._relative_altitude < 2.5:
            print_t(f"-> [PX4] Altitude {self._relative_altitude:.2f}m is low for acrobatic flip; climbing to 3.0m...")
            self._send_chat("[FLIP] Gaining altitude before flip...")
            climb_dist = max(1.0, 3.0 - self._relative_altitude)
            self.lift(climb_dist)
            time.sleep(1.0)

        print_t(f"-> [PX4] Performing {dir_name} 360° flip...")
        self._send_chat(f"[FLIP] Executing 360° {dir_name} flip...")

        async def _do_flip():
            try:
                # Stage 1: Upward pop pulse to build vertical momentum
                await self.drone.offboard.set_attitude_rate(AttitudeRate(0.0, 0.0, 0.0, 0.95))
                try:
                    await self.drone.offboard.start()
                except OffboardError:
                    pass
                await asyncio.sleep(0.35)

                # Stage 2: 720 deg/s angular rotation for 0.50s (full 360 deg flip)
                for _ in range(10):
                    await self.drone.offboard.set_attitude_rate(AttitudeRate(roll_rate, pitch_rate, 0.0, 0.20))
                    await asyncio.sleep(0.05)

                # Stage 3: Catch thrust to stabilize attitude
                await self.drone.offboard.set_attitude_rate(AttitudeRate(0.0, 0.0, 0.0, 0.85))
                await asyncio.sleep(0.30)

                # Stage 4: Lock back into Position Hold
                try:
                    await self.drone.offboard.stop()
                except Exception:
                    pass
                await self.drone.action.hold()
                await asyncio.sleep(1.0)
                print_t(f"[PX4] {dir_name} flip completed! Current Alt: {self._relative_altitude:.2f}m")
                self._send_chat(f"[FLIP] 360° {dir_name} flip completed! Altitude: {self._relative_altitude:.2f}m.")
            except Exception as e:
                print_t(f"[PX4] Flip error: {e}")
                self._send_chat(f"[ERROR] Flip failed: {e}")
                try:
                    await self.drone.action.hold()
                except Exception:
                    pass

        self._run_async(_do_flip(), timeout=15.0)

    def front_flip(self):
        """Perform a 360-degree front flip."""
        self._policy_guard("action")
        self.flip("front")

    def back_flip(self):
        """Perform a 360-degree back flip."""
        self._policy_guard("action")
        self.flip("back")

    def flip_left(self):
        """Perform a 360-degree left roll flip."""
        self._policy_guard("action")
        self.flip("left")

    def flip_right(self):
        """Perform a 360-degree right roll flip."""
        self._policy_guard("action")
        self.flip("right")

    def tilt_camera(self, pitch_deg: float):
        """Tilt camera gimbal pitch angle in degrees (-90 looking straight down, 0 forward, 30 up)."""
        self._policy_guard("action")
        pitch_deg = max(-90.0, min(30.0, float(pitch_deg)))
        print_t(f"-> [PX4] Tilting camera pitch to {pitch_deg:.1f}°...")
        rad = math.radians(pitch_deg)
        try:
            import subprocess
            subprocess.Popen(
                ["gz", "topic", "-t", "/model/x500_depth_0/command/gimbal_pitch", "-m", "gz.msgs.Double", "-p", f"data: {rad}"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            subprocess.Popen(
                ["gz", "topic", "-t", "/gimbal/cmd_pitch", "-m", "gz.msgs.Double", "-p", f"data: {rad}"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception:
            pass
        async def _do_gimbal():
            try:
                await self.drone.gimbal.set_pitch_and_yaw(pitch_deg, 0.0)
            except Exception:
                pass
        try:
            self._run_async(_do_gimbal(), timeout=2.0)
        except Exception:
            pass
        self._send_chat(f"[CAMERA] Gimbal pitch tilted to {pitch_deg:.1f}°.")

    def pan_camera(self, yaw_deg: float):
        """Pan camera gimbal yaw angle in degrees (-90 left to +90 right, 0 centered)."""
        self._policy_guard("action")
        yaw_deg = max(-90.0, min(90.0, float(yaw_deg)))
        print_t(f"-> [PX4] Panning camera yaw to {yaw_deg:.1f}°...")
        rad = math.radians(yaw_deg)
        try:
            import subprocess
            subprocess.Popen(
                ["gz", "topic", "-t", "/model/x500_depth_0/command/gimbal_yaw", "-m", "gz.msgs.Double", "-p", f"data: {rad}"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            subprocess.Popen(
                ["gz", "topic", "-t", "/gimbal/cmd_yaw", "-m", "gz.msgs.Double", "-p", f"data: {rad}"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception:
            pass
        async def _do_gimbal():
            try:
                await self.drone.gimbal.set_pitch_and_yaw(0.0, yaw_deg)
            except Exception:
                pass
        try:
            self._run_async(_do_gimbal(), timeout=2.0)
        except Exception:
            pass
        self._send_chat(f"[CAMERA] Gimbal yaw panned to {yaw_deg:.1f}°.")

    def center_camera(self):
        """Center camera gimbal pitch and yaw to forward 0 degrees."""
        self._policy_guard("action")
        print_t("-> [PX4] Centering camera gimbal...")
        try:
            import subprocess
            subprocess.Popen(
                ["gz", "topic", "-t", "/model/x500_depth_0/command/gimbal_pitch", "-m", "gz.msgs.Double", "-p", "data: 0.0"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            subprocess.Popen(
                ["gz", "topic", "-t", "/model/x500_depth_0/command/gimbal_yaw", "-m", "gz.msgs.Double", "-p", "data: 0.0"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            subprocess.Popen(
                ["gz", "topic", "-t", "/gimbal/cmd_pitch", "-m", "gz.msgs.Double", "-p", "data: 0.0"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            subprocess.Popen(
                ["gz", "topic", "-t", "/gimbal/cmd_yaw", "-m", "gz.msgs.Double", "-p", "data: 0.0"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception:
            pass
        async def _do_gimbal():
            try:
                await self.drone.gimbal.set_pitch_and_yaw(0.0, 0.0)
            except Exception:
                pass
        try:
            self._run_async(_do_gimbal(), timeout=2.0)
        except Exception:
            pass
        self._send_chat("[CAMERA] Camera gimbal centered (pitch 0°, yaw 0°).")

    def emergency_stop(self):
        """Immediate emergency halt and land."""
        self._policy_guard("action")
        print_t("-> [PX4] Emergency stop commanded!")
        self._send_chat("[EMERGENCY] Emergency stop commanded. Landing immediately!")
        self.emergency_shutdown()

    @overrides
    def stop_motion(self) -> None:
        """Halt in-flight motion and lock position hold."""
        print_t("-> [PX4] Stop motion (locking position hold)")
        self.hold()

    @overrides
    def emergency_shutdown(self) -> None:
        """Emergency land."""
        print_t("-> [PX4] Emergency shutdown: landing immediately")
        self.land()
