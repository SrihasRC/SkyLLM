import asyncio
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

DEFAULT_GAZEBO_TOPIC = "/world/default/model/x500_depth_0/link/camera_link/sensor/IMX214/image"
DEFAULT_MAVSDK_ADDR = "udpin://0.0.0.0:14540"
DEFAULT_CRUISE_SPEED = 1.0       # m/s
DEFAULT_YAW_RATE = 30.0          # deg/s
DEFAULT_CLIMB_SPEED = 0.8        # m/s


class PX4Observation(RobotObservation):
    """
    Observation provider for the PX4 simulation in Gazebo.
    Captures video frames directly from Gazebo Harmonic's IMX214 camera topic,
    stores them as PIL Images for the TypeFly web UI, and runs YOLO object detection.
    """
    def __init__(self, robot_info: RobotInfo, rate: int = 10):
        super().__init__(robot_info, rate)
        self.yolo_client = YoloClient(robot_info)
        extra = robot_info.extra or {}
        self.gazebo_topic = extra.get("gazebo_topic", DEFAULT_GAZEBO_TOPIC)

        self._frame_lock = threading.Lock()
        self._last_frame_time = 0.0
        self._gz_node: Optional[Any] = None
        self._ros_node: Optional[Any] = None
        self._ros_thread: Optional[threading.Thread] = None

        # Create an initial placeholder image so web UI doesn't show blank/broken frame
        self._placeholder = self._create_placeholder("Connecting to PX4 Simulation Camera...")
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
                self._ros_node = RosNode("typefly_px4_cam")

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
        """Process latest camera frame through YOLO client."""
        await self.yolo_client.detect(image)

    @overrides
    def fetch_processed_result(self) -> dict[str, Any]:
        """Fetch latest detected objects from YOLO client."""
        _, object_list = self.yolo_client.latest_result
        return {
            "yolo": object_list
        }


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
        self._flight_mode = "UNKNOWN"
        self._armed = False
        self._battery_percent = 100.0
        self._battery_voltage = 16.0
        self._num_satellites = 0
        self._home_lat = 0.0
        self._home_lon = 0.0
        self._home_alt = 0.0
        self.max_distance = 50.0
        self._running = True

        # Dedicated background asyncio event loop for MAVSDK commands
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()

        obs = PX4Observation(robot_info)
        super().__init__(robot_info, obs)

        # Register drone skills
        self.skillset.add_skill(self.arm, "Arm the drone")
        self.skillset.add_skill(self.disarm, "Disarm the drone motors safely")
        self.skillset.add_skill(self.takeoff, "Take off to hover altitude")
        self.skillset.add_skill(self.land, "Land the drone safely")
        self.skillset.add_skill(self.lift, "Move up or down by distance in meters (positive up, negative down)")
        self.skillset.add_skill(self.climb, "Climb upwards by distance in meters (positive number)")
        self.skillset.add_skill(self.descend, "Descend downwards by distance in meters (positive number)")
        self.skillset.add_skill(self.hold, "Hover in place")
        self.skillset.add_skill(self.set_speed, "Set drone flight speed in meters per second (0.5 to 5.0 m/s)")
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

    @overrides
    def start(self) -> bool:
        """Connect to PX4 SITL, start telemetry monitoring, and start camera observation."""
        print_t(f"[PX4] Connecting to drone at {self.system_address}...")
        try:
            self._run_async(self._connect_drone(), timeout=25.0)
            self.obs.start()
            print_t("[PX4] Drone connected and observation started successfully!")
            return True
        except Exception as e:
            print_t(f"[PX4] Failed to start: {e}")
            self.obs.start()
            return False

    async def _connect_drone(self):
        await self.drone.connect(system_address=self.system_address)
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print_t("[PX4] MAVSDK connected to PX4 SITL!")
                self._connected = True
                break

        # Start background telemetry monitoring loop
        asyncio.run_coroutine_threadsafe(self._telemetry_watch(), self._loop)

        # Wait for position lock
        print_t("[PX4] Checking telemetry position lock...")
        async for health in self.drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                print_t("[PX4] Position estimate healthy!")
                break
            await asyncio.sleep(0.5)

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
            except Exception:
                pass

        async def _home_loop():
            try:
                async for home in self.drone.telemetry.home():
                    self._home_lat = home.latitude_deg
                    self._home_lon = home.longitude_deg
                    self._home_alt = home.absolute_altitude_m
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
            _home_loop()
        )

    async def _do_arm_and_takeoff(self):
        """Arm the drone and initiate takeoff to hover altitude."""
        try:
            await self.drone.action.arm()
            print_t("[PX4] Armed successfully.")
        except Exception as e:
            print_t(f"[PX4] Arm error: {e}")

        try:
            await self.drone.action.takeoff()
            print_t("[PX4] Takeoff commanded! Climbing to hover altitude...")
            self._in_air = True
        except Exception as e:
            print_t(f"[PX4] Takeoff warning: {e}")

        # Wait until airborne and hovering stably (around 2.5m)
        t0 = time.time()
        while time.time() - t0 < 8.0:
            if self._relative_altitude >= 1.5:
                break
            await asyncio.sleep(0.5)
        await asyncio.sleep(2.0)

        # Transition to rock-solid HOLD mode
        try:
            await self.drone.action.hold()
            print_t(f"[PX4] Takeoff complete. Holding stably at {self._relative_altitude:.1f}m.")
        except Exception as e:
            print_t(f"[PX4] Failed to hold after takeoff: {e}")

    def _ensure_flying(self, allow_takeoff: bool = True) -> bool:
        """Ensure drone is armed and airborne before executing movement commands."""
        if not self._in_air and self._relative_altitude < 0.3:
            if not allow_takeoff:
                return False
            print_t("-> [PX4] Drone is on ground. Automatically arming and taking off first...")
            self._run_async(self._do_arm_and_takeoff(), timeout=20.0)
        return True

    @overrides
    def stop(self) -> bool:
        """Land the drone and stop camera observation."""
        print_t("[PX4] Stopping drone...")
        try:
            if self._connected:
                self.land()
        except Exception as e:
            print_t(f"[PX4] Stop warning: {e}")

        self._running = False
        self.obs.stop()
        return True

    async def _do_move(self, dx: float, dy: float):
        """Move closed-loop to target position using Offboard position setpoints."""
        rad = math.radians(self._heading_deg)
        # In body frame: dx is forward(+)/back(-), dy is left(+)/right(-)
        # In NED frame: positive dy is body-left, so body-right is -dy
        body_fwd = dx
        body_right = -dy
        delta_north = body_fwd * math.cos(rad) - body_right * math.sin(rad)
        delta_east = body_fwd * math.sin(rad) + body_right * math.cos(rad)

        target_n = self._north_m + delta_north
        target_e = self._east_m + delta_east
        target_d = self._down_m
        target_yaw = self._heading_deg

        # Prime initial setpoint before starting offboard
        await self.drone.offboard.set_position_ned(
            PositionNedYaw(self._north_m, self._east_m, self._down_m, target_yaw)
        )
        try:
            await self.drone.offboard.start()
        except OffboardError:
            pass

        # Send target position setpoint
        await self.drone.offboard.set_position_ned(
            PositionNedYaw(target_n, target_e, target_d, target_yaw)
        )

        dist_total = math.hypot(dx, dy)
        timeout = max(5.0, (dist_total / max(0.5, self.cruise_speed)) * 2.0 + 3.0)
        t0 = time.time()
        while time.time() - t0 < timeout:
            dist_remaining = math.hypot(self._north_m - target_n, self._east_m - target_e)
            if dist_remaining < 0.2:
                break
            await asyncio.sleep(0.1)

        # Lock firmly in HOLD mode
        try:
            await self.drone.action.hold()
        except Exception:
            pass

    @overrides
    def _move(self, dx: float, dy: float):
        """
        Move by dx (forward/back) and dy (left/right) in meters.
        dx > 0: forward, dx < 0: backward
        dy > 0: left, dy < 0: right
        """
        self._ensure_flying()
        parts = []
        if abs(dx) > 0.01:
            dir_x = "forward" if dx > 0 else "backward"
            parts.append(f"{dir_x} {abs(dx):.1f}m")
        if abs(dy) > 0.01:
            dir_y = "left" if dy > 0 else "right"
            parts.append(f"{dir_y} {abs(dy):.1f}m")
        desc = " and ".join(parts) if parts else f"dx={dx:.1f}m, dy={dy:.1f}m"
        print_t(f"-> [PX4] Precision flight: {desc}...")
        self._run_async(self._do_move(dx, dy), timeout=60.0)

    async def _do_rotate(self, deg: float):
        """Rotate closed-loop to target yaw using Offboard position setpoints."""
        # deg > 0: counter-clockwise (left) -> yaw decreases in NED
        # deg < 0: clockwise (right) -> yaw increases in NED
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
        """Climb or descend closed-loop using Offboard position setpoints."""
        target_d = self._down_m - dist
        # Safety clamp: do not descend below 0.3m above ground
        target_d = min(-0.3, target_d)

        await self.drone.offboard.set_position_ned(
            PositionNedYaw(self._north_m, self._east_m, self._down_m, self._heading_deg)
        )
        try:
            await self.drone.offboard.start()
        except OffboardError:
            pass

        await self.drone.offboard.set_position_ned(
            PositionNedYaw(self._north_m, self._east_m, target_d, self._heading_deg)
        )

        timeout = max(4.0, (abs(dist) / max(0.3, self.climb_speed)) * 2.0 + 3.0)
        t0 = time.time()
        while time.time() - t0 < timeout:
            alt_err = abs(self._down_m - target_d)
            if alt_err < 0.2:
                break
            await asyncio.sleep(0.1)

        try:
            await self.drone.action.hold()
        except Exception:
            pass

    def lift(self, dist: float):
        """Climb or descend by dist meters (positive up, negative down)."""
        dist = self._policy_guard("move", dist)
        if abs(dist) < 0.05:
            return

        if dist < 0:
            if not self._ensure_flying(allow_takeoff=False):
                print_t(f"-> [PX4] Drone is on ground ({self._relative_altitude:.1f}m). Cannot descend.")
                return
            print_t(f"-> [PX4] Descending {abs(dist):.1f}m (current alt: {self._relative_altitude:.1f}m)...")
            self._run_async(self._do_lift(dist), timeout=30.0)
            print_t(f"-> [PX4] Descent complete (current alt: {self._relative_altitude:.1f}m)")
        else:
            self._ensure_flying(allow_takeoff=True)
            print_t(f"-> [PX4] Climbing {abs(dist):.1f}m (current alt: {self._relative_altitude:.1f}m)...")
            self._run_async(self._do_lift(dist), timeout=30.0)
            print_t(f"-> [PX4] Climb complete (current alt: {self._relative_altitude:.1f}m)")

    def climb(self, dist: float = 1.0):
        """Climb upwards by dist meters."""
        self.lift(abs(dist))

    def descend(self, dist: float = 1.0):
        """Descend downwards by dist meters."""
        self.lift(-abs(dist))

    def set_speed(self, speed: float):
        """Set drone flight speed in m/s (between 0.5 and 5.0 m/s)."""
        speed = max(0.5, min(5.0, float(speed)))
        self.cruise_speed = speed
        print_t(f"-> [PX4] Cruise speed set to {self.cruise_speed:.1f} m/s")

    def arm(self):
        """Arm the drone."""
        self._policy_guard("action")
        print_t("-> [PX4] Arm commanded")
        async def _do_arm():
            try:
                await self.drone.action.arm()
                print_t("[PX4] Armed successfully.")
                self._send_chat("[ROBOT] Drone motors armed successfully.")
            except Exception as e:
                print_t(f"[PX4] Arm error: {e}")
                self._send_chat(f"[ERROR] Arm failed: {e}")
        self._run_async(_do_arm(), timeout=10.0)

    def disarm(self):
        """Disarm the drone motors safely (FlySky switch / stick disarm)."""
        self._policy_guard("action")
        print_t("-> [PX4] Disarm commanded")
        async def _do_disarm():
            try:
                await self.drone.action.disarm()
                print_t("[PX4] Disarmed successfully.")
                self._send_chat("[ROBOT] Drone motors disarmed.")
            except Exception as e:
                print_t(f"[PX4] Disarm error: {e}")
                self._send_chat(f"[ERROR] Disarm failed: {e}")
        self._run_async(_do_disarm(), timeout=10.0)

    def takeoff(self):
        """Take off to hover altitude."""
        self._policy_guard("action")
        print_t("-> [PX4] Takeoff commanded")
        self._send_chat("[ROBOT] Taking off...")
        self._run_async(self._do_arm_and_takeoff(), timeout=20.0)
        self._send_chat(f"[ROBOT] Airborne and holding at {self._relative_altitude:.1f}m.")

    def land(self):
        """Land the drone safely."""
        self._policy_guard("action")
        print_t("-> [PX4] Land commanded")
        self._send_chat("[ROBOT] Landing in progress...")
        async def _do_land():
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
        self._run_async(_do_land(), timeout=15.0)
        self._in_air = False

    def hold(self):
        """Hover in place and lock GPS/altitude position."""
        self._policy_guard("action")
        print_t("-> [PX4] Hold (hover in place / Loiter lock)")
        async def _do_hold():
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
        """Set the default takeoff hover altitude in meters."""
        self._policy_guard("action")
        altitude_m = max(1.0, min(50.0, float(altitude_m)))
        print_t(f"-> [PX4] Setting takeoff altitude to {altitude_m:.1f}m...")
        async def _do_set():
            try:
                await self.drone.action.set_takeoff_altitude(altitude_m)
                print_t(f"[PX4] Takeoff altitude configured to {altitude_m:.1f}m.")
                self._send_chat(f"[TAKEOFF] Default hover altitude set to {altitude_m:.1f}m.")
            except Exception as e:
                print_t(f"[PX4] Set takeoff altitude error: {e}")
        self._run_async(_do_set(), timeout=10.0)

    def get_telemetry_data(self) -> dict[str, Any]:
        """Return raw telemetry dictionary for UI and API endpoints."""
        return {
            "latitude": round(self._latitude_deg, 6),
            "longitude": round(self._longitude_deg, 6),
            "altitude_rel": round(self._relative_altitude, 2),
            "altitude_abs": round(self._absolute_altitude_m, 2),
            "heading": round(self._heading_deg, 1),
            "speed": round(self.cruise_speed, 1),
            "battery": round(self._battery_percent, 1),
            "voltage": round(self._battery_voltage, 2),
            "satellites": self._num_satellites,
            "flight_mode": self._flight_mode,
            "in_air": self._in_air,
            "armed": self._armed,
            "home": {
                "latitude": round(self._home_lat, 6),
                "longitude": round(self._home_lon, 6),
                "altitude": round(self._home_alt, 2),
            }
        }

    def get_telemetry(self) -> dict[str, Any]:
        """Get complete flight status, battery, GPS coordinates, and flight mode."""
        self._policy_guard("action")
        data = self.get_telemetry_data()
        msg = (
            f"Mode: {data['flight_mode']} | "
            f"Battery: {data['battery']:.0f}% ({data['voltage']:.1f}V) | "
            f"GPS: ({data['latitude']:.6f}, {data['longitude']:.6f}) Sats: {data['satellites']} | "
            f"Alt: {data['altitude_rel']:.1f}m | Hdg: {data['heading']:.1f}° | Armed: {data['armed']}"
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
