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
from mavsdk.offboard import PositionNedYaw, VelocityBodyYawspeed, OffboardError

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
        self._north_m = 0.0
        self._east_m = 0.0
        self._down_m = 0.0
        self._heading_deg = 0.0
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
        self.skillset.add_skill(self.takeoff, "Take off to hover altitude")
        self.skillset.add_skill(self.land, "Land the drone safely")
        self.skillset.add_skill(self.lift, "Move up or down by distance in meters (positive up, negative down)")
        self.skillset.add_skill(self.climb, "Climb upwards by distance in meters (positive number)")
        self.skillset.add_skill(self.descend, "Descend downwards by distance in meters (positive number)")
        self.skillset.add_skill(self.hold, "Hover in place")
        self.skillset.add_skill(self.set_speed, "Set drone flight speed in meters per second (0.5 to 5.0 m/s)")

    def _run_loop(self):
        """Run the persistent asyncio event loop."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_async(self, coro, timeout: float = 30.0):
        """Execute an async coroutine synchronously on the background MAVSDK event loop."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

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

        async def _pos_loop():
            try:
                async for pos in self.drone.telemetry.position():
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

        await asyncio.gather(_in_air_loop(), _pos_loop(), _ned_loop(), _heading_loop())

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
            except Exception as e:
                print_t(f"[PX4] Arm error: {e}")
        self._run_async(_do_arm(), timeout=10.0)

    def takeoff(self):
        """Take off to hover altitude."""
        self._policy_guard("action")
        print_t("-> [PX4] Takeoff commanded")
        self._run_async(self._do_arm_and_takeoff(), timeout=20.0)

    def land(self):
        """Land the drone safely."""
        self._policy_guard("action")
        print_t("-> [PX4] Land commanded")
        async def _do_land():
            try:
                try:
                    await self.drone.offboard.stop()
                except Exception:
                    pass
                await self.drone.action.land()
                print_t("[PX4] Landing in progress...")
            except Exception as e:
                print_t(f"[PX4] Land error: {e}")
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
            except Exception:
                pass
        self._run_async(_do_hold(), timeout=5.0)

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
