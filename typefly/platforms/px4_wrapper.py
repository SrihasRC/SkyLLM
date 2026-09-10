import asyncio
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
from mavsdk.offboard import VelocityBodyYawspeed, OffboardError

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
        self.max_distance = 50.0
        self._running = True

        # Flight stick state (Position Control inputs)
        # pitch: forward (+), backward (-)
        # roll: right (+), left (-)
        # throttle: climb (>0.5), hover (0.5), descend (<0.5)
        # yaw: clockwise (+), counter-clockwise (-)
        self._pitch = 0.0
        self._roll = 0.0
        self._throttle = 0.5
        self._yaw = 0.0

        # Dedicated background asyncio event loop for MAVSDK commands
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()

        obs = PX4Observation(robot_info)
        super().__init__(robot_info, obs)

        # Register drone skills
        self.skillset.add_skill(self.arm, "Arm the drone in Position Control mode")
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

    def _run_async(self, coro, timeout: float = 20.0):
        """Execute an async coroutine synchronously on the background MAVSDK event loop."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    @overrides
    def start(self) -> bool:
        """Connect to PX4 SITL, start continuous control feed, and start camera observation."""
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

    async def _manual_control_loop(self):
        """Continuously stream stick positions to PX4 at 20Hz (Position Control)."""
        while self._running:
            try:
                await self.drone.manual_control.set_manual_control_input(
                    self._pitch, self._roll, self._throttle, self._yaw
                )
            except Exception:
                pass
            await asyncio.sleep(0.05)

    async def _connect_drone(self):
        await self.drone.connect(system_address=self.system_address)
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print_t("[PX4] MAVSDK connected to PX4 SITL!")
                self._connected = True
                break

        # Start background manual control stream loop
        asyncio.run_coroutine_threadsafe(self._manual_control_loop(), self._loop)
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
        """Continuously monitor in-air status and relative altitude from PX4 telemetry."""
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

        await asyncio.gather(_in_air_loop(), _pos_loop())

    async def _do_arm_and_takeoff(self):
        """Arm in position control and initiate takeoff to hover altitude."""
        try:
            await self.drone.manual_control.start_position_control()
            await self.drone.action.arm()
            print_t("[PX4] Armed in Position Control mode.")
        except Exception:
            try:
                await self.drone.action.arm()
                await self.drone.manual_control.start_position_control()
                print_t("[PX4] Armed in Position Control mode.")
            except Exception as e:
                print_t(f"[PX4] Arm error: {e}")

        try:
            await self.drone.action.takeoff()
            print_t("[PX4] Takeoff commanded! Climbing to hover altitude...")
            self._in_air = True
        except Exception as e:
            print_t(f"[PX4] Takeoff warning: {e}")

        # Hover altitude ascent time
        await asyncio.sleep(4.0)
        self._throttle = 0.5
        # Lock in stable HOLD (Loiter) mode after takeoff completes
        try:
            await self.drone.action.hold()
            print_t("[PX4] Takeoff complete. Drone locked in stable HOLD mode.")
        except Exception as e:
            print_t(f"[PX4] Failed to switch to hold after takeoff: {e}")

    async def _do_ensure_position_control(self):
        try:
            await self.drone.manual_control.start_position_control()
        except Exception:
            pass

    def _ensure_position_control(self):
        try:
            self._run_async(self._do_ensure_position_control(), timeout=5.0)
        except Exception:
            pass

    async def _do_hold(self):
        """Command active GPS and altitude hold (Loiter)."""
        self._pitch = 0.0
        self._roll = 0.0
        self._throttle = 0.5
        self._yaw = 0.0
        try:
            await self.drone.action.hold()
        except Exception:
            pass

    def _hold(self):
        """Internal helper to command active GPS hold without policy guard."""
        try:
            self._run_async(self._do_hold(), timeout=5.0)
        except Exception:
            pass

    def _ensure_flying(self, allow_takeoff: bool = True) -> bool:
        """Ensure drone is armed and airborne before executing movement commands."""
        if not self._in_air and self._relative_altitude < 0.3:
            if not allow_takeoff:
                return False
            print_t("-> [PX4] Drone is on ground. Automatically arming and taking off first...")
            self._run_async(self._do_arm_and_takeoff(), timeout=20.0)

        # Switch to Position Control (POSCTL) mode so manual control stick deflections are accepted
        self._ensure_position_control()
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

    @overrides
    def _move(self, dx: float, dy: float):
        """
        Move by dx (forward/back) and dy (left/right) in meters.
        dx > 0: forward, dx < 0: backward
        dy > 0: left, dy < 0: right
        """
        self._ensure_flying()
        vel = max(0.5, self.cruise_speed)
        # Moderate stick deflection (~0.40 - 0.45) keeps drone level and prevents altitude loss or wobble
        speed_val = min(0.65, max(0.25, 0.45 * (vel / 1.5)))

        if abs(dx) > 0.01:
            pitch = speed_val if dx > 0 else -speed_val
            duration = max(0.4, abs(dx) / vel)
            direction = "forward" if dx > 0 else "backward"
            print_t(f"-> [PX4] Flying {direction} {abs(dx):.1f}m ({duration:.1f}s)")
            self._pitch = pitch
            time.sleep(duration)
            self._pitch = 0.0
            time.sleep(0.4)

        if abs(dy) > 0.01:
            # dy > 0 is left; negative roll is left in PX4 manual control
            roll = -speed_val if dy > 0 else speed_val
            duration = max(0.4, abs(dy) / vel)
            direction = "left" if dy > 0 else "right"
            print_t(f"-> [PX4] Flying {direction} {abs(dy):.1f}m ({duration:.1f}s)")
            self._roll = roll
            time.sleep(duration)
            self._roll = 0.0
            time.sleep(0.4)

        # Active brake into GPS hold (Loiter) to eliminate drift and floating
        self._hold()

    @overrides
    def _rotate(self, deg: float):
        """
        Rotate by deg degrees.
        deg > 0: rotate left (counter-clockwise)
        deg < 0: rotate right (clockwise)
        """
        self._ensure_flying()
        yaw = -0.45 if deg > 0 else 0.45
        duration = max(0.4, abs(deg) / 35.0)  # ~35 deg/s for steady on-axis turn
        direction = "left" if deg > 0 else "right"
        print_t(f"-> [PX4] Rotating {direction} {abs(deg):.1f} deg ({duration:.1f}s)")
        self._yaw = yaw
        time.sleep(duration)
        self._yaw = 0.0
        time.sleep(0.4)
        # Active GPS lock after turning
        self._hold()

    def lift(self, dist: float):
        """Climb or descend by dist meters (positive up, negative down)."""
        dist = self._policy_guard("move", dist)
        if abs(dist) < 0.05:
            return

        if dist < 0:
            # Descend
            if not self._ensure_flying(allow_takeoff=False):
                print_t(f"-> [PX4] Drone is on ground ({self._relative_altitude:.1f}m). Cannot descend.")
                return

            print_t(f"-> [PX4] Descending {abs(dist):.1f}m (current alt: {self._relative_altitude:.1f}m)")
            target_alt = max(0.4, self._relative_altitude + dist)
            self._throttle = 0.25  # smooth controlled descent stick
            start_time = time.time()
            timeout = abs(dist) / 0.8 + 5.0
            while time.time() - start_time < timeout:
                if self._relative_altitude <= target_alt or not self._in_air:
                    break
                time.sleep(0.1)
            self._throttle = 0.5  # neutral throttle
            time.sleep(0.3)
            self._hold()
            print_t(f"-> [PX4] Descent complete (current alt: {self._relative_altitude:.1f}m)")
        else:
            # Climb
            self._ensure_flying(allow_takeoff=True)
            print_t(f"-> [PX4] Climbing {abs(dist):.1f}m (current alt: {self._relative_altitude:.1f}m)")
            target_alt = self._relative_altitude + dist
            self._throttle = 0.75  # smooth controlled climb stick
            start_time = time.time()
            timeout = abs(dist) / 0.8 + 5.0
            while time.time() - start_time < timeout:
                if self._relative_altitude >= target_alt:
                    break
                time.sleep(0.1)
            self._throttle = 0.5  # neutral throttle
            time.sleep(0.3)
            self._hold()
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
        """Arm the drone in Position Control mode."""
        self._policy_guard("action")
        print_t("-> [PX4] Arm commanded")
        async def _do_arm():
            try:
                await self.drone.manual_control.start_position_control()
                await self.drone.action.arm()
                print_t("[PX4] Armed in Position Control mode.")
            except Exception as e:
                print_t(f"[PX4] Arm error: {e}")
        self._run_async(_do_arm(), timeout=10.0)

    def takeoff(self):
        """Take off to hover altitude."""
        self._policy_guard("action")
        print_t("-> [PX4] Takeoff commanded")
        self._run_async(self._do_arm_and_takeoff(), timeout=15.0)

    def land(self):
        """Land the drone safely."""
        self._policy_guard("action")
        print_t("-> [PX4] Land commanded")
        self._pitch = 0.0
        self._roll = 0.0
        self._throttle = 0.5
        self._yaw = 0.0
        async def _do_land():
            try:
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
        self._hold()

    @overrides
    def stop_motion(self) -> None:
        """Halt in-flight motion and lock position hold."""
        print_t("-> [PX4] Stop motion (locking position hold)")
        self._hold()

    @overrides
    def emergency_shutdown(self) -> None:
        """Emergency land."""
        print_t("-> [PX4] Emergency shutdown: landing immediately")
        self.land()
