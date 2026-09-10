import os
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
from PIL import Image

from typefly.robot_info import RobotInfo
from typefly.platforms.px4_wrapper import PX4Wrapper, PX4Observation
from typefly.llm_controller import LLMController


class TestPX4Wrapper(unittest.TestCase):
    def setUp(self):
        self.robot_info = RobotInfo(
            robot_id="px4_test",
            robot_type="px4",
            extra={
                "mavsdk_address": "udp://:14540",
                "cruise_speed": 1.5,
                "yaw_rate": 45.0,
                "climb_speed": 1.0,
            }
        )

    def test_px4_wrapper_initialization(self):
        wrapper = PX4Wrapper(self.robot_info)
        self.assertEqual(wrapper.cruise_speed, 1.5)
        self.assertEqual(wrapper.yaw_rate, 45.0)
        self.assertEqual(wrapper.climb_speed, 1.0)
        self.assertTrue(wrapper.camera_moves_with_body)

        # Check required skills
        skills = wrapper.skillset.skills
        self.assertIn("move_forward", skills)
        self.assertIn("move_backward", skills)
        self.assertIn("move_left", skills)
        self.assertIn("move_right", skills)
        self.assertIn("rotate_left", skills)
        self.assertIn("rotate_right", skills)
        self.assertIn("lift", skills)
        self.assertIn("climb", skills)
        self.assertIn("descend", skills)
        self.assertIn("takeoff", skills)
        self.assertIn("land", skills)
        self.assertIn("hold", skills)
        self.assertIn("set_speed", skills)
        self.assertIn("return_to_launch", skills)
        self.assertIn("return_home", skills)
        self.assertIn("goto_gps", skills)
        self.assertIn("get_gps_position", skills)
        self.assertIn("get_home_position", skills)
        self.assertIn("orbit", skills)
        self.assertIn("set_rtl_altitude", skills)
        self.assertIn("set_takeoff_altitude", skills)
        self.assertIn("get_telemetry", skills)
        self.assertIn("emergency_stop", skills)

        wrapper.set_speed(2.5)
        self.assertEqual(wrapper.cruise_speed, 2.5)

        # Test telemetry data retrieval
        telem = wrapper.get_telemetry_data()
        self.assertIn("latitude", telem)
        self.assertIn("longitude", telem)
        self.assertIn("battery", telem)
        self.assertIn("flight_mode", telem)

    def test_px4_observation_placeholder(self):
        obs = PX4Observation(self.robot_info)
        img = obs.image
        self.assertIsNotNone(img)
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (640, 480))

    def test_px4_observation_gz_callback(self):
        obs = PX4Observation(self.robot_info)
        mock_msg = MagicMock()
        mock_msg.height = 100
        mock_msg.width = 100
        # 100x100x3 bytes
        mock_msg.data = b"\x00" * (100 * 100 * 3)
        obs._on_gz_image(mock_msg)
        self.assertEqual(obs.image.size, (100, 100))

    def test_llm_controller_dispatch(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "dummy_key"}):
            controller = LLMController(self.robot_info)
            self.assertIsInstance(controller.robot, PX4Wrapper)
            pov = controller.fetch_robot_pov()
            self.assertIsNotNone(pov)


if __name__ == "__main__":
    unittest.main()
