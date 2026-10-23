import os
import glob
import json
from typing import Dict, Any, List, Optional
from skyllm.utils import CURRENT_PROJ_DIR, print_t

CONFIG_PATH = os.path.join(CURRENT_PROJ_DIR, 'config/robot_info.json')

# Standard hardware device profiles
USB_BY_ID_PATTERNS = [
    "/dev/serial/by-id/usb-Auterion_PX4_FMU_v6X.x_0-if00",
    "/dev/serial/by-id/*Auterion*",
    "/dev/serial/by-id/*PX4*",
    "/dev/serial/by-id/*Pixhawk*",
]
RADIO_BY_ID_PATTERNS = [
    "/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0",
    "/dev/serial/by-id/*CP2102*",
    "/dev/serial/by-id/*FTDI*",
]

DEFAULT_PROFILES = {
    "radio": {
        "id": "radio",
        "name": "📡 Radio Telemetry (Wireless)",
        "default_address": "serial:///dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0:57600",
        "fallback_address": "serial:///dev/ttyUSB0:57600",
        "patterns": RADIO_BY_ID_PATTERNS,
        "dev_fallback": "/dev/ttyUSB0",
        "baud": 57600
    },
    "usb": {
        "id": "usb",
        "name": "⚡ USB Direct (Pixhawk)",
        "default_address": "serial:///dev/serial/by-id/usb-Auterion_PX4_FMU_v6X.x_0-if00:57600",
        "fallback_address": "serial:///dev/ttyACM0:57600",
        "patterns": USB_BY_ID_PATTERNS,
        "dev_fallback": "/dev/ttyACM0",
        "baud": 57600
    },
    "sim": {
        "id": "sim",
        "name": "🖥 Simulation (Gazebo SITL)",
        "default_address": "udpin://0.0.0.0:14540",
        "fallback_address": "udpin://0.0.0.0:14540",
        "patterns": [],
        "dev_fallback": None,
        "baud": None
    }
}

def detect_device_path(profile_id: str) -> Optional[str]:
    """Detect matching serial device path for a given profile."""
    prof = DEFAULT_PROFILES.get(profile_id)
    if not prof or not prof["patterns"]:
        return None

    # Check by-id patterns
    for pat in prof["patterns"]:
        matches = glob.glob(pat)
        if matches:
            return matches[0]

    # Check dev fallback
    fallback = prof.get("dev_fallback")
    if fallback and os.path.exists(fallback):
        return fallback

    return None

def get_profile_address(profile_id: str) -> str:
    """Return the best available MAVSDK address for a profile."""
    prof = DEFAULT_PROFILES.get(profile_id)
    if not prof:
        return "udpin://0.0.0.0:14540"

    if profile_id == "sim":
        return prof["default_address"]

    detected = detect_device_path(profile_id)
    if detected:
        return f"serial://{detected}:{prof['baud']}"
    
    return prof["default_address"]

def get_current_configured_address() -> str:
    """Read the currently configured mavsdk_address from robot_info.json."""
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as f:
                data = json.load(f)
                return data.get("extra", {}).get("mavsdk_address", "")
    except Exception as e:
        print_t(f"[ConnManager] Error reading config: {e}")
    return ""

def determine_active_mode(address: str) -> str:
    """Determine whether an address belongs to usb, radio, or sim."""
    addr = str(address).lower()
    if "cp2102" in addr or "ttyusb" in addr:
        return "radio"
    if "auterion" in addr or "ttyacm" in addr or "fmu" in addr or "pixhawk" in addr:
        return "usb"
    if "udpin" in addr or "14540" in addr or "gazebo" in addr:
        return "sim"
    if "serial" in addr:
        return "usb"
    return "sim"

def get_connection_status(robot_wrapper=None) -> Dict[str, Any]:
    """Get status of available connections and current active connection."""
    curr_addr = ""
    is_connected = False

    if robot_wrapper is not None:
        curr_addr = getattr(robot_wrapper, "system_address", "")
        is_connected = getattr(robot_wrapper, "_connected", False)

    if not curr_addr:
        curr_addr = get_current_configured_address()

    active_mode = determine_active_mode(curr_addr)

    options = []
    for pid, prof in DEFAULT_PROFILES.items():
        if pid == "sim":
            detected = True
            dev_path = "Network UDP (14540)"
        else:
            detected_path = detect_device_path(pid)
            detected = bool(detected_path)
            dev_path = detected_path or prof["dev_fallback"]

        options.append({
            "id": pid,
            "name": prof["name"],
            "detected": detected,
            "device": dev_path,
            "address": get_profile_address(pid),
            "is_active": (pid == active_mode)
        })

    return {
        "current_address": curr_addr,
        "active_mode": active_mode,
        "is_connected": is_connected,
        "options": options
    }

def save_address_to_config(address: str) -> bool:
    """Save the updated mavsdk_address to config/robot_info.json."""
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as f:
                data = json.load(f)
        else:
            data = {"robot_id": "pixhawk_hw_1", "robot_type": "px4", "extra": {}}

        if "extra" not in data or data["extra"] is None:
            data["extra"] = {}

        data["extra"]["mavsdk_address"] = address

        with open(CONFIG_PATH, 'w') as f:
            json.dump(data, f, indent=4)
        print_t(f"[ConnManager] Saved new mavsdk_address to {CONFIG_PATH}: {address}")
        return True
    except Exception as e:
        print_t(f"[ConnManager] Failed to save config: {e}")
        return False

def switch_connection_mode(robot_wrapper, mode_or_address: str) -> Dict[str, Any]:
    """
    Switch connection mode (e.g. 'usb', 'radio', 'sim' or explicit address).
    Updates robot_info.json and reconnects the robot.
    """
    if mode_or_address in DEFAULT_PROFILES:
        target_address = get_profile_address(mode_or_address)
        mode = mode_or_address
    else:
        target_address = mode_or_address
        mode = determine_active_mode(target_address)

    # 1. Update config file
    save_address_to_config(target_address)

    # 2. Reconnect robot if running
    connected = False
    message = ""
    if robot_wrapper is not None and hasattr(robot_wrapper, "reconnect"):
        try:
            connected = robot_wrapper.reconnect(target_address)
            message = "Reconnected successfully" if connected else "Reconnection attempt timed out"
        except Exception as e:
            connected = False
            message = str(e)
    else:
        message = "Config updated. Start or restart WebUI to connect."

    return {
        "status": "ok",
        "mode": mode,
        "address": target_address,
        "connected": connected,
        "message": message
    }
