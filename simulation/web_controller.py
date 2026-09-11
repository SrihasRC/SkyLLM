#!/usr/bin/env bash
#!/usr/bin/env python3
"""Web Controller for PX4 Drone and 2-Axis Gimbal.

Provides an interactive Web Ground Control Station (GCS) on http://localhost:8080.
Connects to MAVSDK (PX4) and Gazebo gimbal topics, and gracefully handles offline states.
"""

import asyncio
import json
import os
import shutil
import subprocess
from aiohttp import web

DIR = os.path.dirname(os.path.abspath(__file__))
HTML_FILE = os.path.join(DIR, "drone_control_ui.html")

# Drone state
state = {
    "connected": False,
    "armed": False,
    "mode": "MANUAL",
    "altitude": 0.0,
    "battery": 98,
    "gimbal_pitch": -45.0,
    "gimbal_yaw": 0.0,
    "roll": 0.0,
    "pitch": 0.0,
    "yaw": 0.0,
    "throttle": 0.5,
}

# Gimbal helper
def send_gimbal_cmd(topic, val):
    # Detect gz or ign command
    cmd = "gz" if shutil.which("gz") else "ign"
    msg_type = "gz.msgs.Double"
    try:
        subprocess.Popen(
            [cmd, "topic", "-t", topic, "-m", msg_type, "-p", f"data: {val:.4f}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"[Gimbal Error] {e}")


# Web Handlers
async def handle_index(request):
    if os.path.exists(HTML_FILE):
        return web.FileResponse(HTML_FILE)
    return web.Response(text="drone_control_ui.html not found", status=404)


async def handle_status(request):
    return web.json_response(state)


async def handle_command(request):
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"status": "error", "message": "Invalid JSON"}, status=400)

    action = data.get("action")
    print(f"[Web GCS Action] {action}: {data}")

    if action == "arm":
        state["armed"] = True
        state["altitude"] = max(state["altitude"], 3.5)
    elif action == "disarm":
        state["armed"] = False
        state["altitude"] = 0.0
    elif action == "land":
        state["armed"] = False
        state["altitude"] = 0.0
    elif action == "gimbal":
        pitch = data.get("pitch")
        yaw = data.get("yaw")
        if pitch is not None:
            state["gimbal_pitch"] = float(pitch)
            send_gimbal_cmd("/gimbal/cmd_pitch", float(pitch) * 3.14159 / 180.0)
        if yaw is not None:
            state["gimbal_yaw"] = float(yaw)
            send_gimbal_cmd("/gimbal/cmd_yaw", float(yaw) * 3.14159 / 180.0)
    elif action == "throttle":
        delta = data.get("delta", 0)
        state["altitude"] = max(0.0, min(50.0, state["altitude"] + delta))

    return web.json_response({"status": "ok", "state": state})


async def try_mavsdk_loop():
    """Attempt background connection to MAVSDK without blocking web server."""
    try:
        from mavsdk import System
        drone = System()
        await drone.connect(system_address="udp://:14540")
        async for s in drone.core.connection_state():
            if s.is_connected:
                state["connected"] = True
                print("[MAVSDK] Connected to PX4!")
                break
    except Exception as e:
        print(f"[MAVSDK] Drone connection waiting ({e}). Running in standalone web GCS mode.")


async def init_app():
    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_get("/api/status", handle_status)
    app.router.add_post("/api/command", handle_command)
    return app


def main():
    port = int(os.environ.get("PORT", 8080))
    print("========================================================")
    print(f" TypeFly Web Ground Control Station (GCS)")
    print(f" URL: http://localhost:{port}")
    print("========================================================")
    app = init_app()
    asyncio.ensure_future(try_mavsdk_loop())
    web.run_app(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
