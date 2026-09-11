import asyncio
import subprocess
import sys
from mavsdk import System
import KeyPressModule as kp

kp.init()
drone = System()

pitch_x, roll_y, throttle_z, yaw_r = 0.0, 0.0, 0.5, 0.0
is_connected = False

# Gimbal state
gimbal_pitch = 0.7854  # initial 45 deg down
gimbal_yaw = 0.0
GIMBAL_STEP = 0.03  # rad per tick (~1.7 deg)
GIMBAL_PITCH_MIN = -1.57
GIMBAL_PITCH_MAX = 0.5
GIMBAL_YAW_MIN = -1.57
GIMBAL_YAW_MAX = 1.57


# Flight speed multiplier (0.1 to 1.0)
speed_val = 0.6


def log(msg):
    sys.stdout.write(f"{msg}\r\n")
    sys.stdout.flush()


_gimbal_procs = {}

def set_gimbal(topic, value):
    prev = _gimbal_procs.get(topic)
    if prev is not None and prev.poll() is None:
        return
    _gimbal_procs[topic] = subprocess.Popen(
        ['gz', 'topic', '-t', topic, '-m', 'gz.msgs.Double',
         '-p', f'data: {value:.4f}'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


async def getKeyboardInput(my_drone):
    global pitch_x, roll_y, throttle_z, yaw_r, gimbal_pitch, gimbal_yaw, speed_val
    while True:
        pitch_x, roll_y, throttle_z, yaw_r = 0.0, 0.0, 0.5, 0.0

        # Speed adjustment controls
        if kp.consumeKey("+") or kp.consumeKey("=") or kp.consumeKey("]"):
            speed_val = min(1.0, round(speed_val + 0.1, 2))
            log(f"-- Flight Speed: {int(speed_val * 100)}% (scale={speed_val:.1f})")
        elif kp.consumeKey("-") or kp.consumeKey("_") or kp.consumeKey("["):
            speed_val = max(0.1, round(speed_val - 0.1, 2))
            log(f"-- Flight Speed: {int(speed_val * 100)}% (scale={speed_val:.1f})")
        elif kp.consumeKey("1"):
            speed_val = 0.3
            log(f"-- Flight Speed: 30% (Slow / Precision)")
        elif kp.consumeKey("2"):
            speed_val = 0.6
            log(f"-- Flight Speed: 60% (Normal)")
        elif kp.consumeKey("3"):
            speed_val = 1.0
            log(f"-- Flight Speed: 100% (Fast / Turbo)")

        # Flight controls:
        # x: forward/backward (pitch), y: right/left (roll), z: up/down (throttle), r: clockwise/anticlockwise (yaw)
        if kp.getKey("UP"):
            pitch_x = speed_val
        elif kp.getKey("DOWN"):
            pitch_x = -speed_val

        if kp.getKey("LEFT"):
            roll_y = -speed_val
        elif kp.getKey("RIGHT"):
            roll_y = speed_val

        if kp.getKey("w"):
            throttle_z = 1.0
        elif kp.getKey("s"):
            throttle_z = 0.0

        if kp.getKey("a"):
            yaw_r = -speed_val
        elif kp.getKey("d"):
            yaw_r = speed_val

        # Actions
        if kp.consumeKey("i"):
            asyncio.ensure_future(print_flight_mode(my_drone))
        elif kp.consumeKey("r"):
            try:
                await my_drone.manual_control.start_position_control()
                await my_drone.action.arm()
                log("-- Armed in Position Control! Hold 'w' to climb, or press 't' to auto-takeoff.")
            except Exception as e:
                try:
                    await my_drone.action.arm()
                    await my_drone.manual_control.start_position_control()
                    log("-- Armed in Position Control! Hold 'w' to climb, or press 't' to auto-takeoff.")
                except Exception as e2:
                    log(f"-- Arm failed: {e2}")
        elif kp.consumeKey("t"):
            try:
                log("-- Initiating Takeoff to hover altitude...")
                await my_drone.action.takeoff()
                log("-- Takeoff commanded! Use WASD and Arrow keys to fly.")
            except Exception as e:
                log(f"-- Takeoff failed: {e}")
        elif kp.consumeKey("l"):
            try:
                await my_drone.action.land()
                log("-- Landing commanded!")
            except Exception as e:
                log(f"-- Land failed: {e}")

        # Gimbal controls
        if kp.getKey("j"):
            gimbal_pitch = max(GIMBAL_PITCH_MIN, gimbal_pitch - GIMBAL_STEP)
            set_gimbal('/gimbal/cmd_pitch', gimbal_pitch)
        elif kp.getKey("k"):
            gimbal_pitch = min(GIMBAL_PITCH_MAX, gimbal_pitch + GIMBAL_STEP)
            set_gimbal('/gimbal/cmd_pitch', gimbal_pitch)
        if kp.getKey("n"):
            gimbal_yaw = max(GIMBAL_YAW_MIN, gimbal_yaw - GIMBAL_STEP)
            set_gimbal('/gimbal/cmd_yaw', gimbal_yaw)
        elif kp.getKey("m"):
            gimbal_yaw = min(GIMBAL_YAW_MAX, gimbal_yaw + GIMBAL_STEP)
            set_gimbal('/gimbal/cmd_yaw', gimbal_yaw)

        await asyncio.sleep(0.05)


async def print_flight_mode(my_drone):
    async for flight_mode in my_drone.telemetry.flight_mode():
        log(f"FlightMode: {flight_mode}")
        break


async def manual_control_drone(my_drone):
    global pitch_x, roll_y, throttle_z, yaw_r
    while True:
        try:
            await my_drone.manual_control.set_manual_control_input(pitch_x, roll_y, throttle_z, yaw_r)
        except Exception:
            pass
        await asyncio.sleep(0.05)


async def run_drone():
    global is_connected
    asyncio.ensure_future(getKeyboardInput(drone))
    await drone.connect(system_address="udp://:14540")
    log("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            log("-- Connected to drone!")
            break
    # Start manual control loop so input is continuously fed to MAVSDK
    asyncio.ensure_future(manual_control_drone(drone))
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            log("-- Global position state is good enough for flying.")
            break
    is_connected = True
    log("-- Ready! Press 'r' to arm (or 't' for auto-takeoff).")


async def run():
    await asyncio.gather(run_drone())


if __name__ == "__main__":
    asyncio.ensure_future(run())
    asyncio.get_event_loop().run_forever()
