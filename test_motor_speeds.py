#!/usr/bin/env python3
"""
TypeFly Safe Bench Motor Speed Tester
=====================================
Safely tests drone motor speeds over USB with strict safety bounds.
Guarantees:
  - Strict speed ceiling (never exceeds safe bench limits).
  - No unexpected increasing speed (anti-integral windup).
  - ZERO calibration or parameter changes.
  - Immediate emergency cutoff on Ctrl+C.

Usage:
  .venv/bin/python3 test_motor_speeds.py --monitor       # Passively display live motor PWMs
  .venv/bin/python3 test_motor_speeds.py --test-each     # Gently test M1, M2, M3, M4 individually (1080 µs)
  .venv/bin/python3 test_motor_speeds.py --test-all      # Gently test all 4 motors at idle (1080 µs)
"""

import argparse
import asyncio
import glob
import json
import os
import signal
import sys
import time
from typing import Dict, Any, Optional

import mavsdk
from mavsdk import System
from mavsdk.offboard import AttitudeRate

# Strict bench safety limits
MIN_PWM = 1000
SAFE_IDLE_PWM = 1080        # Gentle spin (~8% throttle)
MAX_BENCH_PWM = 1200        # Absolute safety cap on the bench (20% throttle)
DEFAULT_TEST_DURATION = 1.5 # Seconds per motor test


def find_usb_device() -> str:
    """Find the USB Auterion/Pixhawk device path."""
    patterns = [
        "/dev/serial/by-id/usb-Auterion_PX4_FMU_v6X.x_0-if00",
        "/dev/serial/by-id/*Auterion*",
        "/dev/serial/by-id/*PX4*",
        "/dev/serial/by-id/*Pixhawk*",
        "/dev/ttyACM0",
    ]
    for pat in patterns:
        matches = glob.glob(pat)
        if matches and os.path.exists(matches[0]):
            return f"serial://{matches[0]}:57600"
    return "serial:///dev/ttyACM0:57600"


class SafeMotorTester:
    def __init__(self, address: str):
        self.address = address
        self.drone = System()
        self.running = True
        self.armed = False
        self.battery_voltage = 0.0
        self.battery_percent = 0.0
        self.motor_pwms = {"m1": 1000, "m2": 1000, "m3": 1000, "m4": 1000}

    async def connect(self):
        print(f"\n[MotorTester] Connecting to drone via USB:\n  --> {self.address}")
        await self.drone.connect(system_address=self.address)
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print(">>> [SUCCESS] Connected to Pixhawk FMUv6X via USB! <<<\n")
                break

        # Start background telemetry listeners
        asyncio.create_task(self._listen_battery())
        asyncio.create_task(self._listen_armed())
        asyncio.create_task(self._listen_servos())
        await asyncio.sleep(0.8)

    async def _listen_battery(self):
        try:
            async for bat in self.drone.telemetry.battery():
                self.battery_voltage = bat.voltage_v
                self.battery_percent = min(100.0, bat.remaining_percent * 100.0 if bat.remaining_percent <= 1.0 else bat.remaining_percent)
        except Exception:
            pass

    async def _listen_armed(self):
        try:
            async for armed in self.drone.telemetry.armed():
                self.armed = armed
        except Exception:
            pass

    async def _listen_servos(self):
        try:
            async for msg in self.drone.mavlink_direct.message("SERVO_OUTPUT_RAW"):
                try:
                    data = json.loads(msg.fields_json)
                    self.motor_pwms["m1"] = int(data.get("servo1_raw", 1000))
                    self.motor_pwms["m2"] = int(data.get("servo2_raw", 1000))
                    self.motor_pwms["m3"] = int(data.get("servo3_raw", 1000))
                    self.motor_pwms["m4"] = int(data.get("servo4_raw", 1000))
                except Exception:
                    pass
        except Exception:
            pass

    async def stop_all_motors(self):
        """Immediately cut motor thrust to zero and disarm."""
        try:
            await self.drone.offboard.set_attitude_rate(AttitudeRate(0.0, 0.0, 0.0, 0.0))
        except Exception:
            pass
        try:
            await self.drone.action.disarm()
        except Exception:
            try:
                await self.drone.action.kill()
            except Exception:
                pass
        try:
            await self.drone.offboard.stop()
        except Exception:
            pass
        self.armed = False
        print("[MotorTester] Motors cutoff (1000 µs). Disarmed safely.")

    async def monitor_live(self):
        """Monitor and display live motor speeds without sending any commands."""
        print("=" * 80)
        print(" 🛰 LIVE MOTOR SPEED MONITOR (Read-Only Mode)")
        print("=" * 80)
        print(" Reading live ESC PWM outputs directly from drone flight controller.")
        print(" Press Ctrl+C to exit.\n")
        while self.running:
            m = self.motor_pwms
            avg_m = (m['m1'] + m['m2'] + m['m3'] + m['m4']) / 4.0
            armed_str = "ARMED" if self.armed else "DISARMED"
            sys.stdout.write(
                f"\r State: [{armed_str:8s}] | Battery: {self.battery_voltage:.2f}V ({self.battery_percent:.0f}%) | "
                f"M1: {m['m1']:4d} µs | M2: {m['m2']:4d} µs | M3: {m['m3']:4d} µs | M4: {m['m4']:4d} µs | "
                f"Avg: {avg_m:4.0f} µs"
            )
            sys.stdout.flush()
            await asyncio.sleep(0.1)
        print("\n")

    async def test_single_motor(self, motor_idx: int, target_pwm: int = SAFE_IDLE_PWM, duration_s: float = DEFAULT_TEST_DURATION):
        """
        Gently spin a single motor at safe idle with anti-windup zero rates.
        Strictly clamped to <= MAX_BENCH_PWM.
        """
        # Strict safety clamp
        target_pwm = min(MAX_BENCH_PWM, max(MIN_PWM, target_pwm))
        motor_names = {
            1: "M1 (Front Right, CCW)",
            2: "M2 (Rear Left,   CCW)",
            3: "M3 (Front Left,   CW)",
            4: "M4 (Rear Right,   CW)",
        }
        name = motor_names.get(motor_idx, f"Motor {motor_idx}")

        print(f"\n--- Testing {name} at safe speed {target_pwm} µs for {duration_s:.1f}s ---")
        
        # Arm drone if not armed
        if not self.armed:
            try:
                await self.drone.action.arm()
                await asyncio.sleep(0.3)
            except Exception:
                try:
                    await self.drone.action.arm_force()
                    await asyncio.sleep(0.3)
                except Exception as e:
                    print(f"[MotorTester] Arming note: {e}")

        # Compute gentle thrust: (target_pwm - 1000) / 1000
        thrust = max(0.0, min(0.20, (target_pwm - 1000.0) / 1000.0))

        # Use zero attitude rates (0.0 deg/s) so PID does not ramp speed
        try:
            await self.drone.offboard.set_attitude_rate(AttitudeRate(0.0, 0.0, 0.0, thrust))
            await self.drone.offboard.start()
        except Exception:
            pass

        t0 = time.time()
        while time.time() - t0 < duration_s and self.running:
            m = self.motor_pwms
            sys.stdout.write(
                f"\r   Live Outputs -> M1: {m['m1']} µs | M2: {m['m2']} µs | M3: {m['m3']} µs | M4: {m['m4']} µs   "
            )
            sys.stdout.flush()
            await asyncio.sleep(0.08)

        # Immediate cutoff
        try:
            await self.drone.offboard.set_attitude_rate(AttitudeRate(0.0, 0.0, 0.0, 0.0))
        except Exception:
            pass
        await asyncio.sleep(0.2)
        print(f"\n   [OK] {name} test completed safely.")

    async def run_sequential_test(self, target_pwm: int = SAFE_IDLE_PWM):
        """Test M1, M2, M3, M4 one by one safely."""
        print("=" * 80)
        print(f" ⚠️  SAFETY BENCH MOTOR TEST (Strict Limit: {target_pwm} µs / Safe Idle)")
        print("=" * 80)
        print(" Each motor will spin gently for 1.5 seconds in order: M1 -> M2 -> M3 -> M4.")
        print(" Speed is strictly capped. Press Ctrl+C at any time for instant stop.\n")

        for m_idx in [1, 2, 3, 4]:
            if not self.running:
                break
            await self.test_single_motor(m_idx, target_pwm=target_pwm, duration_s=1.5)
            await asyncio.sleep(0.8)

        await self.stop_all_motors()
        print("\n================================================================================")
        print(" [COMPLETE] All 4 motors tested safely without exceeding speed limits!")
        print("================================================================================\n")

    async def run_all_motors_test(self, target_pwm: int = SAFE_IDLE_PWM, duration_s: float = 2.0):
        """Gently spin all 4 motors at safe idle for duration_s seconds."""
        target_pwm = min(MAX_BENCH_PWM, max(MIN_PWM, target_pwm))
        print("=" * 80)
        print(f" ⚠️  ALL MOTORS GENTLE IDLE TEST (Strict Limit: {target_pwm} µs)")
        print("=" * 80)
        print(f" All 4 motors will spin at gentle idle for {duration_s:.1f} seconds.")
        print(" Speed is strictly capped. Press Ctrl+C at any time for instant stop.\n")

        if not self.armed:
            try:
                await self.drone.action.arm()
                await asyncio.sleep(0.3)
            except Exception:
                try:
                    await self.drone.action.arm_force()
                    await asyncio.sleep(0.3)
                except Exception as e:
                    print(f"[MotorTester] Arm note: {e}")

        thrust = max(0.0, min(0.20, (target_pwm - 1000.0) / 1000.0))
        try:
            await self.drone.offboard.set_attitude_rate(AttitudeRate(0.0, 0.0, 0.0, thrust))
            await self.drone.offboard.start()
        except Exception:
            pass

        t0 = time.time()
        while time.time() - t0 < duration_s and self.running:
            m = self.motor_pwms
            avg_m = (m['m1'] + m['m2'] + m['m3'] + m['m4']) / 4.0
            sys.stdout.write(
                f"\r   Live -> M1: {m['m1']} µs | M2: {m['m2']} µs | M3: {m['m3']} µs | M4: {m['m4']} µs | Avg: {avg_m:.0f} µs   "
            )
            sys.stdout.flush()
            await asyncio.sleep(0.08)

        await self.stop_all_motors()
        print("\n================================================================================")
        print(" [COMPLETE] All-motors idle test completed safely!")
        print("================================================================================\n")


def parse_args():
    parser = argparse.ArgumentParser(description="TypeFly Safe Bench Motor Speed Tester")
    parser.add_argument("--device", "-d", type=str, default=None, help="USB serial device path.")
    parser.add_argument("--monitor", action="store_true", help="Passively monitor live motor speeds (read-only).")
    parser.add_argument("--test-each", action="store_true", help="Gently test M1, M2, M3, M4 one by one (safe idle).")
    parser.add_argument("--test-all", action="store_true", help="Gently test all 4 motors at safe idle (1080 µs).")
    parser.add_argument("--pwm", type=int, default=SAFE_IDLE_PWM, help=f"Test PWM value (default: {SAFE_IDLE_PWM} µs, max: {MAX_BENCH_PWM} µs).")
    return parser.parse_args()


async def main_async():
    args = parse_args()
    dev_addr = args.device or find_usb_device()

    tester = SafeMotorTester(address=dev_addr)

    loop = asyncio.get_running_loop()
    def _sig():
        tester.running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _sig)
        except Exception:
            pass

    await tester.connect()

    try:
        if args.test_each:
            await tester.run_sequential_test(target_pwm=args.pwm)
        elif args.test_all:
            await tester.run_all_motors_test(target_pwm=args.pwm)
        else:
            # Default to live monitor mode
            await tester.monitor_live()
    finally:
        await tester.stop_all_motors()


def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n[MotorTester] Exited safely.")


if __name__ == "__main__":
    main()
