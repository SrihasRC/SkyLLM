#!/usr/bin/env python3
"""
TypeFly RC Flight Recorder & Motor Calibration Analyzer
======================================================
Passively records drone motor PWM speeds and pilot RC commands during manual
RC flight via the telemetry radio module (CP2102 USB dongle or USB direct).

ZERO CONTROL COMMANDS ARE SENT:
The RC remote control (FlySky / transmitter) retains 100% full, uninterrupted
control of the drone at all times.

Logs recorded to CSV:
  - Timestamp (ISO and elapsed seconds)
  - Flight Mode & Armed State
  - RC Stick Inputs: Roll (CH1), Pitch (CH2), Throttle (CH3), Yaw (CH4), Aux (CH5-CH8)
  - Live Motor ESC Outputs: M1, M2, M3, M4 (microseconds) and Average PWM
  - Drone Telemetry: Relative Altitude (m & cm), Roll/Pitch/Yaw (deg), Battery Voltage

Usage:
  .venv/bin/python3 record_rc_flight.py
  .venv/bin/python3 record_rc_flight.py --device radio
  .venv/bin/python3 record_rc_flight.py --device /dev/ttyUSB0 --baud 57600
  .venv/bin/python3 record_rc_flight.py --output flight_logs/test1.csv
"""

import argparse
import asyncio
import csv
import datetime
import json
import math
import os
import signal
import sys
import time
from typing import Dict, Any, List, Optional

import mavsdk
from mavsdk import System

# Import TypeFly connection manager for auto-detecting the radio module
try:
    from skyllm.connection_manager import (
        detect_device_path,
        get_profile_address,
        get_current_configured_address,
        determine_active_mode,
        DEFAULT_PROFILES,
    )
except ImportError:
    detect_device_path = None
    get_profile_address = None
    get_current_configured_address = None
    determine_active_mode = None
    DEFAULT_PROFILES = {}


def resolve_connection_address(device_arg: Optional[str], baud_arg: int) -> str:
    """Resolve the MAVSDK connection address for the radio module or custom port."""
    if device_arg:
        arg_lower = device_arg.lower().strip()
        if arg_lower == "radio" and get_profile_address:
            return get_profile_address("radio")
        if arg_lower == "usb" and get_profile_address:
            return get_profile_address("usb")
        if arg_lower == "sim":
            return "udpin://0.0.0.0:14540"
        if device_arg.startswith("serial://") or device_arg.startswith("udpin://") or device_arg.startswith("udp://"):
            return device_arg
        if os.path.exists(device_arg) or "/dev/" in device_arg:
            return f"serial://{device_arg}:{baud_arg}"
        return device_arg

    # Default: Try reading from robot_info.json or auto-detect radio dongle
    if get_current_configured_address:
        saved_addr = get_current_configured_address()
        if saved_addr and "serial://" in saved_addr:
            dev_path = saved_addr.replace("serial://", "").split(":")[0]
            if os.path.exists(dev_path):
                return saved_addr

    if get_profile_address:
        radio_addr = get_profile_address("radio")
        dev_path = radio_addr.replace("serial://", "").split(":")[0]
        if os.path.exists(dev_path):
            return radio_addr

    # Fallback default radio address
    fallback_pattern = "/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0"
    if os.path.exists(fallback_pattern):
        return f"serial://{fallback_pattern}:{baud_arg}"
    if os.path.exists("/dev/ttyUSB0"):
        return f"serial:///dev/ttyUSB0:{baud_arg}"

    return f"serial://{fallback_pattern}:{baud_arg}"


class FlightRecorder:
    def __init__(self, address: str, output_csv: str, sample_rate_hz: float = 20.0, no_display: bool = False):
        self.address = address
        self.output_csv = output_csv
        self.sample_interval = 1.0 / max(1.0, min(100.0, sample_rate_hz))
        self.no_display = no_display

        self.drone = System()
        self.running = True
        self.start_time: Optional[float] = None
        self.samples: List[Dict[str, Any]] = []

        # Current live state
        self.connected = False
        self.armed = False
        self.in_air = False
        self.flight_mode = "UNKNOWN"
        self.relative_altitude_m = 0.0
        self.roll_deg = 0.0
        self.pitch_deg = 0.0
        self.yaw_deg = 0.0
        self.battery_voltage = 0.0
        self.battery_percent = 0.0
        self.rc_signal_percent = 0.0

        # RC channels (microseconds, 1000 - 2000)
        self.rc_channels = {
            "ch1": 1500,  # Roll
            "ch2": 1500,  # Pitch
            "ch3": 1000,  # Throttle
            "ch4": 1500,  # Yaw
            "ch5": 1000,  # Mode / Aux1
            "ch6": 1000,  # Aux2
            "ch7": 1000,  # Aux3
            "ch8": 1000,  # Aux4
        }

        # Motor PWMs (microseconds, 1000 - 2000)
        self.motor_pwms = {
            "m1": 1000,
            "m2": 1000,
            "m3": 1000,
            "m4": 1000,
        }

        # CSV Writer setup
        os.makedirs(os.path.dirname(os.path.abspath(self.output_csv)), exist_ok=True)
        self.csv_file = open(self.output_csv, mode="w", newline="", encoding="utf-8")
        self.fieldnames = [
            "timestamp_iso",
            "elapsed_s",
            "armed",
            "in_air",
            "flight_mode",
            "rc_roll_us",
            "rc_pitch_us",
            "rc_throttle_us",
            "rc_yaw_us",
            "rc_ch5_us",
            "rc_ch6_us",
            "rc_ch7_us",
            "rc_ch8_us",
            "motor1_pwm_us",
            "motor2_pwm_us",
            "motor3_pwm_us",
            "motor4_pwm_us",
            "avg_motor_pwm_us",
            "altitude_m",
            "altitude_cm",
            "attitude_roll_deg",
            "attitude_pitch_deg",
            "heading_yaw_deg",
            "battery_v",
            "battery_pct",
        ]
        self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
        self.writer.writeheader()
        self.csv_file.flush()

    async def start(self):
        """Connect to PX4 via radio module and start background telemetry monitors."""
        print(f"\n[Recorder] Initiating connection to drone at:\n  --> {self.address}")
        print("[Recorder] Connecting... (Ensure drone battery is connected and Radio dongle is plugged in)\n")

        await self.drone.connect(system_address=self.address)

        async def _wait_for_conn():
            async for state in self.drone.core.connection_state():
                if state.is_connected:
                    return True
            return False

        try:
            await asyncio.wait_for(_wait_for_conn(), timeout=15.0)
            self.connected = True
            print("================================================================================")
            print("  >>> SUCCESS: CONNECTED TO DRONE VIA RADIO TELEMETRY! <<<")
            print("================================================================================")
            print(" [PASSIVE MODE ACTIVE] Zero commands will be sent.")
            print(" You can now arm and fly the drone using your physical RC Remote Controller.")
            print(f" Saving continuous flight log to:\n   {self.output_csv}")
            print(" Press Ctrl+C at any time to finish flight and view calibration analysis.\n")
            await asyncio.sleep(1.0)
        except asyncio.TimeoutError:
            print("\n[ERROR] Connection timed out after 15s!")
            print("Check that:")
            print(" 1. The telemetry radio module is firmly plugged into USB.")
            print(" 2. The drone is powered on and the air radio LED is solid green.")
            print(" 3. The baud rate is 57600.")
            sys.exit(1)

        self.start_time = time.time()

        # Launch concurrent background listeners
        tasks = [
            asyncio.create_task(self._listen_armed()),
            asyncio.create_task(self._listen_flight_mode()),
            asyncio.create_task(self._listen_in_air()),
            asyncio.create_task(self._listen_position()),
            asyncio.create_task(self._listen_attitude()),
            asyncio.create_task(self._listen_battery()),
            asyncio.create_task(self._listen_rc_status()),
            asyncio.create_task(self._listen_servo_raw()),
            asyncio.create_task(self._listen_rc_channels()),
            asyncio.create_task(self._sampling_loop()),
        ]

        if not self.no_display:
            tasks.append(asyncio.create_task(self._display_loop()))

        await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------ MAVLink & Telemetry Listeners ------------------ #

    async def _listen_armed(self):
        try:
            async for armed in self.drone.telemetry.armed():
                self.armed = armed
        except Exception:
            pass

    async def _listen_flight_mode(self):
        try:
            async for mode in self.drone.telemetry.flight_mode():
                self.flight_mode = str(mode).split(".")[-1]
        except Exception:
            pass

    async def _listen_in_air(self):
        try:
            async for in_air in self.drone.telemetry.in_air():
                self.in_air = in_air
        except Exception:
            pass

    async def _listen_position(self):
        try:
            async for pos in self.drone.telemetry.position():
                self.relative_altitude_m = pos.relative_altitude_m
        except Exception:
            pass

    async def _listen_attitude(self):
        try:
            async for att in self.drone.telemetry.attitude_euler():
                self.roll_deg = att.roll_deg
                self.pitch_deg = att.pitch_deg
                self.yaw_deg = att.yaw_deg
        except Exception:
            pass

    async def _listen_battery(self):
        try:
            async for bat in self.drone.telemetry.battery():
                pct = bat.remaining_percent
                self.battery_percent = min(100.0, pct * 100.0 if pct <= 1.0 else pct)
                self.battery_voltage = bat.voltage_v
        except Exception:
            pass

    async def _listen_rc_status(self):
        try:
            async for rc in self.drone.telemetry.rc_status():
                self.rc_signal_percent = rc.signal_strength_percent
        except Exception:
            pass

    async def _listen_servo_raw(self):
        """Listen for MAVLink SERVO_OUTPUT_RAW to record real ESC motor PWM outputs."""
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

    async def _listen_rc_channels(self):
        """Listen for MAVLink RC_CHANNELS to record pilot stick commands."""
        try:
            async for msg in self.drone.mavlink_direct.message("RC_CHANNELS"):
                try:
                    data = json.loads(msg.fields_json)
                    self.rc_channels["ch1"] = int(data.get("chan1_raw", 1500))
                    self.rc_channels["ch2"] = int(data.get("chan2_raw", 1500))
                    self.rc_channels["ch3"] = int(data.get("chan3_raw", 1000))
                    self.rc_channels["ch4"] = int(data.get("chan4_raw", 1500))
                    self.rc_channels["ch5"] = int(data.get("chan5_raw", 1000))
                    self.rc_channels["ch6"] = int(data.get("chan6_raw", 1000))
                    self.rc_channels["ch7"] = int(data.get("chan7_raw", 1000))
                    self.rc_channels["ch8"] = int(data.get("chan8_raw", 1000))
                except Exception:
                    pass
        except Exception:
            pass

    # ------------------ High-Frequency Logging Loop ------------------ #

    async def _sampling_loop(self):
        """Record state snapshot to CSV at high frequency."""
        while self.running:
            t_now = time.time()
            elapsed = round(t_now - (self.start_time or t_now), 3)

            m1 = self.motor_pwms["m1"]
            m2 = self.motor_pwms["m2"]
            m3 = self.motor_pwms["m3"]
            m4 = self.motor_pwms["m4"]
            avg_pwm = round((m1 + m2 + m3 + m4) / 4.0, 1)

            alt_m = round(self.relative_altitude_m, 3)
            alt_cm = round(self.relative_altitude_m * 100.0, 1)

            record = {
                "timestamp_iso": datetime.datetime.now().isoformat(),
                "elapsed_s": elapsed,
                "armed": 1 if self.armed else 0,
                "in_air": 1 if self.in_air else 0,
                "flight_mode": self.flight_mode,
                "rc_roll_us": self.rc_channels["ch1"],
                "rc_pitch_us": self.rc_channels["ch2"],
                "rc_throttle_us": self.rc_channels["ch3"],
                "rc_yaw_us": self.rc_channels["ch4"],
                "rc_ch5_us": self.rc_channels["ch5"],
                "rc_ch6_us": self.rc_channels["ch6"],
                "rc_ch7_us": self.rc_channels["ch7"],
                "rc_ch8_us": self.rc_channels["ch8"],
                "motor1_pwm_us": m1,
                "motor2_pwm_us": m2,
                "motor3_pwm_us": m3,
                "motor4_pwm_us": m4,
                "avg_motor_pwm_us": avg_pwm,
                "altitude_m": alt_m,
                "altitude_cm": alt_cm,
                "attitude_roll_deg": round(self.roll_deg, 2),
                "attitude_pitch_deg": round(self.pitch_deg, 2),
                "heading_yaw_deg": round(self.yaw_deg, 1),
                "battery_v": round(self.battery_voltage, 2),
                "battery_pct": round(self.battery_percent, 1),
            }

            self.samples.append(record)
            self.writer.writerow(record)
            if len(self.samples) % 10 == 0:
                self.csv_file.flush()

            await asyncio.sleep(self.sample_interval)

    # ------------------ Live Terminal Dashboard ------------------ #

    async def _display_loop(self):
        """Render a clean live terminal dashboard (refreshes every 0.15s)."""
        def _bar(val: int, min_val: int = 1000, max_val: int = 2000, width: int = 16) -> str:
            frac = max(0.0, min(1.0, (val - min_val) / max(1.0, (max_val - min_val))))
            filled = int(frac * width)
            return "█" * filled + "░" * (width - filled)

        def _center_bar(val: int, min_val: int = 1000, max_val: int = 2000, width: int = 16) -> str:
            diff = (val - 1500) / 500.0
            half = width // 2
            if abs(diff) < 0.05:
                return "░" * half + "|" + "░" * half
            if diff > 0:
                filled = min(half, int(diff * half))
                return "░" * half + "|" + "█" * filled + "░" * (half - filled)
            else:
                filled = min(half, int(abs(diff) * half))
                return "░" * (half - filled) + "█" * filled + "|" + "░" * half

        while self.running:
            elapsed = time.time() - (self.start_time or time.time())
            mins = int(elapsed // 60)
            secs = elapsed % 60

            armed_badge = "\033[92m[ ARMED ]\033[0m" if self.armed else "\033[91m[DISARMED]\033[0m"
            air_badge = "\033[96m[AIRBORNE]\033[0m" if self.in_air else "\033[90m[GROUND]\033[0m"

            m1 = self.motor_pwms["m1"]
            m2 = self.motor_pwms["m2"]
            m3 = self.motor_pwms["m3"]
            m4 = self.motor_pwms["m4"]
            avg_m = (m1 + m2 + m3 + m4) / 4.0

            r_roll = self.rc_channels["ch1"]
            r_pitch = self.rc_channels["ch2"]
            r_thr = self.rc_channels["ch3"]
            r_yaw = self.rc_channels["ch4"]

            in_air_samples = [s for s in self.samples if s.get("in_air", 0) == 1]
            if in_air_samples:
                hover_pwms = [s["avg_motor_pwm_us"] for s in in_air_samples]
                hover_avg = sum(hover_pwms) / len(hover_pwms)
                hover_min = min(hover_pwms)
                hover_max = max(hover_pwms)
                hover_str = f"{hover_avg:.0f} µs (Range: {hover_min:.0f} - {hover_max:.0f} µs)"
            else:
                hover_str = "Waiting for liftoff..."

            sys.stdout.write("\033[H\033[J")
            dashboard = f"""
================================================================================
 🛸 TYPEFLY RC FLIGHT RECORDER & MOTOR ANALYZER
================================================================================
 Link:    {self.address}
 State:   {armed_badge}  {air_badge}  | Mode: \033[93m{self.flight_mode}\033[0m
 Flight:  Time: \033[97m{mins:02d}:{secs:04.1f}\033[0m | Battery: \033[92m{self.battery_voltage:.1f}V ({self.battery_percent:.0f}%)\033[0m
 Height:  \033[96m{self.relative_altitude_m:.2f} M ({self.relative_altitude_m*100:.0f} CM)\033[0m | Roll: {self.roll_deg:+.1f}° | Pitch: {self.pitch_deg:+.1f}°
--------------------------------------------------------------------------------
 🎮 PILOT RC TRANSMITTER INPUTS (Microseconds):
  Throttle [CH3]: {r_thr:4d} µs  [{_bar(r_thr)}]  {((r_thr-1000)/10.0):.0f}%
  Roll     [CH1]: {r_roll:4d} µs  [{_center_bar(r_roll)}]
  Pitch    [CH2]: {r_pitch:4d} µs  [{_center_bar(r_pitch)}]
  Yaw      [CH4]: {r_yaw:4d} µs  [{_center_bar(r_yaw)}]
  Aux CH5 (Mode): {self.rc_channels['ch5']:4d} µs | CH6: {self.rc_channels['ch6']:4d} µs | CH7: {self.rc_channels['ch7']:4d} µs
--------------------------------------------------------------------------------
 ⚡ LIVE DRONE MOTOR OUTPUTS (ESC PWMs):
  M1 (Front Right, CCW): {m1:4d} µs  [{_bar(m1)}]
  M2 (Rear Left,   CCW): {m2:4d} µs  [{_bar(m2)}]
  M3 (Front Left,   CW): {m3:4d} µs  [{_bar(m3)}]
  M4 (Rear Right,   CW): {m4:4d} µs  [{_bar(m4)}]
  Live Average Motor PWM: \033[93m{avg_m:4.0f} µs\033[0m
--------------------------------------------------------------------------------
 📊 FLIGHT DATA STATS:
  Samples Recorded:  {len(self.samples)} samples ({self.sample_interval*1000:.0f}ms interval)
  Detected Hover PWM: \033[92m{hover_str}\033[0m
  Saved Log File:    {self.output_csv}
--------------------------------------------------------------------------------
 [Press Ctrl+C to stop recording and generate calibration report]
================================================================================
"""
            sys.stdout.write(dashboard)
            sys.stdout.flush()
            await asyncio.sleep(0.15)

    def close(self):
        """Close log file and compute report."""
        self.running = False
        try:
            self.csv_file.flush()
            self.csv_file.close()
        except Exception:
            pass

    def generate_analysis_report(self):
        """Analyze recorded flight samples and print motor calibration summary."""
        if not self.samples:
            print("\n[Analysis] No samples recorded.")
            return

        print("\n\n" + "=" * 80)
        print(" 📋 TYPEFLY RC FLIGHT ANALYSIS & MOTOR CALIBRATION REPORT")
        print("=" * 80)

        total_samples = len(self.samples)
        total_time = self.samples[-1]["elapsed_s"]

        armed_samples = [s for s in self.samples if s["armed"] == 1]
        airborne_samples = [s for s in self.samples if s["in_air"] == 1 or s["altitude_m"] >= 0.15]
        ground_armed_samples = [s for s in armed_samples if s not in airborne_samples]

        print(f" Flight Summary:")
        print(f"  • Total Duration:     {total_time:.1f} seconds")
        print(f"  • Total Samples:      {total_samples} samples")
        print(f"  • Time Armed:         {len(armed_samples)*self.sample_interval:.1f} seconds")
        print(f"  • Time Airborne:      {len(airborne_samples)*self.sample_interval:.1f} seconds")
        max_alt = max(s["altitude_m"] for s in self.samples)
        print(f"  • Max Altitude:       {max_alt:.2f} m ({max_alt*100:.0f} cm)")

        print("\n ⚡ Motor PWM Operating Ranges (µs):")

        # Ground Idle
        if ground_armed_samples:
            idle_pwms = [s["avg_motor_pwm_us"] for s in ground_armed_samples if s["avg_motor_pwm_us"] > 1020]
            if idle_pwms:
                print(f"  • Ground Armed Idle:   {sum(idle_pwms)/len(idle_pwms):.0f} µs (Min: {min(idle_pwms):.0f}, Max: {max(idle_pwms):.0f})")

        # In-Air Hover
        if airborne_samples:
            hover_samples = [s for s in airborne_samples if 0.20 <= s["altitude_m"] <= 1.20]
            target_set = hover_samples if hover_samples else airborne_samples

            m1_vals = [s["motor1_pwm_us"] for s in target_set]
            m2_vals = [s["motor2_pwm_us"] for s in target_set]
            m3_vals = [s["motor3_pwm_us"] for s in target_set]
            m4_vals = [s["motor4_pwm_us"] for s in target_set]
            avg_vals = [s["avg_motor_pwm_us"] for s in target_set]

            m1_avg = sum(m1_vals) / len(m1_vals)
            m2_avg = sum(m2_vals) / len(m2_vals)
            m3_avg = sum(m3_vals) / len(m3_vals)
            m4_avg = sum(m4_vals) / len(m4_vals)
            overall_hover_avg = sum(avg_vals) / len(avg_vals)

            print(f"  • Stable Hover PWM:    {overall_hover_avg:.0f} µs (Min: {min(avg_vals):.0f}, Max: {max(avg_vals):.0f})")
            print(f"  • Motor Balance (M1..M4):")
            print(f"      - M1 (Front-Right, CCW): {m1_avg:.0f} µs")
            print(f"      - M2 (Rear-Left,   CCW): {m2_avg:.0f} µs")
            print(f"      - M3 (Front-Left,   CW): {m3_avg:.0f} µs")
            print(f"      - M4 (Rear-Right,   CW): {m4_avg:.0f} µs")

            pitch_bias = (m2_avg + m4_avg) - (m1_avg + m3_avg)
            roll_bias = (m2_avg + m3_avg) - (m1_avg + m4_avg)
            print(f"  • Trim / Center-of-Gravity Bias:")
            if abs(pitch_bias) > 30:
                tendency = "tail-heavy (rear motors working harder)" if pitch_bias > 0 else "nose-heavy (front motors working harder)"
                print(f"      - Pitch Bias: {pitch_bias:+.0f} µs ({tendency})")
            else:
                print(f"      - Pitch Bias: {pitch_bias:+.0f} µs (Well-balanced front/rear)")

            if abs(roll_bias) > 30:
                tendency = "left-heavy (left motors working harder)" if roll_bias > 0 else "right-heavy (right motors working harder)"
                print(f"      - Roll Bias:  {roll_bias:+.0f} µs ({tendency})")
            else:
                print(f"      - Roll Bias:  {roll_bias:+.0f} µs (Well-balanced left/right)")

            abs_max_pwm = max(max(s["motor1_pwm_us"], s["motor2_pwm_us"], s["motor3_pwm_us"], s["motor4_pwm_us"]) for s in self.samples)
            print(f"  • Peak Motor PWM Seen: {abs_max_pwm:.0f} µs")

            print("\n" + "-" * 80)
            print(" 🎯 RECOMMENDED SETTINGS TO APPLY TO TYPEFLY WEBUI:")
            print("-" * 80)
            print(f"  1. Recommended Hover Base PWM:   {overall_hover_avg:.0f} µs")
            norm_thrust = round(max(0.0, min(1.0, (overall_hover_avg - 1000.0) / 1000.0)), 2)
            print(f"  2. Normalized Hover Thrust:      {norm_thrust:.2f} ({(norm_thrust*100):.0f}% throttle)")
            print(f"  3. Safe Bench Test Low Throttle: {max(1150, int(overall_hover_avg * 0.82))} µs")
            print(f"  4. Gentle Takeoff Spool PWM:     {max(1200, int(overall_hover_avg * 0.90))} µs")
            print("-" * 80)
        else:
            print("  (Drone was not airborne long enough to compute hover statistics)")

        summary_txt_path = self.output_csv.replace(".csv", "_summary.txt")
        try:
            with open(summary_txt_path, "w", encoding="utf-8") as f:
                f.write(f"TypeFly RC Flight Analysis Report\n")
                f.write(f"Date: {datetime.datetime.now().isoformat()}\n")
                f.write(f"CSV Log: {self.output_csv}\n")
                f.write(f"Total Duration: {total_time:.1f}s, Samples: {total_samples}\n")
                if airborne_samples:
                    f.write(f"Hover Average PWM: {overall_hover_avg:.1f} us\n")
                    f.write(f"M1: {m1_avg:.1f}, M2: {m2_avg:.1f}, M3: {m3_avg:.1f}, M4: {m4_avg:.1f}\n")
            print(f" Saved Text Summary: {summary_txt_path}")
        except Exception:
            pass

        print(f" Full Detailed CSV Log: {self.output_csv}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="TypeFly RC Flight Recorder: Record motor speeds and RC commands during manual flight via Radio module."
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default=None,
        help="Serial port or connection profile ('radio', 'usb', '/dev/ttyUSB0', etc.). Defaults to auto-detecting the CP2102 radio module.",
    )
    parser.add_argument(
        "--baud", "-b",
        type=int,
        default=57600,
        help="Baud rate for serial connection (default: 57600).",
    )
    parser.add_argument(
        "--rate", "-r",
        type=float,
        default=20.0,
        help="Recording frequency in Hz (default: 20 Hz / 50ms).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path for output CSV file. Defaults to 'flight_logs/rc_flight_YYYYMMDD_HHMMSS.csv'.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable interactive terminal dashboard (log in background).",
    )
    return parser.parse_args()


async def main_async():
    args = parse_args()

    address = resolve_connection_address(args.device, args.baud)

    if args.output:
        output_csv = args.output
    else:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = os.path.join("flight_logs", f"rc_flight_{timestamp_str}.csv")

    recorder = FlightRecorder(
        address=address,
        output_csv=output_csv,
        sample_rate_hz=args.rate,
        no_display=args.no_display,
    )

    loop = asyncio.get_running_loop()

    def _sig_handler():
        recorder.running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _sig_handler)
        except (NotImplementedError, RuntimeError):
            pass

    try:
        await recorder.start()
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        recorder.close()
        recorder.generate_analysis_report()


def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n[Recorder] Finished.")


if __name__ == "__main__":
    main()
