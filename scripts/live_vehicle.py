"""Advance one vehicle in the CAN thread; renderers only read snapshots."""
from __future__ import annotations

import argparse
from collections import deque
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import threading
import time

from vehicle import VehicleController


class LiveVehicle:
    def __init__(self, config_path, pose, link, log_dir):
        self._ctrl = VehicleController(config_path)
        self._ctrl.reset(pose["x"], pose["y"], pose["yaw"])
        self.link = link
        self._lock = threading.Lock()
        self._manual = (0.0, 0.0, "P", False)
        self._last_t = None
        self._history = deque(maxlen=3000)
        self._history.append((time.monotonic(), self._ctrl.get_state()))
        self.sequence = 0
        self.reset_index = 0
        self.dt = 0.0
        self.source = "keys"
        self.overruns = 0
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "run_meta.json").write_text(json.dumps(dict(
            vehicle_config=str(Path(config_path).resolve()),
            vehicle_config_yaml=Path(config_path).read_text(), target_period_s=0.01,
            odom_alignment="local_receive_time; MCU source clock offset is unknown"), indent=2) + "\n")
        self._trace = (self.log_dir / "sim_truth.csv").open("w", newline="", buffering=65536)
        self._writer = csv.writer(self._trace)
        self._writer.writerow(("sim_mono", "sim_wall", "sequence", "reset_index", "dt", "overruns",
                               "x", "y", "yaw", "speed_mps", "steer_deg", "gear", "yaw_rate_dps",
                               "pulse_fl", "pulse_fr", "pulse_rl", "pulse_rr", "source", "brake"))

    def start(self):
        self.link.enable_trace(self.log_dir / "can_tx.csv")
        self.link.start(self.tick)

    def set_controls(self, steer, accel, gear, brake_pressed=False):
        with self._lock:
            self._manual = (float(steer), float(accel), str(gear), bool(brake_pressed))

    def get_state(self):
        with self._lock:
            return self._ctrl.get_state()

    def timed_state(self):
        """Return a physics pose and its source time from the same locked sample."""
        with self._lock:
            return self._history[-1]

    def status(self):
        with self._lock:
            return dict(sequence=self.sequence, reset_index=self.reset_index, dt=self.dt,
                        source=self.source, overruns=self.overruns)

    def reset(self, x, y, yaw, speed=0.0):
        with self._lock:
            state = self._ctrl.reset(x, y, yaw, speed, preserve_pulses=True)
            self._manual = (0.0, 0.0, "P", False)
            self._last_t = time.monotonic()
            self._history.clear()
            self._history.append((self._last_t, state))
            self.reset_index += 1
            return state

    def apply_to_world(self, world, z=0.0):
        state = self.get_state()
        world.set_ego_pose(state.x, state.y, z, state.yaw)
        return state

    def tick(self, now):
        with self._lock:
            if self._last_t is not None:
                now = max(now, self._last_t)  # a UI reset may have just acquired this lock
            dt = 0.0 if self._last_t is None else max(0.0, now - self._last_t)
            self._last_t = now
            self.dt = dt
            self.overruns += int(dt > 0.02)
            steer, accel, gear, brake = self._manual
            command = self.link.controls(self._ctrl.get_state().gear)
            self.source = "CAN" if command is not None and not brake else "keys"
            if self.source == "CAN":
                steer, accel, gear = command
            self._ctrl.set_controls(steer, accel, gear)
            # Use real elapsed time with <=10 ms numerical steps. A late TX
            # cycle is logged; never turn its missing slots into a send burst.
            steps = max(1, int(math.ceil(dt / 0.01)))
            state = self._ctrl.get_state()
            for _ in range(steps):
                state = self._ctrl.step(dt / steps)
            self._history.append((now, state))
            self.sequence += 1
            self._writer.writerow(("%.9f" % now, "%.6f" % time.time(), self.sequence, self.reset_index,
                                   "%.9f" % dt, self.overruns, "%.9f" % state.x, "%.9f" % state.y,
                                   "%.9f" % state.yaw, state.speed_mps, state.steer_wheel_deg,
                                   state.gear, math.degrees(state.yaw_rate_rps),
                                   *state.wheel_pulse_count, self.source, int(brake)))
            return state, argparse.Namespace(accel=accel, brake_pressed=brake), dt

    def pose_at(self, stamp):
        """Interpolate truth at local receive time, including transport delay.

        MCU source time is in another clock domain. This removes UI/render
        backlog only; it does not claim to synchronize the source clocks.
        """
        with self._lock:
            history = list(self._history)
        if stamp <= history[0][0]:
            return history[0]
        if stamp >= history[-1][0]:
            return history[-1]
        for i in range(len(history) - 2, -1, -1):
            ta, a = history[i]
            if ta <= stamp:
                tb, b = history[i + 1]
                f = (stamp - ta) / (tb - ta)
                yaw_delta = (b.yaw - a.yaw + math.pi) % (2 * math.pi) - math.pi
                return stamp, replace(a, x=a.x + f * (b.x - a.x), y=a.y + f * (b.y - a.y),
                                      yaw=a.yaw + f * yaw_delta)

    def close(self):
        self.link.close()
        with self._lock:
            self._trace.close()
