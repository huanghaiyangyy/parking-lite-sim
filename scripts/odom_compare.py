#!/usr/bin/env python3
"""Read Android wheel odometry forwarded by odom_relay.

Each line is:
    stamp x_wheel y_wheel yaw_wheel x y yaw
The first three pose fields are the MCU wheel-pulse odometry. The last three
are the other odometry output and are kept for display only.
"""
from __future__ import annotations

import math
import csv
from pathlib import Path
import socket
import threading
import time
from odom_trace import TraceCsvWriter


class OdomCompare:
    def __init__(self, serial, port=46000):
        self.serial = serial
        self.port = port
        self._lock = threading.Lock()
        self.sample = None
        self.error = None
        self._stop = threading.Event()
        self._trace = None
        self._trace_writer = None
        self._sequence = 0
        self._mcu_trace = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def status(self):
        with self._lock:
            return None if self.sample is None else dict(self.sample), self.error

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.5)
        with self._lock:
            if self._trace is not None:
                self._trace.close()
                self._trace = self._trace_writer = None
            if self._mcu_trace is not None:
                self._mcu_trace.close()
                self._mcu_trace = None

    def enable_trace(self, path):
        with self._lock:
            self._trace = Path(path).open("w", newline="", buffering=65536)
            self._trace_writer = csv.writer(self._trace)
            self._trace_writer.writerow(("rx_mono", "rx_wall", "sequence", "stamp", "x_wheel",
                                         "y_wheel", "yaw_wheel", "x", "y", "yaw"))
            self._mcu_trace = TraceCsvWriter(Path(path).with_name('mcu_odom_trace.csv'))

    def _run(self):
        # Android odom_relay connects here. adb reverse cannot bind this port.
        server = socket.socket()
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind(("0.0.0.0", self.port))
            server.listen(1)
            server.settimeout(1)
        except Exception as exc:
            server.close()
            with self._lock:
                self.error = "listen: %s" % exc
            return
        try:
            while not self._stop.is_set():
                try:
                    sock, _addr = server.accept()
                except socket.timeout:
                    continue
                with sock:
                    sock.settimeout(0.5)
                    buf = b""
                    while not self._stop.is_set():
                        try:
                            chunk = sock.recv(4096)
                        except socket.timeout:
                            continue
                        except OSError as exc:
                            with self._lock:
                                self.error = str(exc)
                            break
                        if not chunk:
                            break
                        buf += chunk
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            self._take(line)
                        if len(buf) > 4096:
                            break
        finally:
            server.close()

    def _take(self, line):
        if line.startswith(b'ODT1 '):
            with self._lock:
                if self._mcu_trace is not None:
                    self._mcu_trace.take(line)
            return
        parts = line.split()
        if len(parts) != 7:
            return
        try:
            values = [float(item) for item in parts]
        except ValueError:
            return
        if not all(math.isfinite(v) for v in values):
            return
        sample = {
            "stamp": values[0],
            "x": values[1],
            "y": values[2],
            "yaw": values[3],
            "fused_x": values[4],
            "fused_y": values[5],
            "fused_yaw": values[6],
            "received_monotonic": time.monotonic(),
            "received_wall": time.time(),
        }
        with self._lock:
            self._sequence += 1
            sample["sequence"] = self._sequence
            self.sample = sample
            self.error = None
            if self._trace_writer is not None:
                self._trace_writer.writerow(("%.9f" % sample["received_monotonic"],
                                             "%.6f" % sample["received_wall"], self._sequence, *values))


def _wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _body(dx, dy, yaw0):
    """Displacement in the vehicle frame at the comparison origin."""
    c, s = math.cos(yaw0), math.sin(yaw0)
    return dx * c + dy * s, -dx * s + dy * c


def pose_error(sim_x, sim_y, sim_yaw, odom, origin):
    """Compare motion since the first sample.

    dx/dy are forward/left in each source's own frame at that sample.
    The two world frames do not share an origin, so raw x/y are not compared.
    """
    if origin is None:
        return None
    sim_dx, sim_dy = _body(
        sim_x - origin["sim_x"], sim_y - origin["sim_y"], origin["sim_yaw"])
    odom_dx, odom_dy = _body(
        odom["x"] - origin["x"], odom["y"] - origin["y"], origin["yaw"])
    sim_dyaw = _wrap(sim_yaw - origin["sim_yaw"])
    odom_dyaw = _wrap(odom["yaw"] - origin["yaw"])
    return {
        "sim_dx": sim_dx,
        "sim_dy": sim_dy,
        "sim_dyaw": sim_dyaw,
        "odom_dx": odom_dx,
        "odom_dy": odom_dy,
        "odom_dyaw": odom_dyaw,
        "err_dx": odom_dx - sim_dx,
        "err_dy": odom_dy - sim_dy,
        "err_dyaw": _wrap(odom_dyaw - sim_dyaw),
    }
