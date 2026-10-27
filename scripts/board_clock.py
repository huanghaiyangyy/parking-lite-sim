#!/usr/bin/env python3
"""8155 boot-microsecond clock, sampled from the TX2 before a frame is drawn.

Images use the MCU-compatible boot clock returned by Pylon on the 8155.
Raw ClockCycles has a different origin and must not be used as image PTS.
The TX2 keeps its own monotonic clock only as a local ruler. Each sample asks
the board for boot microseconds, then

    board_us(now) = sample.board_us + (tx2_now - sample.tx2_us)

Map the physics pose's source time, even when drawing or transport happens
later. camera_sensors multiplies this by 1000 and applies ToPylonTime.
"""
import ctypes
import socket
import subprocess
import threading
import time

# CLOCK uses a dedicated port, separate from the video stream.


class _Timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


def _monotonic_us():
    if hasattr(time, "monotonic_ns"):
        return time.monotonic_ns() // 1000
    ts = _Timespec()
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.clock_gettime(1, ctypes.byref(ts)) != 0:  # CLOCK_MONOTONIC
        raise OSError(ctypes.get_errno(), "clock_gettime")
    return ts.tv_sec * 1000000 + ts.tv_nsec // 1000


class BoardClock:
    """Short-lived offset from the TX2 monotonic clock onto the 8155 boot clock."""

    def __init__(self, serial="", port=27165, resample_s=5.0, max_rtt_us=40000, timeout_s=6.0):
        if resample_s <= 0:
            raise ValueError("resample interval must be positive")
        self.serial = serial
        self.port = port
        self.resample_s = resample_s
        self.max_rtt_us = max_rtt_us
        self.timeout_s = timeout_s
        self.tx2_us = 0
        self.board_us = 0
        self.rtt_us = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        if not self._ready.wait(self.timeout_s):
            self.close()
            raise RuntimeError("8155 clock sample timed out") from self._error

    def _tx2_us(self):
        return _monotonic_us()

    def _forward(self):
        adb = ["adb"]
        if self.serial:
            adb += ["-s", self.serial]
        subprocess.run(adb + ["forward", "tcp:%d" % self.port, "tcp:%d" % self.port],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10,
                       check=True)

    def _read_sample(self):
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=5)
        try:
            sock.settimeout(5)
            before = self._tx2_us()
            sock.sendall(b"CLOCK\n")
            data = b""
            while b"\n" not in data and len(data) < 64:
                block = sock.recv(64)
                if not block:
                    break
                data += block
            after = self._tx2_us()
        finally:
            sock.close()
        text = data.split(b"\n", 1)[0].strip()
        if not text.isdigit():
            raise RuntimeError("8155 clock sample missing: %r" % data)
        return after - before, before + (after - before) // 2, int(text)

    def _loop(self):
        while not self._stop.is_set():
            try:
                self._forward()
                rtt, tx2_us, board_us = min(self._read_sample() for _ in range(20))
                if rtt > self.max_rtt_us:
                    raise RuntimeError("8155 clock RTT %d us exceeds %d us" %
                                       (rtt, self.max_rtt_us))
            except (OSError, socket.error, subprocess.SubprocessError, RuntimeError) as exc:
                self._error = exc
                self._stop.wait(1.0)
                continue
            with self._lock:
                self.rtt_us, self.tx2_us, self.board_us = rtt, tx2_us, board_us
            self._ready.set()
            self._stop.wait(self.resample_s)

    def now_us(self):
        return self.to_board_us(self._tx2_us())

    def to_board_us(self, source_mono_us):
        """Convert the source pose time; never replace it with render/send time."""
        with self._lock:
            if not self.board_us:
                raise RuntimeError("8155 clock is not sampled yet")
            age_us = self._tx2_us() - self.tx2_us
            if age_us > max(3 * self.resample_s, 5.0) * 1000000:
                raise RuntimeError("8155 clock sample is stale (%d us)" % age_us)
            return self.board_us + int(source_mono_us) - self.tx2_us

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)
