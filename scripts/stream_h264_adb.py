#!/usr/bin/env python3
"""One persistent VAAPI encoder, one 2x2 H264 stream, one AU per PLIT v2 packet.

framecrc is the first tee slave: it announces packet length before the raw
slave writes stdout. This avoids waiting for the next AUD or a whole second.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import queue
import re
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HEADER = struct.Struct("<4sBBHBBIQIHH")
SEND_STATUS = struct.Struct("<QddQ4d")  # sent, fps, Mbps, preview sequence, capture pose
MAX_PAYLOAD = 4 * 1024 * 1024


def sequence_color(seq):
    """512 distinct RGB codes; adjacent frames change one Gray-code bit."""
    value = seq % 512
    gray = value ^ (value >> 1)
    return tuple(40 + 24 * ((gray >> shift) & 7) for shift in (0, 3, 6))


def _nal_types(data: bytes) -> set[int]:
    return {data[m.end()] & 31 for m in re.finditer(b"\x00\x00(?:\x00)?\x01", data)
            if m.end() < len(data)}


def _pack(seq, pts_us, width, height, au):
    if not 0 < len(au) <= MAX_PAYLOAD:
        raise ValueError("invalid encoded AU length")
    return HEADER.pack(b"PLIT", 2, 1, int(5 in _nal_types(au)), 255, 1,
                       seq & 0xffffffff, pts_us, len(au), width, height) + au


def _rgb_to_nv12(rgb):
    h, w = rgb.shape[:2]
    i420 = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV_I420)
    u = i420[h:h+h//4].reshape(h//2, w//2)
    v = i420[h+h//4:].reshape(h//2, w//2)
    uv = np.empty((h//2, w), dtype=np.uint8)
    uv[:, 0::2], uv[:, 1::2] = u, v
    return i420[:h].tobytes() + uv.tobytes()


def _read_exact(stream, size):
    result = bytearray()
    while len(result) < size:
        block = stream.read(size-len(result))
        if not block:
            raise EOFError("encoder ended inside a packet")
        result.extend(block)
    return bytes(result)


def connect_adb(serial="", port=27164):
    if not serial:
        devices = subprocess.check_output(["adb", "devices"], text=True)
        found = [line.split()[0] for line in devices.splitlines()[1:]
                 if line.strip().endswith("\tdevice")]
        if len(found) != 1:
            raise ValueError("select exactly one adb device using --serial")
        serial = found[0]
    # A newly created forward can print its port. Camera workers reserve stdout
    # for the binary preview protocol, so keep adb diagnostics on stderr only.
    subprocess.run(["adb", "-s", serial, "forward", f"tcp:{port}", f"tcp:{port}"],
                   check=True, stdout=subprocess.DEVNULL)
    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(10)
    return sock


class Encoder:
    def __init__(self, width, height, fps, bitrate, emit):
        self.width, self.height = width, height
        self.pending = queue.Queue(maxsize=16)
        self.emit, self.error, self.sent, self.bytes = emit, None, 0, 0
        read_fd, write_fd = os.pipe()
        self.lengths = os.fdopen(read_fd, "rb")
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "2",
               "-filter_threads", "2", "-vaapi_device", "/dev/dri/renderD128",
               "-probesize", "32", "-analyzeduration", "0", "-f", "rawvideo",
               "-pix_fmt", "nv12", "-s", f"{width}x{height}", "-r", str(fps),
               "-i", "pipe:0", "-map", "0:v", "-vf", "hwupload", "-an",
               "-c:v", "h264_vaapi", "-b:v", bitrate, "-maxrate", bitrate,
               "-g", str(fps), "-bf", "0", "-aud", "1", "-flags", "-global_header",
               "-color_range", "tv", "-colorspace", "smpte170m", "-f", "tee",
               f"[f=framecrc:flush_packets=1]pipe:{write_fd}|[f=h264:flush_packets=1]pipe:1"]
        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         pass_fds=(write_fd,), bufsize=0)
        finally:
            os.close(write_fd)
        self.reader = threading.Thread(target=self._drain, daemon=True)
        self.reader.start()

    def _drain(self):
        try:
            for line in self.lengths:
                if line.startswith(b"#") or not line.strip():
                    continue
                fields = line.split(b",")
                if len(fields) < 6 or int(fields[0]) != 0:
                    raise ValueError(f"unexpected framecrc line: {line!r}")
                size = int(fields[4])
                if not 0 < size <= MAX_PAYLOAD:
                    raise ValueError("encoded AU exceeds protocol limit")
                au = _read_exact(self.proc.stdout, size)
                seq, pts = self.pending.get(timeout=5)
                packet = _pack(seq, pts, self.width, self.height, au)
                self.emit(packet)
                self.sent += 1
                self.bytes += len(packet)
        except BaseException as exc:
            self.error = exc
            self.proc.kill()

    def push(self, seq, pts, rgb):
        if self.error:
            raise RuntimeError("encoder/transport reader failed") from self.error
        self.pending.put((seq, pts), timeout=5)
        view = memoryview(_rgb_to_nv12(rgb))
        while view:
            n = self.proc.stdin.write(view)
            if not n:
                raise BrokenPipeError("encoder input closed")
            view = view[n:]

    def finish(self):
        self.proc.stdin.close()
        try:
            code = self.proc.wait(timeout=15)
        except BaseException:
            self.proc.kill(); self.proc.wait()
            raise
        self.reader.join(timeout=10)
        self.lengths.close()
        if self.reader.is_alive() or code or self.error:
            raise RuntimeError(f"encoder drain failed: exit={code}, error={self.error}")
        if not self.pending.empty():
            raise RuntimeError("encoder did not output every submitted frame")


def _test_pattern(width, height, seq, fps):
    colors = ((230, 30, 30), (30, 220, 30), (30, 30, 230), (220, 220, 30))
    tw, th = width//2, height//2
    image = np.empty((height, width, 3), dtype=np.uint8)
    yy, xx = np.indices((th, tw))
    grid = (((xx + seq*8)//64 + yy//64) & 1).astype(bool)
    for cam in range(4):
        color = np.asarray(colors[(cam + seq//fps) % 4], dtype=np.uint8)
        tile = image[(cam//2)*th:(cam//2+1)*th, (cam%2)*tw:(cam%2+1)*tw]
        tile[:] = color
        tile[grid] = color//2
    return image


def self_test():
    packets = []
    enc = Encoder(1280, 960, 10, "4M", packets.append)
    for seq in range(8):
        enc.push(seq, seq*100000, _test_pattern(1280, 960, seq, 10))
    enc.finish()
    assert len(packets) == 8
    for seq, packet in enumerate(packets):
        h = HEADER.unpack_from(packet)
        assert h[:3] == (b"PLIT", 2, 1) and h[6:8] == (seq, seq*100000)
        assert h[8] == len(packet)-HEADER.size and h[9:] == (1280, 960)
        assert _nal_types(packet[HEADER.size:]) & {1, 5}
    assert {5, 7, 8}.issubset(_nal_types(packets[0][HEADER.size:]))
    print("PASS: 8 complete AUs; correct sequence/PTS; SPS/PPS/IDR present")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT/"configs/j01.yaml")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--bitrate", default="12M", help="total four-camera bitrate")
    p.add_argument("--scale", type=float, choices=(1.0, 0.5), default=1.0)
    p.add_argument("--capture-scale", type=int, choices=(1, 2, 4), default=4,
                   help="internal sampling scale (default: %(default)s); does not change camera output size")
    p.add_argument("--seconds", type=float, default=0.0)
    p.add_argument("--loop-motion", action="store_true",
                   help="move back and forth inside the lot for continuous viewing")
    p.add_argument("--port", type=int, default=27164)
    p.add_argument("--serial", default="")
    p.add_argument("--host", help="direct TCP destination; default uses adb")
    p.add_argument("--file", type=Path)
    patterns = p.add_mutually_exclusive_group()
    patterns.add_argument("--test-pattern", action="store_true")
    patterns.add_argument("--sequence-pattern", action="store_true",
                          help="uniform 512-frame color code for visible latency/coherence checks")
    p.add_argument("--frame-log", type=Path,
                   help="JSONL of source and send monotonic timestamps, one record per AU")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)
    cv2.setNumThreads(2)
    if args.self_test:
        self_test(); return 0
    if not 1 <= args.fps <= 30 or args.seconds < 0:
        p.error("fps must be 1..30; seconds must be nonnegative")
    world = None
    if args.test_pattern or args.sequence_pattern:
        width, height = int(2560*args.scale), int(1920*args.scale)
    else:
        from scene.world import World
        world = World.from_config(args.config)
        world.rig.render_cfg["pinhole_capture_scale"] = args.capture_scale
        if args.scale != 1.0:
            from scripts.view_render_worker import _scaled_rig
            world.rig = _scaled_rig(world, args.scale)
        width, height = (v*2 for v in world.rig.image_size)
    if (width, height) not in ((2560, 1920), (1280, 960)):
        p.error("AIS stream requires 1280x960 or 640x480 per camera")
    sock, output, encoder, frame_log = None, None, None, None
    try:
        if world:
            # Build GL shaders / KB textures before assigning live frame PTS.
            world.render_camera_grid()
        if args.file:
            args.file.parent.mkdir(parents=True, exist_ok=True)
            output = args.file.open("wb")
            emit = output.write
        else:
            sock = (socket.create_connection((args.host, args.port), timeout=10)
                    if args.host else connect_adb(args.serial, args.port))
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(10)
            emit = sock.sendall
        if args.frame_log:
            args.frame_log.parent.mkdir(parents=True, exist_ok=True)
            frame_log = args.frame_log.open("w", buffering=1)
        start_ns = time.monotonic_ns()
        transport_emit = emit

        def timed_emit(packet):
            transport_emit(packet)
            if frame_log:
                header = HEADER.unpack_from(packet)
                frame_log.write(json.dumps(dict(seq=header[6],
                    source_ns=start_ns + header[7]*1000,
                    sent_ns=time.monotonic_ns(), bytes=len(packet),
                    pattern="sequence512" if args.sequence_pattern else None)) + "\n")

        encoder = Encoder(width, height, args.fps, args.bitrate, timed_emit)
        start, report, seq = start_ns/1e9, time.monotonic(), 0
        try:
            while not args.seconds or time.monotonic()-start < args.seconds:
                tick_ns = time.monotonic_ns()
                tick = tick_ns/1e9
                if args.sequence_pattern:
                    image = np.empty((height, width, 3), dtype=np.uint8)
                    image[:] = sequence_color(seq)
                elif args.test_pattern:
                    image = _test_pattern(width, height, seq, args.fps)
                else:
                    distance = 4*math.sin(.1*(tick-start)) if args.loop_motion else .4*(tick-start)
                    world.set_ego_pose(.5 + distance, 1.5, 0., 0.)
                    image = world.render_camera_grid()
                encoder.push(seq, (tick_ns-start_ns)//1000, image)
                seq += 1
                now = time.monotonic()
                if now-report >= 1:
                    elapsed = now-start
                    print(f"rendered={seq} render_fps={seq/elapsed:.2f} sent={encoder.sent} "
                          f"send_fps={encoder.sent/elapsed:.2f} Mbps={encoder.bytes*8/elapsed/1e6:.2f}", flush=True)
                    report = now
                delay = 1/args.fps-(time.monotonic()-tick)
                if delay > 0: time.sleep(delay)
        except KeyboardInterrupt:
            pass
        encoder.finish()
        print(f"complete: submitted={seq} sent={encoder.sent} bytes={encoder.bytes}")
    finally:
        if encoder and encoder.proc.poll() is None:
            encoder.proc.kill(); encoder.proc.wait()
        if output: output.close()
        if frame_log: frame_log.close()
        if sock: sock.close()
        if world: world.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
