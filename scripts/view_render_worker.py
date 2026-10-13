#!/usr/bin/env python3
"""Offscreen worker for the live window (god view + 4 fisheye).

ModernGL's standalone context and pygame both call GLX MakeCurrent on the
same X display. Whichever runs second loses: the window stays black, or
fbo.read returns zeros. This process owns GL only and has no DISPLAY.

Commands (one line each):
  god x y z yaw ex ey ez tx ty tz 0 fov
  cams x y z yaw
  quit
"""

from __future__ import annotations

import argparse
import os
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.pop("DISPLAY", None)
os.environ.pop("WAYLAND_DISPLAY", None)

from scene.world import World  # noqa: E402

# Live preview is half of config image size. One pinhole per camera.
_PREVIEW_SCALE = 0.5


def _readline() -> str:
    line = sys.stdin.buffer.readline()
    if not line:
        raise EOFError
    return line.decode().strip()


def _write_frame(img: np.ndarray) -> None:
    h, w = img.shape[:2]
    payload = np.ascontiguousarray(img).tobytes()
    sys.stdout.buffer.write(struct.pack("<III", w, h, len(payload)))
    sys.stdout.buffer.write(payload)


def _scaled_rig(world: World, scale: float):
    """Copy of the camera rig with image size and K scaled together."""
    from sensors.cameras import CameraModel, CameraRig

    src = world.rig
    w = max(160, int(round(src.image_size[0] * scale)))
    h = max(120, int(round(src.image_size[1] * scale)))
    # Keep principal point on the same relative pixel (matches pinhole_K_from_fov).
    sx = (w - 1) / (src.image_size[0] - 1)
    sy = (h - 1) / (src.image_size[1] - 1)
    cams = {}
    for name, cam in src.cameras.items():
        K = np.array(cam.K, dtype=np.float64, copy=True)
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy
        cams[name] = CameraModel(
            name=name,
            translation_m=cam.translation_m.copy(),
            rotation_deg=cam.rotation_deg.copy(),
            K=K,
            D=cam.D.copy(),
        )
    return CameraRig(
        cameras=cams,
        image_size=(w, h),
        render_cfg=dict(src.render_cfg),
        vehicle_cfg=dict(src.vehicle_cfg),
        assets_root=src.assets_root,
    )


def stream_cameras(world, args):
    """Own the GL/encoder cadence; a slow preview consumer cannot delay video."""
    import cv2
    from stream_h264_adb import Encoder, SEND_STATUS, connect_adb
    cv2.setNumThreads(2)
    width, height = world.rig.image_size
    if (width, height) != (1280, 960):
        raise ValueError("board streaming requires 1280x960 per camera")
    world.render_camera_grid()
    sock = (socket.create_connection((args.host, args.port), timeout=10)
            if args.host else connect_adb(args.serial, args.port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(10)
    encoder = Encoder(width*2, height*2, args.fps, args.bitrate, sock.sendall)
    condition, stopped = threading.Condition(), threading.Event()
    pose, snapshot = None, None

    def preview_io():
        nonlocal pose
        try:
            while not stopped.is_set():
                line = _readline()
                if line == "quit":
                    break
                requested = tuple(float(v) for v in line.split())
                if len(requested) != 4:
                    raise ValueError("camera command requires x y z yaw")
                with condition:
                    pose = requested
                    condition.notify_all()
                    condition.wait_for(lambda: snapshot is not None or stopped.is_set())
                    if stopped.is_set():
                        break
                    images, status = snapshot
                for name in world.rig.names:
                    _write_frame(images[name])
                sys.stdout.buffer.write(SEND_STATUS.pack(*status))
                sys.stdout.buffer.flush()
        except (EOFError, BrokenPipeError):
            pass
        finally:
            stopped.set()
            with condition:
                condition.notify_all()

    io = threading.Thread(target=preview_io, daemon=True)
    sys.stdout.buffer.write(("ready %s\n" % " ".join(world.rig.names)).encode())
    sys.stdout.buffer.flush()
    io.start()
    try:
        with condition:
            condition.wait_for(lambda: pose is not None or stopped.is_set())
        start_ns, report, seq = time.monotonic_ns(), time.monotonic(), 0
        next_frame = time.monotonic()
        report_sent, report_bytes, send_fps, send_mbps = 0, 0, 0.0, 0.0
        while not stopped.wait(max(0, next_frame-time.monotonic())):
            with condition:
                captured_pose = pose
            tick_ns = time.monotonic_ns()
            world.set_ego_pose(*captured_pose)
            grid = world.render_camera_grid()
            encoder.push(seq, (tick_ns-start_ns)//1000, grid)
            images = {name: cv2.resize(grid[row*height:(row+1)*height,
                                          col*width:(col+1)*width],
                                      (width//2, height//2), interpolation=cv2.INTER_AREA)
                      for name, (row, col) in dict(front=(0, 0), left=(0, 1),
                                                 right=(1, 0), rear=(1, 1)).items()}
            now = time.monotonic()
            if now-report >= 1:
                send_fps = (encoder.sent-report_sent)/(now-report)
                send_mbps = (encoder.bytes-report_bytes)*8/(now-report)/1e6
                print(f"window_stream rendered={seq+1} sent={encoder.sent} "
                      f"send_fps={send_fps:.2f} Mbps={send_mbps:.2f} "
                      f"report_time={now:.6f} pose={list(captured_pose)}", file=sys.stderr, flush=True)
                report, report_sent, report_bytes = now, encoder.sent, encoder.bytes
            with condition:
                snapshot = images, (encoder.sent, send_fps, send_mbps, seq, *captured_pose)
                condition.notify_all()
            seq += 1
            next_frame = max(next_frame+1/args.fps, time.monotonic())
        encoder.finish()
    finally:
        stopped.set()
        with condition:
            condition.notify_all()
        io.join(timeout=1)
        if encoder.proc.poll() is None:
            encoder.proc.kill(); encoder.proc.wait()
        sock.close()
        world.release()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream-adb", action="store_true")
    parser.add_argument("--serial", default="")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--bitrate", default="12M")
    parser.add_argument("--host", help="direct TCP destination for the stream")
    parser.add_argument("--port", type=int, default=27164)
    parser.add_argument("--capture-scale", type=int, choices=(1, 2, 4))
    args = parser.parse_args()
    if not 1 <= args.fps <= 30:
        parser.error("fps must be 1..30")
    # One mode per process. A second ModernGL context in this process (full-res
    # god view plus half-res fisheye) makes fbo.read come back sheared.
    mode = _readline()
    config = _readline()
    if mode == "god":
        world = World.from_config(config)
        sys.stdout.buffer.write(b"ready\n")
        sys.stdout.buffer.flush()
        while True:
            try:
                line = _readline()
            except EOFError:
                break
            if line == "quit":
                break
            parts = [float(x) for x in line.split()]
            if len(parts) != 12:
                continue
            x, y, z, yaw, ex, ey, ez, tx, ty, tz, _ux, fov = parts
            world.set_ego_pose(x, y, z, yaw)
            img = world.render_god_view(
                eye_flu=(ex, ey, ez),
                target_flu=(tx, ty, tz),
                up_flu=(0.0, 0.0, 1.0),
                fov_y_deg=fov,
            )
            _write_frame(img)
            sys.stdout.buffer.flush()
        world.release()
        return 0

    if mode == "cams":
        world = World.from_config(config)
        if args.capture_scale is not None:
            world.rig.render_cfg["pinhole_capture_scale"] = args.capture_scale
        if args.stream_adb:
            return stream_cameras(world, args)
        names = tuple(world.rig.names)
        preview = _scaled_rig(world, _PREVIEW_SCALE)
        world.release()
        try:
            sys.stdout.buffer.write(("ready %s\n" % " ".join(names)).encode())
            sys.stdout.buffer.flush()
            while True:
                try:
                    line = _readline()
                except EOFError:
                    break
                if line == "quit":
                    break
                parts = [float(x) for x in line.split()]
                if len(parts) != 4:
                    continue
                images = preview.render_cameras(tuple(parts))
                for name in names:
                    _write_frame(images[name])
                sys.stdout.buffer.flush()
        finally:
            preview.release()
        return 0

    sys.stderr.write("view_render_worker: expected god or cams, got %r\n" % mode)
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(0)
