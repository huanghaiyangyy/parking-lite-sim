#!/usr/bin/env python3
"""Inspect the parking scene: orbit stills, turntable GIF, optional live window.

The sim itself is offscreen ModernGL (no editor viewport). This script is the
viewer: it poses a free pinhole camera in world FLU and writes images. A live
orbit window is attempted only when pygame can open a display.
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene.world import World
from stream_h264_adb import SEND_STATUS, _read_exact


def _save(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)
    print("  wrote %s shape=%s" % (path, arr.shape))


def _label(arr: np.ndarray, text: str) -> Image.Image:
    im = Image.fromarray(arr).copy()
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    pad = 6
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([0, 0, tw + 2 * pad, th + 2 * pad], fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 80), font=font)
    return im


def _ego_target_flu(world: World, height_m: float = 0.65) -> Tuple[float, float, float]:
    """Look-at point: geometric centre of the ego (rear-axle pose + overhang)."""
    e = world.ego_pose
    v = world.config.get("vehicle") or {}
    length = float(v.get("length_m", 4.4))
    rear_oh = v.get("rear_overhang_m")
    fwd = (length * 0.5 - float(rear_oh)) if rear_oh is not None else 0.0
    yaw = float(e["yaw"])
    x = float(e["x"]) + fwd * math.cos(yaw)
    y = float(e["y"]) + fwd * math.sin(yaw)
    return (x, y, height_m)


def orbit_eye(
    target: Tuple[float, float, float],
    azimuth_deg: float,
    elevation_deg: float,
    radius_m: float,
) -> Tuple[float, float, float]:
    """World-FLU eye position. azimuth 0 = in front of +x, 90 = +y (left)."""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    ce, se = math.cos(el), math.sin(el)
    dx = radius_m * ce * math.cos(az)
    dy = radius_m * ce * math.sin(az)
    dz = radius_m * se
    return (target[0] + dx, target[1] + dy, target[2] + dz)


def render_orbit(
    world: World,
    out_dir: Path,
    n: int = 8,
    radius_m: float = 8.5,
    elevation_deg: float = 22.0,
) -> List[np.ndarray]:
    target = _ego_target_flu(world)
    frames: List[np.ndarray] = []
    for i in range(n):
        az = 360.0 * i / n
        eye = orbit_eye(target, az, elevation_deg, radius_m)
        img = world.render_god_view(
            eye_flu=eye, target_flu=target, up_flu=(0.0, 0.0, 1.0), fov_y_deg=40.0
        )
        frames.append(img)
        _save(out_dir / ("orbit_%02d_az%03d.png" % (i, int(az))), img)
    return frames


def build_orbit_collage(frames: List[np.ndarray], n_cols: int = 4) -> Image.Image:
    n = len(frames)
    n_rows = (n + n_cols - 1) // n_cols
    cell_w, cell_h = 420, 280
    gap, margin = 8, 12
    canvas = Image.new(
        "RGB",
        (
            margin * 2 + n_cols * cell_w + (n_cols - 1) * gap,
            margin * 2 + n_rows * cell_h + (n_rows - 1) * gap,
        ),
        (24, 24, 28),
    )
    for i, arr in enumerate(frames):
        col, row = i % n_cols, i // n_cols
        az = 360.0 * i / n
        im = _label(arr, "orbit az=%.0f" % az)
        im.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
        x = margin + col * (cell_w + gap) + (cell_w - im.size[0]) // 2
        y = margin + row * (cell_h + gap) + (cell_h - im.size[1]) // 2
        canvas.paste(im, (x, y))
    return canvas


def save_gif(frames: List[np.ndarray], path: Path, duration_ms: int = 280) -> None:
    imgs = []
    for arr in frames:
        im = Image.fromarray(arr)
        im.thumbnail((640, 480), Image.Resampling.LANCZOS)
        imgs.append(im)
    imgs[0].save(
        path,
        save_all=True,
        append_images=imgs[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    print("  wrote %s frames=%d bytes=%d" % (path, len(imgs), path.stat().st_size))


def ensure_x11() -> str:
    """Bind DISPLAY/XAUTHORITY to a working X server (this machine uses :1, not :0)."""
    displays: List[str] = []
    cur = os.environ.get("DISPLAY") or ""
    if cur:
        displays.append(cur)
    displays.extend([":1", ":0"])
    auths: List[Optional[str]] = []
    for p in (
        os.environ.get("XAUTHORITY"),
        "/run/user/%d/gdm/Xauthority" % os.getuid(),
        os.path.expanduser("~/.Xauthority"),
    ):
        if p and os.path.isfile(p) and p not in auths:
            auths.append(p)
    if not auths:
        auths.append(None)

    seen = set()
    for d in displays:
        if d in seen:
            continue
        seen.add(d)
        for xa in auths:
            env = os.environ.copy()
            env["DISPLAY"] = d
            if xa:
                env["XAUTHORITY"] = xa
            try:
                subprocess.run(
                    ["xdpyinfo"],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=2,
                    check=True,
                )
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                continue
            os.environ["DISPLAY"] = d
            if xa:
                os.environ["XAUTHORITY"] = xa
            print("  X11 DISPLAY=%s XAUTHORITY=%s" % (d, xa or "(none)"))
            return d
    raise RuntimeError(
        "No working X11 display. Tried %s. On this host the GNOME session is usually :1."
        % displays
    )


def _pan_axes(eye, target, up=(0.0, 0.0, 1.0)):
    f = np.array(target, dtype=np.float64) - np.array(eye, dtype=np.float64)
    n = np.linalg.norm(f) + 1e-9
    f = f / n
    upv = np.array(up, dtype=np.float64)
    right = np.cross(f, upv)
    rn = np.linalg.norm(right) + 1e-9
    right = right / rn
    up_cam = np.cross(right, f)
    return right, up_cam


class _GodViewProxy:
    """Render in children that do not touch the X display.

    pygame and ModernGL cannot share a process here (the window stays black).
    God view and the fisheye rig also cannot share a process: two framebuffer
    sizes on one GL context make the readback shear into stripes.
    """

    def __init__(self, config: Path, stream=None, capture_scale=4):
        self.streaming = stream is not None
        self.stream_stats = (0, 0.0, 0.0)
        self.god = self._open("god", config)
        extra = ["--capture-scale", str(capture_scale)]
        if stream:
            extra += ["--stream-adb", "--serial", stream.serial, "--fps", str(stream.fps),
                      "--bitrate", stream.bitrate]
        self.cams = self._open("cams", config, extra)
        ready = self.cams.stdout.readline()
        parts = ready.decode(errors="replace").split()
        if not parts or parts[0] != "ready":
            raise RuntimeError("camera worker failed to start: %r" % ready)
        self.cam_names = tuple(parts[1:]) or ("front", "rear", "left", "right")
        got = self.god.stdout.readline()
        if got.strip() != b"ready":
            raise RuntimeError("god worker failed to start: %r" % got)

    @staticmethod
    def _open(mode: str, config: Path, extra=()) -> subprocess.Popen:
        env = os.environ.copy()
        env.pop("DISPLAY", None)
        env.pop("WAYLAND_DISPLAY", None)
        proc = subprocess.Popen(
            [sys.executable, str(ROOT / "scripts" / "view_render_worker.py"), *extra],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            env=env,
        )
        assert proc.stdin is not None
        proc.stdin.write(("%s\n%s\n" % (mode, config)).encode())
        proc.stdin.flush()
        return proc

    @staticmethod
    def _read_frame(proc: subprocess.Popen) -> np.ndarray:
        assert proc.stdout is not None
        header = proc.stdout.read(12)
        if len(header) != 12:
            raise RuntimeError("render worker closed")
        w, h, nbytes = struct.unpack("<III", header)
        buf = bytearray()
        while len(buf) < nbytes:
            chunk = proc.stdout.read(nbytes - len(buf))
            if not chunk:
                raise RuntimeError("render worker closed mid-frame")
            buf.extend(chunk)
        return np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3).copy()

    @staticmethod
    def _send(proc: subprocess.Popen, line: str) -> None:
        assert proc.stdin is not None
        proc.stdin.write(line.encode())
        proc.stdin.flush()

    def render(
        self,
        ego,
        eye,
        target,
        fov: float,
    ) -> np.ndarray:
        vals = [
            float(ego["x"]),
            float(ego["y"]),
            float(ego["z"]),
            float(ego["yaw"]),
            float(eye[0]),
            float(eye[1]),
            float(eye[2]),
            float(target[0]),
            float(target[1]),
            float(target[2]),
            0.0,
            float(fov),
        ]
        self._send(self.god, " ".join("%.6f" % v for v in vals) + "\n")
        return self._read_frame(self.god)

    def render_cameras(self, ego) -> dict:
        vals = [float(ego["x"]), float(ego["y"]), float(ego["z"]), float(ego["yaw"])]
        self._send(self.cams, " ".join("%.6f" % v for v in vals) + "\n")
        images = {name: self._read_frame(self.cams) for name in self.cam_names}
        if self.streaming:
            self.stream_stats = SEND_STATUS.unpack(_read_exact(self.cams.stdout, SEND_STATUS.size))[:3]
        return images

    def close(self) -> None:
        for proc in (self.god, self.cams):
            if proc.poll() is not None:
                continue
            try:
                if proc.stdin is not None:
                    proc.stdin.write(b"quit\n")
                    proc.stdin.flush()
                    proc.stdin.close()
                proc.wait(timeout=2)
            except Exception:
                proc.kill()


def try_interactive(world: World, size=(1280, 800), stream=None, capture_scale=4) -> int:
    """Draggable orbit window. LMB orbit, RMB pan, wheel zoom."""
    try:
        import pygame
    except ImportError:
        print("  live window skipped: pip install pygame")
        return 2

    try:
        ensure_x11()
    except RuntimeError as exc:
        print("  live window skipped:", exc)
        return 2

    os.environ.setdefault("SDL_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR", "0")
    pygame.init()
    pygame.display.set_caption(
        "parking-lite — arrows drive  |  LMB orbit  |  RMB pan  |  wheel zoom  |  C cameras"
    )
    god_w, god_h = size
    cam_strip_h = 220
    win_w, win_h = god_w, god_h + cam_strip_h
    try:
        screen = pygame.display.set_mode((win_w, win_h), pygame.SWSURFACE)
    except pygame.error as exc:
        print("  pygame.display.set_mode failed:", exc)
        pygame.quit()
        return 2

    try:
        font = pygame.font.SysFont("dejavusans,dejavu sans,sans", 16)
    except Exception:
        font = pygame.font.Font(None, 18)

    from vehicle import VehicleController

    rel = world.config.get("vehicle_config") or "configs/vehicle.yaml"
    vcfg_path = Path(rel)
    if not vcfg_path.is_file():
        vcfg_path = ROOT / rel
    ctrl = VehicleController(vcfg_path)
    ego0 = world.ego_pose
    ctrl.reset(ego0["x"], ego0["y"], ego0["yaw"], speed=0.0)
    ctrl.apply_to_world(world, z=0.0)

    look_offset = np.zeros(3, dtype=np.float64)
    follow = True
    az, el, radius = 215.0, 18.0, 8.0
    fov = 40.0
    dragging_orbit = False
    dragging_pan = False
    dirty = True
    show_cams = True
    clock = pygame.time.Clock()
    print("  window %dx%d on DISPLAY=%s" % (win_w, win_h, os.environ.get("DISPLAY")))
    print("  arrows = drive, LMB orbit, RMB pan, wheel zoom, F follow, R reset, C cameras")
    try:
        renderer = _GodViewProxy(Path(world.config_path), stream, capture_scale)
    except Exception as exc:
        print("  live window skipped:", exc)
        pygame.quit()
        return 2

    def look_at() -> np.ndarray:
        base = np.array(_ego_target_flu(world), dtype=np.float64)
        if follow:
            return base + look_offset
        return base + look_offset

    def render_frame() -> np.ndarray:
        tgt = look_at()
        eye = orbit_eye(tuple(tgt), az, el, radius)
        return renderer.render(world.ego_pose, eye, tuple(tgt), fov)

    def rgb_surface(arr: np.ndarray, size=None) -> "pygame.Surface":
        # 24-bit RGB blits onto this 32-bit window with the wrong pitch and
        # shows as stripes. Copy into a surface that matches the display.
        if size is not None and (arr.shape[1], arr.shape[0]) != size:
            src = Image.fromarray(np.ascontiguousarray(arr), "RGB")
            arr = np.asarray(src.resize(size, Image.Resampling.BILINEAR))
        h, w = arr.shape[:2]
        surf = pygame.Surface((w, h)).convert()
        pygame.surfarray.blit_array(surf, np.transpose(arr, (1, 0, 2)))
        return surf

    def render_cam_strip(images=None) -> pygame.Surface:
        if images is None:
            images = renderer.render_cameras(world.ego_pose)
        names = list(renderer.cam_names)
        cell_w = god_w // max(1, len(names))
        strip = pygame.Surface((god_w, cam_strip_h))
        strip.fill((12, 12, 14))
        label_font = font
        for i, name in enumerate(names):
            ch, cw = images[name].shape[:2]
            # Fit inside the cell, keep aspect, leave a label row.
            label_h = 18
            avail_h = cam_strip_h - label_h - 4
            scale = min((cell_w - 4) / cw, avail_h / ch)
            tw, th = max(1, int(cw * scale)), max(1, int(ch * scale))
            surf = rgb_surface(images[name], (tw, th))
            x0 = i * cell_w + (cell_w - tw) // 2
            strip.blit(surf, (x0, label_h))
            tag = label_font.render(name, True, (255, 230, 80), (0, 0, 0))
            strip.blit(tag, (i * cell_w + 6, 1))
        return strip

    img = render_frame()
    surf = rgb_surface(img, (god_w, god_h))
    cam_strip = render_cam_strip() if show_cams else None
    next_camera = time.monotonic() + (1/stream.fps if stream else 0)
    running = True
    while running:
        dt = clock.get_time() / 1000.0
        if dt <= 0.0:
            dt = 1.0 / 60.0
        # A slow fisheye frame must still move the car by the time that passed.
        dt = min(dt, 0.25)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_f:
                    follow = not follow
                    if follow:
                        look_offset[:] = 0.0
                    dirty = True
                elif event.key == pygame.K_r:
                    ctrl.reset(ego0["x"], ego0["y"], ego0["yaw"], speed=0.0)
                    ctrl.apply_to_world(world, z=0.0)
                    look_offset[:] = 0.0
                    follow = True
                    az, el, radius, fov = 215.0, 18.0, 8.0, 40.0
                    dirty = True
                elif event.key == pygame.K_w:
                    look_offset[0] += 0.4
                    dirty = True
                elif event.key == pygame.K_s:
                    look_offset[0] -= 0.4
                    dirty = True
                elif event.key == pygame.K_a:
                    look_offset[1] += 0.4
                    dirty = True
                elif event.key == pygame.K_d:
                    look_offset[1] -= 0.4
                    dirty = True
                elif event.key == pygame.K_e:
                    look_offset[2] += 0.3
                    dirty = True
                elif event.key == pygame.K_x:
                    look_offset[2] -= 0.3
                    dirty = True
                elif event.key == pygame.K_c:
                    show_cams = not show_cams
                    dirty = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    dragging_orbit = True
                elif event.button == 3:
                    dragging_pan = True
                elif event.button == 4:
                    radius = max(2.0, radius * 0.90)
                    dirty = True
                elif event.button == 5:
                    radius = min(40.0, radius * 1.10)
                    dirty = True
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging_orbit = False
                elif event.button == 3:
                    dragging_pan = False
            elif event.type == pygame.MOUSEMOTION:
                dx, dy = event.rel
                if dragging_orbit:
                    az = (az - dx * 0.35) % 360.0
                    el = float(np.clip(el + dy * 0.25, -5.0, 85.0))
                    dirty = True
                elif dragging_pan:
                    tgt = look_at()
                    eye = orbit_eye(tuple(tgt), az, el, radius)
                    right, up_cam = _pan_axes(eye, tuple(tgt))
                    scale = 0.0025 * radius
                    look_offset -= right * dx * scale
                    look_offset += up_cam * dy * scale
                    dirty = True
            elif event.type == pygame.MOUSEWHEEL:
                radius = float(np.clip(radius * (0.90 if event.y > 0 else 1.10), 2.0, 40.0))
                dirty = True

        keys = pygame.key.get_pressed()
        st_now = ctrl.get_state()
        vx = st_now.speed_mps
        steer = 0.0
        if keys[pygame.K_LEFT]:
            steer += 380.0
        if keys[pygame.K_RIGHT]:
            steer -= 380.0
        up = bool(keys[pygame.K_UP])
        down = bool(keys[pygame.K_DOWN])
        if keys[pygame.K_SPACE]:
            gear, accel = "N", 0.0
        elif up and not down:
            if vx < -0.08:
                gear, accel = "R", -3.2
            else:
                gear, accel = "D", 1.8
        elif down and not up:
            if vx > 0.08:
                gear, accel = "D", -3.2
            else:
                gear, accel = "R", 1.5
        elif abs(vx) > 0.03:
            gear = "D" if vx > 0.0 else "R"
            accel = -1.4
        else:
            gear, accel = "N", 0.0

        ctrl.set_controls(steer, accel, gear)
        st = ctrl.step(dt)
        ctrl.apply_to_world(world, z=0.0)
        moving = abs(st.speed_mps) > 1e-4 or abs(steer) > 1e-3 or up or down
        if moving:
            dirty = True

        # Board mode refreshes the overview at the camera cadence, leaving
        # the UI loop free to collect driving input between video frames.
        if dirty and (not stream or time.monotonic() >= next_camera):
            img = render_frame()
            surf = rgb_surface(img, (god_w, god_h))
            if show_cams and not stream:
                cam_strip = render_cam_strip()
            dirty = False

        if stream and time.monotonic() >= next_camera:
            images = renderer.render_cameras(world.ego_pose)
            if show_cams:
                cam_strip = render_cam_strip(images)
            next_camera = max(next_camera + 1/stream.fps, time.monotonic())

        screen.fill((8, 8, 10))
        screen.blit(surf, (0, 0))
        if show_cams and cam_strip is not None:
            screen.blit(cam_strip, (0, god_h))
        hud = [
            "Arrows drive  Space brake  LMB orbit  RMB pan  wheel zoom  F follow=%s  C cams=%s  R reset"
            % ("on" if follow else "off", "on" if show_cams else "off"),
            "gear %s  v=%+.2f m/s  steer=%+.0f  yaw=%.0f  xy=(%.2f, %.2f)  %.0f fps"
            % (
                st.gear,
                st.speed_mps,
                st.steer_wheel_deg,
                math.degrees(st.yaw),
                st.x,
                st.y,
                clock.get_fps(),
            ),
        ]
        if stream:
            sent, send_fps, send_mbps = renderer.stream_stats
            hud.append(f"Board SEND {send_fps:.2f} fps  {send_mbps:.2f} Mbps  frames={sent}"
                       f"  target={stream.fps} fps/{stream.bitrate}  capture={capture_scale}x")
        y = 8
        for line in hud:
            text = font.render(line, True, (255, 230, 80), (0, 0, 0))
            screen.blit(text, (8, y))
            y += text.get_height() + 2
        pygame.display.flip()
        clock.tick(60)

    renderer.close()
    pygame.quit()
    print("  window closed")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Orbit / inspect parking-lite scene")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "j01.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "output" / "j01_view")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--interactive", action="store_true", help="Open orbit window after stills")
    parser.add_argument("--stream-adb", action="store_true", help="send the window's four camera frames to the board")
    parser.add_argument("--serial", default="")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--bitrate", default="12M", help="total four-camera H264 bitrate")
    parser.add_argument("--capture-scale", type=int, choices=(1, 2, 4), default=4,
                        help="internal pinhole sampling scale (default: %(default)s); output stays 1280x960 per camera")
    parser.add_argument(
        "--window",
        action="store_true",
        help="Open a draggable orbit window only (skip PNG/GIF stills)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not 1 <= args.fps <= 30:
        parser.error("fps must be 1..30")
    if args.stream_adb and not (args.window or args.interactive):
        parser.error("--stream-adb requires --window or --interactive")
    stream = args if args.stream_adb else None

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    print("view_scene")
    print("  config:", args.config)
    print("  output:", out_dir)

    world = World.from_config(args.config)
    world.rig.render_cfg["pinhole_capture_scale"] = args.capture_scale
    world.set_ego_pose(0.5, 1.5, 0.0, math.radians(28.0))
    ego = world.rig.vehicle_cfg
    print(
        "  ego mesh: J01.obj"
        if (Path(world.rig.assets_root) / "vehicle" / "J01.obj").is_file()
        else "  ego mesh: procedural boxes"
    )
    print("  vehicle:", {k: ego.get(k) for k in ("length_m", "width_m", "origin")})

    if args.window:
        rc = try_interactive(world, stream=stream, capture_scale=args.capture_scale)
        return rc

    backend = world.rig.ensure_backend()
    print("  GL:", backend.gl_info)

    # Close 3/4 rear-right, similar to the libavm contact sheet
    target = _ego_target_flu(world)
    yaw_deg = math.degrees(world.ego_pose["yaw"])
    eye_iso = orbit_eye(
        target, azimuth_deg=yaw_deg + 215.0, elevation_deg=18.0, radius_m=7.5
    )
    iso = world.render_god_view(
        eye_flu=eye_iso, target_flu=target, up_flu=(0.0, 0.0, 1.0), fov_y_deg=38.0
    )
    _save(out_dir / "car_iso.png", iso)

    eye_side = orbit_eye(
        target, azimuth_deg=yaw_deg + 90.0, elevation_deg=8.0, radius_m=9.5
    )
    side = world.render_god_view(
        eye_flu=eye_side, target_flu=target, up_flu=(0.0, 0.0, 1.0), fov_y_deg=32.0
    )
    _save(out_dir / "car_side.png", side)

    eye_top = (target[0] - 0.2, target[1] - 0.2, 11.0)
    top = world.render_god_view(
        eye_flu=eye_top, target_flu=target, up_flu=(0.0, 1.0, 0.0), fov_y_deg=50.0
    )
    _save(out_dir / "car_top.png", top)

    print("  orbit stills…")
    frames = render_orbit(world, out_dir, n=max(4, int(args.frames)))
    collage = build_orbit_collage(frames)
    collage_path = out_dir / "orbit_collage.png"
    collage.save(collage_path)
    print("  wrote %s size=%s" % (collage_path, collage.size))
    save_gif(frames, out_dir / "orbit.gif")

    print("  surround + AVM (same pose as demo)…")
    god = world.render_god_view()
    _save(out_dir / "god_view.png", god)
    images = world.render_cameras()
    for name, arr in images.items():
        _save(out_dir / ("%s.png" % name), arr)
    bev = world.stitch_avm(images)
    _save(out_dir / "bev.png", bev)

    if args.interactive:
        try_interactive(world, stream=stream, capture_scale=args.capture_scale)

    world.release()
    print("view_scene OK —", out_dir)
    print("This sim has no editor viewport; stills/GIF are the browse path.")
    print("Live window: python scripts/view_scene.py --interactive  (needs DISPLAY)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
