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

import os
import struct
import sys
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


def main() -> int:
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
        preview = _scaled_rig(world, _PREVIEW_SCALE)
        world.release()
        names = tuple(preview.names)
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
        preview.release()
        return 0

    sys.stderr.write("view_render_worker: expected god or cams, got %r\n" % mode)
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(0)
