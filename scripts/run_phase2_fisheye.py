#!/usr/bin/env python3
"""Phase 2: true OpenCV fisheye (Kannala–Brandt K/D from YAML) surround render."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from scene.world import World
from sensors.fisheye_opencv import (
    clear_lut_cache,
    remap_pinhole_to_fisheye,
    verify_kd_roundtrip,
)


def _save(path: Path, arr) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)
    print("  wrote %s shape=%s bytes=%d" % (path, arr.shape, path.stat().st_size))


def _print_kd(world: World) -> None:
    print("  K/D from YAML (sim GT, OpenCV cv2.fisheye / Kannala–Brandt):")
    print("  config_path:", world.config_path)
    for name in world.rig.names:
        cam = world.rig.cameras[name]
        print("  [%s]" % name)
        print("    K =\n%s" % np.array2string(cam.K, precision=1, suppress_small=True))
        print("    D = %s" % np.array2string(cam.D, precision=4, suppress_small=True))
        err = verify_kd_roundtrip(cam.K, cam.D)
        print("    cv2.fisheye distort↔undistort max err ≈ %.4f px" % err)


def _d_visibility_demo(world: World, out_dir: Path) -> None:
    """Show that changing D visibly changes distortion (front cam, same pinhole)."""

    cam = world.rig.cameras["front"]
    look = world.rig.camera_look_gl("front", world.ego_xyz_yaw())
    backend = world.rig.ensure_backend()
    # Raw wide pinhole (bypass fisheye) via private helper
    pinhole = backend._render_wide_pinhole(look["eye"], look["target"], look["up"])
    fov = float(world.config["render"]["pinhole_fov_y_deg"])

    d_zero = np.zeros(4, dtype=np.float64)
    d_yaml = cam.D.copy()
    d_strong = np.array([-0.12, 0.04, -0.01, 0.002], dtype=np.float64)

    clear_lut_cache()
    img_zero = remap_pinhole_to_fisheye(pinhole, cam.K, d_zero, fov)
    img_yaml = remap_pinhole_to_fisheye(pinhole, cam.K, d_yaml, fov)
    img_strong = remap_pinhole_to_fisheye(pinhole, cam.K, d_strong, fov)

    # Side-by-side collage
    h, w = img_yaml.shape[:2]
    strip = Image.new("RGB", (w * 3, h))
    strip.paste(Image.fromarray(img_zero), (0, 0))
    strip.paste(Image.fromarray(img_yaml), (w, 0))
    strip.paste(Image.fromarray(img_strong), (2 * w, 0))
    path = out_dir / "front_D_compare_zero_yaml_strong.png"
    strip.save(path)
    print("  D-visibility demo (zero | YAML | strong):", path)
    # Mean abs diff should be non-trivial
    mad_yaml = float(np.mean(np.abs(img_yaml.astype(np.float32) - img_zero.astype(np.float32))))
    mad_strong = float(
        np.mean(np.abs(img_strong.astype(np.float32) - img_zero.astype(np.float32)))
    )
    print("  mean|Δ| vs D=0: yaml=%.2f strong=%.2f (expect strong > yaml > 0)" % (mad_yaml, mad_strong))


def main() -> int:
    cfg_path = ROOT / "configs" / "default.yaml"
    out_dir = ROOT / "output" / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Phase 2 OpenCV fisheye")
    print("  config:", cfg_path)

    world = World.from_config(cfg_path)
    mode = (world.config.get("render") or {}).get("fisheye_mode", "?")
    print("  render.fisheye_mode:", mode)
    print(
        "  image_size: %dx%d (WxH)"
        % (world.rig.image_size[0], world.rig.image_size[1])
    )

    # Ego in the lot aisle, slight yaw so paint/pillars read clearly
    pose = (0.5, 1.5, 0.0, math.radians(28.0))
    world.set_ego_pose(*pose)
    e = world.ego_pose
    print(
        "  ego_pose: x=%.3f y=%.3f z=%.3f yaw_deg=%.1f"
        % (e["x"], e["y"], e["z"], math.degrees(e["yaw"]))
    )

    _print_kd(world)

    images = world.render_cameras()
    for name in world.rig.names:
        arr = images[name]
        assert arr.shape == (world.rig.image_size[1], world.rig.image_size[0], 3)
        _save(out_dir / ("%s.png" % name), arr)

    w, h = world.rig.image_size
    collage = Image.new("RGB", (w * 2, h * 2))
    collage.paste(Image.fromarray(images["front"]), (0, 0))
    collage.paste(Image.fromarray(images["right"]), (w, 0))
    collage.paste(Image.fromarray(images["left"]), (0, h))
    collage.paste(Image.fromarray(images["rear"]), (w, h))
    collage_path = out_dir / "collage.png"
    collage.save(collage_path)
    print("  wrote %s bytes=%d" % (collage_path, collage_path.stat().st_size))

    _d_visibility_demo(world, out_dir)

    backend = world.rig.ensure_backend()
    print("  GL:", backend.gl_info)
    print("  output:", out_dir)
    world.release()
    print("Phase 2 OpenCV fisheye OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
