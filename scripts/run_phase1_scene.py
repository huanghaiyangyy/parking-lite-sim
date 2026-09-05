#!/usr/bin/env python3
"""Phase 1: scene construction — posed ego + god view + 4 fisheye smoke."""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from scene.world import World
from sensors.cameras import gl_to_flu
from scene.lot import slot_center_gl


def _save(path: Path, arr) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)
    print("  wrote %s shape=%s" % (path, arr.shape))


def _print_pose(label: str, world: World) -> None:
    e = world.ego_pose
    print(
        "  %s: x=%.3f y=%.3f z=%.3f yaw_rad=%.4f yaw_deg=%.1f"
        % (label, e["x"], e["y"], e["z"], e["yaw"], math.degrees(e["yaw"]))
    )


def main() -> int:
    cfg_path = ROOT / "configs" / "default.yaml"
    out_dir = ROOT / "output" / "phase1"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Phase 1 scene")
    print("  config:", cfg_path)

    world = World.from_config(cfg_path)

    # Pose A: angled in the center aisle (visible vs slot paint + pillars)
    pose_a = (0.5, 1.5, 0.0, math.radians(28.0))
    # Pose B: near an open bay east of the south row (proves mesh follows set_ego_pose)
    # South slot index 4 center in GL → FLU for reference (occupied by static car);
    # we offset into the aisle / open asphalt instead.
    sx, sy, sz = gl_to_flu(slot_center_gl("south", 4))
    pose_b = (sx + 3.5, sy - 4.0, 0.0, math.radians(-55.0))

    # --- Pose A: god view + surround ---
    world.set_ego_pose(*pose_a)
    _print_pose("ego_pose_a", world)
    god_a = world.render_god_view()
    _save(out_dir / "god_view_pose_a.png", god_a)
    _save(out_dir / "god_view.png", god_a)  # primary overview alias

    images = world.render_cameras()
    for name in world.rig.names:
        arr = images[name]
        print("  image[%s]: shape=%s dtype=%s" % (name, arr.shape, arr.dtype))
        _save(out_dir / ("%s.png" % name), arr)

    # --- Pose B: second god view to prove ego motion relative to slots ---
    world.set_ego_pose(*pose_b)
    _print_pose("ego_pose_b", world)
    god_b = world.render_god_view()
    _save(out_dir / "god_view_pose_b.png", god_b)

    print(
        "  image_size: %dx%d (WxH)"
        % (world.rig.image_size[0], world.rig.image_size[1])
    )
    backend = world.rig.ensure_backend()
    print("  GL:", backend.gl_info)

    world.release()
    print("  output:", out_dir)
    print("Phase 1 scene OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
