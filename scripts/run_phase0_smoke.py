#!/usr/bin/env python3
"""Phase 0 smoke: load config, set ego pose, render 4 cameras via World API."""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene.world import World


def main() -> int:
    cfg_path = ROOT / "configs" / "default.yaml"
    print("Phase 0 smoke")
    print("  config:", cfg_path)

    world = World.from_config(cfg_path)
    world.set_ego_pose(0.0, 0.0, 0.0, 0.0)

    ego = world.ego_pose
    print(
        "  ego_pose: x=%.3f y=%.3f z=%.3f yaw_rad=%.4f yaw_deg=%.1f"
        % (ego["x"], ego["y"], ego["z"], ego["yaw"], math.degrees(ego["yaw"]))
    )
    print("  cameras:", ", ".join(world.rig.names))
    print(
        "  image_size: %dx%d (WxH)"
        % (world.rig.image_size[0], world.rig.image_size[1])
    )
    bev = world.config.get("bev") or {}
    print(
        "  bev: fwd/back=%.1f/%.1f m L/R=%.1f/%.1f m res=%.3f m/px"
        % (
            float(bev.get("forward_m", 0)),
            float(bev.get("back_m", 0)),
            float(bev.get("left_m", 0)),
            float(bev.get("right_m", 0)),
            float(bev.get("resolution_m_per_px", 0)),
        )
    )

    try:
        images = world.render_cameras()
    except Exception as exc:  # noqa: BLE001 — smoke reports and exits non-zero
        print("  render FAILED:", type(exc).__name__, exc)
        print("  render deferred / unavailable")
        world.release()
        return 1

    out_dir = ROOT / "output" / "phase0"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except ImportError:
        Image = None  # type: ignore

    for name in world.rig.names:
        arr = images[name]
        print("  image[%s]: shape=%s dtype=%s" % (name, arr.shape, arr.dtype))
        if Image is not None:
            Image.fromarray(arr).save(out_dir / ("%s.png" % name))

    # AVM expected NotImplemented
    try:
        world.stitch_avm(images)
        print("  stitch_avm: unexpected success")
    except NotImplementedError as exc:
        print("  stitch_avm: NotImplementedError (expected) —", str(exc)[:80])

    world.release()
    print("  wrote:", out_dir)
    print("Phase 0 smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
