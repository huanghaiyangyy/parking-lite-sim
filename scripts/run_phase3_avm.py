#!/usr/bin/env python3
"""Phase 3: AVM top-down BEV stitch (IPM + feather blend)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from avm.stitcher import bev_grid_shape, draw_slot_overlay_bev
from scene.world import World


def _save(path: Path, arr) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)
    print("  wrote %s shape=%s bytes=%d" % (path, arr.shape, path.stat().st_size))


def main() -> int:
    cfg_path = ROOT / "configs" / "default.yaml"
    out_dir = ROOT / "output" / "phase3"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Phase 3 AVM / BEV stitch")
    print("  config:", cfg_path)

    world = World.from_config(cfg_path)
    bev_cfg = world.config.get("bev") or {}
    h, w, res = bev_grid_shape(bev_cfg)
    print(
        "  BEV: shape HxW = %dx%d, meters_per_px = %.4f"
        % (h, w, res)
    )
    print(
        "  extents: fwd=%.1f back=%.1f left=%.1f right=%.1f m"
        % (
            float(bev_cfg.get("forward_m", 8)),
            float(bev_cfg.get("back_m", 8)),
            float(bev_cfg.get("left_m", 6)),
            float(bev_cfg.get("right_m", 6)),
        )
    )

    # Ego in aisle (same family as Phase 1/2) so slot paint is visible around car
    pose = (0.5, 1.5, 0.0, math.radians(28.0))
    world.set_ego_pose(*pose)
    e = world.ego_pose
    print(
        "  ego_pose: x=%.3f y=%.3f z=%.3f yaw_deg=%.1f"
        % (e["x"], e["y"], e["z"], math.degrees(e["yaw"]))
    )

    print("  rendering 4 fisheyes (OpenCV K/D)…")
    images = world.render_cameras()
    for name in world.rig.names:
        arr = images[name]
        assert arr.dtype == np.uint8 and arr.ndim == 3
        _save(out_dir / ("%s.png" % name), arr)

    print("  stitching BEV…")
    bev = world.stitch_avm(images)
    assert bev.shape == (h, w, 3), "BEV shape %s != expected (%d,%d,3)" % (
        bev.shape,
        h,
        w,
    )
    _save(out_dir / "bev.png", bev)
    print("  BEV shape=%s meters_per_px=%.4f" % (bev.shape, res))

    # Optional slot-line overlay (world paint → ego BEV)
    overlay = draw_slot_overlay_bev(bev, world.ego_xyz_yaw(), bev_cfg)
    _save(out_dir / "bev_overlay.png", overlay)

    # Coverage / seam diagnostics for Phase 4 hooks
    gray = bev.mean(axis=2)
    nonzero = float(np.mean(gray > 5.0))
    print("  BEV nonzero fraction (excl. near-black): %.3f" % nonzero)
    print("  GL:", world.rig.ensure_backend().gl_info)
    print("  output:", out_dir)

    world.release()
    print("Phase 3 AVM stitch OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
