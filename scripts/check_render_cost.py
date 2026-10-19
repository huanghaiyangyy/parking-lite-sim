#!/usr/bin/env python3
"""Check exact sampling masks and compare lighting/capture costs on one GPU."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scene.world import World


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--capture-scale", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--out", type=Path, default=ROOT / "output/render-cost.json")
    args = parser.parse_args()
    if args.frames < 4:
        parser.error("--frames must be >= 4")
    world = World.from_config(ROOT / "configs/j01.yaml")
    world.rig.render_cfg["pinhole_capture_scale"] = args.capture_scale
    result = {"scope": "four 1280x960 views, RGBA readback, no encoding",
              "capture_scale": args.capture_scale, "pixel_checks": []}
    modes = {"flat": ("flat", "simple", False),
             "before": ("realistic", "shadowed", False),
             "masked_shadowed": ("realistic", "shadowed", True),
             "simple_unmasked": ("realistic", "simple", False),
             "optimized": ("realistic", "simple", True)}
    samples = {name: [] for name in modes}
    try:
        backend = world.rig.ensure_backend()
        result["gl"] = backend.gl_info
        for lighting in ("simple", "shadowed"):
            backend.lighting_style = lighting
            for pose in ((-10, 0, 0, 0), (0.5, 1.5, 0, 0),
                         (-2, -0.7, 0, 0.4), (3, 2, 0.1, -1.2)):
                world.set_ego_pose(*pose)
                backend.pinhole_sample_mask = False
                before = world.render_camera_grid_rgba().copy()
                backend.pinhole_sample_mask = True
                after = world.render_camera_grid_rgba()
                delta = np.abs(before.astype(np.int16) - after.astype(np.int16))
                check = {"lighting": lighting, "pose": pose, "max": int(delta.max()),
                         "mean": float(delta.mean()), "changed": int(np.count_nonzero(delta))}
                result["pixel_checks"].append(check)
                assert check["max"] == 0, check
        for frame in range(args.frames + len(modes)):
            world.set_ego_pose(-10 + frame*0.02, 0, 0, (frame%3)*0.01)
            names = list(modes)
            offset = frame % len(names)
            names = names[offset:] + names[:offset]
            for name in names:
                (backend.material_style, backend.lighting_style,
                 backend.pinhole_sample_mask) = modes[name]
                # Each shadowed mode must pay for a moving-ego shadow update,
                # rather than reusing the preceding mode's cached shadow map.
                backend._shadow_dirty = True
                start = time.perf_counter()
                world.render_camera_grid_rgba()
                elapsed = (time.perf_counter()-start)*1000
                if frame >= len(modes):
                    samples[name].append(elapsed)
        result["samples"] = {name: {"median_ms": float(np.median(values)),
                                   "p95_ms": float(np.percentile(values, 95)),
                                   "values_ms": values} for name, values in samples.items()}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result, indent=2))
    finally:
        world.release()


if __name__ == "__main__":
    main()
