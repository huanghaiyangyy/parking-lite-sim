#!/usr/bin/env python3
"""Compare GPU KB remap / atlas against OpenCV, then time both render paths."""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scene.world import World
from sensors.fisheye_opencv import remap_pinhole_to_fisheye

ORDER = ("front", "left", "right", "rear")


def cpu_grid(world):
    backend = world.rig.ensure_backend()
    images = []
    for name in ORDER:
        cam = world.rig.cameras[name]
        look = world.rig.camera_look_gl(name, world.ego_xyz_yaw())
        pinhole = backend._render_wide_pinhole(look["eye"], look["target"], look["up"])
        images.append(remap_pinhole_to_fisheye(pinhole, cam.K, cam.D, backend.pinhole_fov_y_deg,
                                             out_size=world.rig.image_size))
    return np.concatenate((np.concatenate(images[:2], axis=1),
                           np.concatenate(images[2:], axis=1)), axis=0)


def difference(actual, expected):
    delta = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    result = dict(max=int(delta.max()), mean=float(delta.mean()), changed=int(np.count_nonzero(delta)))
    assert result["max"] == 0, result
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/j01.yaml")
    parser.add_argument("--scale", type=float, choices=(1.0, 0.5), default=1.0)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--capture-scale", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.frames < 2:
        parser.error("frames must be at least two")
    cv2.setNumThreads(2)
    world = World.from_config(args.config)
    world.rig.render_cfg["pinhole_capture_scale"] = args.capture_scale
    if args.scale != 1:
        from scripts.view_render_worker import _scaled_rig
        world.rig = _scaled_rig(world, args.scale)
    try:
        backend = world.rig.ensure_backend()
        result = dict(renderer=backend.gl_info, image_size=world.rig.image_size,
                      scale=args.scale, capture_scale=args.capture_scale, samples={}, pixel_checks=[])
        h, w = backend.height, backend.width
        backend._ensure_capture()
        rng = np.random.default_rng(42)
        pinhole = rng.integers(0, 256, (backend.capture_height, backend.capture_width, 3), dtype=np.uint8)
        cam = world.rig.cameras["front"]
        # Check a second K/D too: identical production intrinsics alone would
        # fail to detect an incorrectly shared texture cache.
        alternate_k, alternate_d = cam.K.copy(), cam.D.copy()
        alternate_k[0, 2] += 11.25
        alternate_d[0] += .01
        for k, d in [(cam.K, cam.D), (alternate_k, alternate_d)]:
            backend._capture_tex.write(pinhole[::-1].copy().tobytes())
            backend._remap_gpu(k, d, backend._fish_fbo, (0, 0, w, h))
            actual = backend._read_fbo_rgb(backend._fish_fbo)
            expected = remap_pinhole_to_fisheye(pinhole, k, d, backend.pinhole_fov_y_deg,
                                              out_size=(w, h))
            result["pixel_checks"].append(difference(actual, expected))
        for pose in [(0., 0., 0., 0.), (.5, 1.5, 0., 0.),
                     (-2., -.7, 0., .4), (3., 2., .1, -1.2)]:
            world.set_ego_pose(*pose)
            result["pixel_checks"].append(difference(world.render_camera_grid(), cpu_grid(world)))
            cameras = world.render_cameras()
            expected = np.concatenate((np.concatenate([cameras[n] for n in ORDER[:2]], axis=1),
                                       np.concatenate([cameras[n] for n in ORDER[2:]], axis=1)), axis=0)
            result["pixel_checks"].append(difference(world.render_camera_grid(), expected))
        # Alternate paths at the same poses to avoid comparing different load
        # or power states. Both include readback and produce the same RGB grid.
        for _ in range(3):
            cpu_grid(world)
            world.render_camera_grid()
        times = {"cpu_remap": [], "gpu_grid": []}
        for frame in range(args.frames):
            world.set_ego_pose(.5 + frame*.04, 1.5, 0., .01*(frame%10))
            for name in (tuple(times) if frame%2 else tuple(reversed(times))):
                start = time.perf_counter()
                cpu_grid(world) if name == "cpu_remap" else world.render_camera_grid()
                times[name].append((time.perf_counter()-start)*1000)
        for name, values in times.items():
            result["samples"][name] = dict(mean_ms=float(np.mean(values)),
                p95_ms=float(np.percentile(values, 95)), fps=1000/float(np.mean(values)), values_ms=values)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=2))
        print("PASS: exact random-pixel remap, two calibrations, four poses, camera order")
        print(json.dumps({k: {n: v for n, v in data.items() if n != "values_ms"}
                          for k, data in result["samples"].items()}, indent=2))
    finally:
        world.release()


if __name__ == "__main__":
    main()
