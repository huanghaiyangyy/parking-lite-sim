#!/usr/bin/env python3
"""Render matched flat/realistic views and check geometry, stability and timing."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scene.lot import build_static_lot, load_compiled_lot, surface_material_ids
from scene.mesh_prims import traffic_cone_mesh
from scene.world import World


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "output/materials")
    parser.add_argument("--capture-scale", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--lighting-style", choices=("simple", "shadowed"), default="simple")
    args = parser.parse_args()
    if args.frames < 2:
        parser.error("--frames must be >= 2")
    args.out.mkdir(parents=True, exist_ok=True)

    cone = np.asarray(traffic_cone_mesh(0, 0), dtype=np.float32).reshape(-1, 3, 9)
    assert np.allclose(np.ptp(cone[:, :, :3], axis=(0, 1)), (0.4, 0.7, 0.4))
    winding = np.cross(cone[:, 1, :3]-cone[:, 0, :3], cone[:, 2, :3]-cone[:, 0, :3])
    assert np.all((winding*cone[:, :, 3:6].mean(axis=1)).sum(axis=1) > 0)
    source = load_compiled_lot(ROOT / "assets/lot/lot_parking_lot_3.bin")
    lot = build_static_lot(ROOT / "assets")
    assert np.array_equal(lot[:len(source)], source), "Original geometry changed"
    assert len(lot) == len(source) + 4*cone.shape[0]*3, "Expected four slot cones"
    ids, counts = np.unique(surface_material_ids(lot), return_counts=True)
    assert set(range(8)).issubset(set(ids.tolist())), "Surface material missing"

    # GL coordinates; close views of actual walls, divider curb and occupied bay.
    views = {
        "wall": ((-0.8, 1.45, -8.0), (-6.4, 1.05, -5.2), 68),
        "curb": ((1.1, 0.70, 1.6), (0.1, 0.12, 3.8), 65),
        "cones": ((0.3, 1.30, -1.0), (2.8, 0.33, 1.05), 62),
        "overview": ((18.0, 22.0, -25.0), (0.0, 0.0, -2.0), 65),
    }
    rendered = {}
    results = {"capture_scale": args.capture_scale, "lighting_style": args.lighting_style,
               "surface_vertex_counts": dict(zip(ids.astype(int).tolist(), counts.tolist())),
               "geometry_unchanged_except_four_cones": True}
    for style in ("flat", "realistic"):
        world = World.from_config(ROOT / "configs/j01.yaml")
        world.rig.render_cfg.update(material_style=style, lighting_style=args.lighting_style,
                                    pinhole_capture_scale=args.capture_scale)
        world.set_ego_pose(-10, 0, 0, 0)
        try:
            backend = world.rig.ensure_backend()
            results["gl"] = backend.gl_info
            for name, (eye, target, fov) in views.items():
                arr = backend.render_pinhole(eye, target, (0, 1, 0), fov_y_deg=fov)
                rendered[style, name] = arr
                Image.fromarray(arr).save(args.out / (name+"_"+style+".png"))
            first = world.render_camera_grid()
            repeat = world.render_camera_grid()
            assert np.array_equal(first, repeat), "Static material flickers"
            Image.fromarray(first).save(args.out / ("fisheye_"+style+".png"))
            images = world.render_cameras()
            bev = world.stitch_avm(images)
            Image.fromarray(bev).save(args.out / ("avm_"+style+".png"))
            rendered[style, "avm"] = bev
            samples = []
            for i in range(args.frames):
                # Invalidate moving-ego shadows, as the streaming path does.
                world.set_ego_pose(-10 + i*0.015, 0, 0, 0)
                start = time.perf_counter()
                world.render_camera_grid()
                samples.append((time.perf_counter()-start)*1000)
            results[style] = {"render_grid_ms_median": float(np.median(samples)),
                              "render_grid_ms_max": max(samples),
                              "static_repeat_identical": True}
        finally:
            world.release()

    for name in (*views, "avm"):
        a, b = rendered["flat", name], rendered["realistic", name]
        assert np.mean(np.abs(a.astype(float)-b.astype(float))) > 1, name
        w = 720
        h = round(a.shape[0]*w/a.shape[1])
        comparison = Image.new("RGB", (w*2, h+38), "#121923")
        draw = ImageDraw.Draw(comparison)
        draw.text((16, 12), "FLAT / original lighting", fill="white")
        draw.text((w+16, 12), "REALISTIC / materials + " + args.lighting_style + " lighting", fill="white")
        comparison.paste(Image.fromarray(a).resize((w, h), Image.Resampling.LANCZOS), (0, 38))
        comparison.paste(Image.fromarray(b).resize((w, h), Image.Resampling.LANCZOS), (w, 38))
        comparison.save(args.out / (name+"_compare.jpg"), quality=93)
    (args.out / "checks.json").write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps(results, indent=2))
    print("Saved material comparisons:", args.out)


if __name__ == "__main__":
    main()
