#!/usr/bin/env python3
"""Phase 4: validate GT K/D/extrinsics via AVM metrics + extrinsic sensitivity."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from avm.stitcher import bev_grid_shape, clear_ipm_cache, draw_slot_overlay_bev
from avm.validate import (
    THRESHOLDS,
    clone_cameras_perturbed,
    draw_corner_overlay,
    make_compare_image,
    run_full_validation,
    stitch_with_cameras,
)
from scene.world import World


def _save(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)
    print("  wrote %s shape=%s bytes=%d" % (path, arr.shape, path.stat().st_size))


def main() -> int:
    cfg_path = ROOT / "configs" / "default.yaml"
    out_dir = ROOT / "output" / "phase4"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Phase 4 — intrinsics / extrinsics validation")
    print("  config:", cfg_path)
    print("  thresholds:")
    for k, v in THRESHOLDS.items():
        print("    %s = %s" % (k, v))

    world = World.from_config(cfg_path)
    bev_cfg = world.config.get("bev") or {}
    vehicle_cfg = world.config.get("vehicle") or {}
    h, w, res = bev_grid_shape(bev_cfg)
    print("  BEV: %dx%d @ %.4f m/px" % (h, w, res))

    # Same aisle pose family as Phase 3 so slot paint is in view
    pose = (0.0, 0.0, 0.0, 0.0)  # aisle origin; markers surround ego
    world.set_ego_pose(*pose)
    e = world.ego_pose
    print(
        "  ego_pose: x=%.3f y=%.3f z=%.3f yaw_deg=%.1f"
        % (e["x"], e["y"], e["z"], math.degrees(e["yaw"]))
    )

    print("  rendering 4 fisheyes (GT K/D/extrinsics)…")
    images = world.render_cameras()
    for name in world.rig.names:
        arr = images[name]
        assert arr.dtype == np.uint8 and arr.ndim == 3
        _save(out_dir / ("%s.png" % name), arr)

    cameras_gt = world.rig.cameras
    img_w, img_h = world.rig.image_size

    print("  stitching BEV with GT calib…")
    clear_ipm_cache()
    bev_gt = stitch_with_cameras(images, cameras_gt, bev_cfg, vehicle_cfg)
    assert bev_gt.shape == (h, w, 3)
    _save(out_dir / "bev_gt.png", bev_gt)

    # Perturb yaw/pitch ~1 deg (within 0.5-2 deg band) on stitch extrinsics only
    yaw_noise = 2.0
    pitch_noise = 2.0
    print(
        "  stitching BEV with perturbed extrinsics "
        "(yaw+/-%.1f deg pitch+/-%.1f deg, patterned)…" % (yaw_noise, pitch_noise)
    )
    cameras_pert = clone_cameras_perturbed(
        cameras_gt, yaw_noise_deg=yaw_noise, pitch_noise_deg=pitch_noise, seed=7
    )
    for name, cam in cameras_pert.items():
        gt = cameras_gt[name]
        print(
            "    %s rot_deg GT %s -> pert %s"
            % (
                name,
                np.array2string(gt.rotation_deg, precision=2),
                np.array2string(cam.rotation_deg, precision=2),
            )
        )
    bev_pert = stitch_with_cameras(images, cameras_pert, bev_cfg, vehicle_cfg)
    _save(out_dir / "bev_perturbed.png", bev_pert)

    print("  computing validation metrics…")
    report = run_full_validation(
        images=images,
        cameras_gt=cameras_gt,
        bev_gt=bev_gt,
        bev_pert=bev_pert,
        cameras_pert=cameras_pert,
        ego_xyz_yaw=world.ego_xyz_yaw(),
        bev_cfg=bev_cfg,
        vehicle_cfg=vehicle_cfg,
        image_size=(img_w, img_h),
    )

    # Overlays
    overlay_gt = draw_corner_overlay(
        bev_gt,
        {"records": report.metrics_gt.get("records") or []},
        world.ego_xyz_yaw(),
        bev_cfg,
        title="GT corners (green=CAD, red=paint)",
    )
    overlay_gt = draw_slot_overlay_bev(overlay_gt, world.ego_xyz_yaw(), bev_cfg)
    _save(out_dir / "bev_gt_overlay.png", overlay_gt)

    overlay_pert = draw_corner_overlay(
        bev_pert,
        {"records": report.metrics_pert.get("records") or []},
        world.ego_xyz_yaw(),
        bev_cfg,
        title="Perturbed corners (green=CAD, red=paint)",
    )
    overlay_pert = draw_slot_overlay_bev(
        overlay_pert, world.ego_xyz_yaw(), bev_cfg, color=(255, 120, 80)
    )
    _save(out_dir / "bev_perturbed_overlay.png", overlay_pert)

    compare = make_compare_image(bev_gt, bev_pert)
    _save(out_dir / "compare.png", compare)

    # Reports
    text = report.to_text()
    report_txt = out_dir / "report.txt"
    report_md = out_dir / "report.md"
    report_txt.write_text(text, encoding="utf-8")
    md_parts = []
    md_parts.append("# Phase 4 Validation Report\n")
    md_parts.append("\n```\n")
    md_parts.append(text)
    md_parts.append("```\n")
    md_parts.append("\n## Artifacts\n\n")
    md_parts.append("- `bev_gt.png` — BEV stitch with GT calib\n")
    md_parts.append("- `bev_perturbed.png` — same images, yaw/pitch-perturbed extrinsics\n")
    md_parts.append("- `compare.png` — GT | perturbed | abs-diff heat\n")
    md_parts.append("- `bev_gt_overlay.png` / `bev_perturbed_overlay.png` — CAD + corner markers\n")
    md_parts.append("- `report.txt` — same numbers as above\n")
    report_md.write_text("".join(md_parts), encoding="utf-8")
    print("  wrote", report_txt)
    print("  wrote", report_md)

    print("")
    print(text)

    print("Checks:")
    for name, status in report.checks.items():
        print("  [%s] %s" % (status, name))
    print("OVERALL: %s" % report.overall)
    print("  output:", out_dir)
    print("  GL:", world.rig.ensure_backend().gl_info)

    world.release()

    if report.overall == "FAIL":
        print("Phase 4 validation finished with FAIL (exit 0; see report)")
    elif report.overall == "WARN":
        print("Phase 4 validation WARN (acceptable residuals; exit 0)")
    else:
        print("Phase 4 validation PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
