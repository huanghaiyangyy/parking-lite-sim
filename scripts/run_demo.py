#!/usr/bin/env python3
"""Phase 5: one-click demo — fisheyes + god view + AVM BEV collage + short ego motion."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont

from avm.stitcher import bev_grid_shape, draw_slot_overlay_bev
from avm.validate import coverage_metrics, validate_bev_smoke
from scene.world import World

# Demo pose in aisle (same family as Phase 1/3) — slot paint visible around ego
DEMO_POSE = (0.5, 1.5, 0.0, math.radians(28.0))

# Short trajectory through the aisle (FLU meters, yaw radians)
MOTION_POSES: List[Tuple[float, float, float, float]] = [
    (0.0, 0.0, 0.0, math.radians(0.0)),
    (0.8, 0.4, 0.0, math.radians(12.0)),
    (1.6, 0.9, 0.0, math.radians(22.0)),
    (2.2, 1.3, 0.0, math.radians(28.0)),
    (2.8, 1.7, 0.0, math.radians(34.0)),
]


def _save(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)
    print("  wrote %s shape=%s bytes=%d" % (path, arr.shape, path.stat().st_size))


def _fit(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """Scale preserving aspect so image fits inside max_w x max_h."""
    w, h = img.size
    scale = min(max_w / float(w), max_h / float(h), 1.0)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    if (nw, nh) == (w, h):
        return img
    return img.resize((nw, nh), Image.Resampling.LANCZOS)


def _label(img: Image.Image, text: str, pad: int = 6) -> Image.Image:
    """Draw a small label bar at top-left."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([0, 0, tw + 2 * pad, th + 2 * pad], fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 80), font=font)
    return out


def _panel(arr: np.ndarray, title: str, max_w: int, max_h: int) -> Image.Image:
    im = Image.fromarray(arr)
    im = _fit(im, max_w, max_h)
    return _label(im, title)


def build_collage(
    god: np.ndarray,
    cams: Dict[str, np.ndarray],
    bev: np.ndarray,
    cell_w: int = 420,
    cell_h: int = 320,
) -> Image.Image:
    """
    Packaged collage layout (dark background):

        [ god_view ] [ BEV ]
        [ front    ] [ rear ]
        [ left     ] [ right ]
    """
    gap = 8
    margin = 12
    cols, rows = 2, 3
    canvas_w = margin * 2 + cols * cell_w + (cols - 1) * gap
    canvas_h = margin * 2 + rows * cell_h + (rows - 1) * gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), (24, 24, 28))

    panels = [
        (0, 0, _panel(god, "god_view", cell_w, cell_h)),
        (1, 0, _panel(bev, "AVM BEV", cell_w, cell_h)),
        (0, 1, _panel(cams["front"], "front", cell_w, cell_h)),
        (1, 1, _panel(cams["rear"], "rear", cell_w, cell_h)),
        (0, 2, _panel(cams["left"], "left", cell_w, cell_h)),
        (1, 2, _panel(cams["right"], "right", cell_w, cell_h)),
    ]
    for col, row, panel in panels:
        # Center panel in cell
        x0 = margin + col * (cell_w + gap)
        y0 = margin + row * (cell_h + gap)
        ox = x0 + (cell_w - panel.size[0]) // 2
        oy = y0 + (cell_h - panel.size[1]) // 2
        canvas.paste(panel, (ox, oy))
    return canvas


def _quick_validation(world: World, bev: np.ndarray) -> Dict[str, float]:
    bev_cfg = world.config.get("bev") or {}
    vehicle_cfg = world.config.get("vehicle") or {}
    smoke = validate_bev_smoke(bev, bev_cfg)
    cov = coverage_metrics(bev, bev_cfg, vehicle_cfg)
    return {**smoke, **cov}


def _save_gif(frames: Sequence[np.ndarray], path: Path, duration_ms: int = 350) -> None:
    """Save BEV sequence as GIF via Pillow (no imageio required)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs = [Image.fromarray(f) for f in frames]
    # Shrink slightly for smaller gif if huge
    max_side = 480
    out_imgs = []
    for im in imgs:
        w, h = im.size
        scale = min(1.0, max_side / float(max(w, h)))
        if scale < 1.0:
            im = im.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                Image.Resampling.LANCZOS,
            )
        out_imgs.append(im)
    out_imgs[0].save(
        path,
        save_all=True,
        append_images=out_imgs[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    print("  wrote %s frames=%d bytes=%d" % (path, len(out_imgs), path.stat().st_size))


def run_motion(world: World, out_dir: Path) -> List[Path]:
    """Render 3–5 BEV frames along a short aisle path + GIF."""
    motion_dir = out_dir / "motion"
    motion_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    frames: List[np.ndarray] = []
    bev_cfg = world.config.get("bev") or {}

    print("  short ego-motion (%d poses)…" % len(MOTION_POSES))
    for i, pose in enumerate(MOTION_POSES):
        world.set_ego_pose(*pose)
        e = world.ego_pose
        print(
            "    frame %d: x=%.2f y=%.2f yaw_deg=%.1f"
            % (i, e["x"], e["y"], math.degrees(e["yaw"]))
        )
        images = world.render_cameras()
        bev = world.stitch_avm(images)
        overlay = draw_slot_overlay_bev(bev, world.ego_xyz_yaw(), bev_cfg)
        name = "bev_frame_%02d.png" % i
        p = motion_dir / name
        _save(p, overlay)
        paths.append(p)
        frames.append(overlay)

    gif_path = motion_dir / "bev_motion.gif"
    _save_gif(frames, gif_path)
    paths.append(gif_path)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 5 one-click parking-lite demo")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "default.yaml",
        help="YAML config path",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "output" / "demo",
        help="Output directory",
    )
    parser.add_argument(
        "--no-motion",
        action="store_true",
        help="Skip short ego-motion BEV sequence / GIF",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip quick BEV validation summary",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg_path = args.config
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Parking-lite demo (Phase 5)")
    print("=" * 60)
    print("  config:", cfg_path)
    print("  output:", out_dir)

    world = World.from_config(cfg_path)
    phase = (world.config.get("project") or {}).get("phase", "?")
    print("  project.phase (config):", phase)

    bev_cfg = world.config.get("bev") or {}
    h, w, res = bev_grid_shape(bev_cfg)
    print("  BEV: %dx%d @ %.4f m/px" % (h, w, res))

    # --- Primary demo pose ---
    world.set_ego_pose(*DEMO_POSE)
    e = world.ego_pose
    print(
        "  ego_pose: x=%.3f y=%.3f z=%.3f yaw_deg=%.1f"
        % (e["x"], e["y"], e["z"], math.degrees(e["yaw"]))
    )

    print("  rendering god view…")
    god = world.render_god_view()
    _save(out_dir / "god_view.png", god)

    print("  rendering 4 fisheyes (OpenCV K/D)…")
    images = world.render_cameras()
    for name in world.rig.names:
        _save(out_dir / ("%s.png" % name), images[name])

    print("  stitching AVM BEV…")
    bev = world.stitch_avm(images)
    assert bev.shape == (h, w, 3), "BEV shape mismatch: %s" % (bev.shape,)
    _save(out_dir / "bev.png", bev)
    overlay = draw_slot_overlay_bev(bev, world.ego_xyz_yaw(), bev_cfg)
    _save(out_dir / "bev_overlay.png", overlay)

    # Packaged collage
    print("  building collage (god + 4 cams + BEV)…")
    collage = build_collage(god, images, overlay)
    collage_path = out_dir / "collage.png"
    collage.save(collage_path)
    print("  wrote %s size=%s bytes=%d" % (collage_path, collage.size, collage_path.stat().st_size))

    # Quick validation summary
    if not args.no_validate:
        print("  quick validation summary…")
        metrics = _quick_validation(world, bev)
        lines = [
            "# Demo quick validation",
            "",
            "Ego pose (FLU): x=%.3f y=%.3f z=%.3f yaw_deg=%.1f"
            % (e["x"], e["y"], e["z"], math.degrees(e["yaw"])),
            "BEV shape: %dx%d @ %.4f m/px" % (h, w, res),
            "nonzero_frac: %.3f" % metrics["nonzero_frac"],
            "coverage_frac (excl. ego ROI): %.3f" % metrics["coverage_frac"],
            "mean_luma: %.1f" % metrics["mean_luma"],
            "mean_luma_outside_roi: %.1f" % metrics["mean_luma_outside_roi"],
            "",
            "For full Phase 4 metrics (landmarks / seam / sensitivity):",
            "  python scripts/run_phase4_validate.py",
            "",
        ]
        summary = "\n".join(lines)
        (out_dir / "validation_summary.txt").write_text(summary, encoding="utf-8")
        print(summary)

    # Short ego motion
    if not args.no_motion:
        run_motion(world, out_dir)

    print("  GL:", world.rig.ensure_backend().gl_info)
    world.release()

    # How to re-run
    print("")
    print("=" * 60)
    print("Demo complete — key outputs")
    print("=" * 60)
    print("  collage:     %s" % (out_dir / "collage.png"))
    print("  god_view:    %s" % (out_dir / "god_view.png"))
    print("  cams:        %s/{front,rear,left,right}.png" % out_dir)
    print("  bev:         %s/bev.png (+ bev_overlay.png)" % out_dir)
    if not args.no_motion:
        print("  motion GIF:  %s/motion/bev_motion.gif" % out_dir)
        print("  motion PNG:  %s/motion/bev_frame_XX.png" % out_dir)
    print("")
    print("Re-run phases:")
    print("  python scripts/run_phase0_smoke.py")
    print("  python scripts/run_phase1_scene.py")
    print("  python scripts/run_phase2_fisheye.py")
    print("  python scripts/run_phase3_avm.py")
    print("  python scripts/run_phase4_validate.py")
    print("  python scripts/run_demo.py            # this script (Phase 5)")
    print("  python scripts/run_demo.py --no-motion")
    print("")
    print("Phase 5 demo OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
