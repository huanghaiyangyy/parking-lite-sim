#!/usr/bin/env python3
"""Demo: configurable kinematic vehicle control + World pose sync.

Simulates ~8 s: D accel → steer curve → brake → short R reverse.
Writes CSV, trajectory plot, and a few god_view frames.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw

from scene.world import World
from vehicle import VehicleController


def _save_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)
    print("  wrote %s shape=%s" % (path, arr.shape))


def _controls_at(t: float) -> tuple[float, float, str]:
    """Piecewise driver schedule (steer_wheel_deg, accel_mps2, gear)."""
    # 0–2 s: D, accelerate straight
    if t < 2.0:
        return 0.0, 1.5, "D"
    # 2–4.5 s: D, hold / light accel while steering left
    if t < 4.5:
        return 280.0, 0.3, "D"
    # 4.5–6.0 s: brake to stop (still D)
    if t < 6.0:
        return 100.0, -3.0, "D"
    # 6.0–6.3 s: N dwell
    if t < 6.3:
        return 0.0, 0.0, "N"
    # 6.3–8.0 s: R reverse with slight steer
    if t < 8.0:
        return -120.0, 1.2, "R"
    # coast
    return 0.0, -2.0, "R"


def _plot_trajectory(xs, ys, out_path: Path) -> None:
    """Minimal trajectory plot without matplotlib (Pillow)."""
    w, h = 640, 640
    margin = 40
    img = Image.new("RGB", (w, h), (24, 28, 36))
    draw = ImageDraw.Draw(img)
    if len(xs) < 2:
        img.save(out_path)
        return
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    # Pad extent
    pad = 0.5
    xmin -= pad
    xmax += pad
    ymin -= pad
    ymax += pad
    span = max(xmax - xmin, ymax - ymin, 1e-3)
    # Center square view
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    xmin = cx - 0.5 * span
    xmax = cx + 0.5 * span
    ymin = cy - 0.5 * span
    ymax = cy + 0.5 * span

    def to_px(x: float, y: float) -> tuple[int, int]:
        # Image: +x right, +y down. World FLU: +x fwd, +y left.
        # Map x→right, y→up on plot (invert image y).
        u = margin + (x - xmin) / span * (w - 2 * margin)
        v = h - margin - (y - ymin) / span * (h - 2 * margin)
        return int(round(u)), int(round(v))

    # Axes
    draw.rectangle([margin, margin, w - margin, h - margin], outline=(60, 70, 90))
    o = to_px(0.0, 0.0)
    draw.line([to_px(xmin, 0), to_px(xmax, 0)], fill=(50, 55, 70), width=1)
    draw.line([to_px(0, ymin), to_px(0, ymax)], fill=(50, 55, 70), width=1)
    draw.ellipse([o[0] - 3, o[1] - 3, o[0] + 3, o[1] + 3], fill=(180, 180, 100))

    pts = [to_px(x, y) for x, y in zip(xs, ys)]
    draw.line(pts, fill=(80, 180, 255), width=2)
    # Start / end markers
    s, e = pts[0], pts[-1]
    draw.ellipse([s[0] - 5, s[1] - 5, s[0] + 5, s[1] + 5], fill=(80, 220, 120))
    draw.ellipse([e[0] - 5, e[1] - 5, e[0] + 5, e[1] + 5], fill=(220, 100, 100))
    draw.text((8, 8), "trajectory FLU (x fwd, y left)", fill=(200, 200, 210))
    draw.text((8, h - 24), "green=start  red=end", fill=(160, 160, 170))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print("  wrote %s" % out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Kinematic vehicle control demo")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "default.yaml",
        help="Scene YAML (reads vehicle_config: for kinematics)",
    )
    parser.add_argument(
        "--vehicle-config",
        type=Path,
        default=None,
        help="Override kinematics YAML (default: scene vehicle_config or configs/vehicle.yaml)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "output" / "vehicle_control",
        help="Output directory",
    )
    args = parser.parse_args()

    scene_cfg = args.config
    world = World.from_config(scene_cfg)
    rel = (world.config.get("vehicle_config") or "configs/vehicle.yaml")
    vehicle_cfg = args.vehicle_config or (ROOT / rel)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Vehicle control demo")
    print("  vehicle_config:", vehicle_cfg)
    print("  scene_config:", scene_cfg)

    ctrl = VehicleController(vehicle_cfg)
    dt = ctrl.dt_s
    # Start near aisle origin, yaw 0 (facing +x)
    ctrl.reset(x=0.0, y=0.0, yaw=0.0, speed=0.0)

    world.set_ego_pose(0.0, 0.0, 0.0, 0.0)

    duration_s = 8.0
    n_steps = int(round(duration_s / dt))
    log_every = max(1, int(round(0.1 / dt)))  # ~10 Hz table rows
    god_every = max(1, int(round(2.0 / dt)))  # god frame every ~2 s

    csv_path = out_dir / "trajectory.csv"
    rows = []
    xs, ys = [], []

    print(
        "  %-6s %7s %7s %4s %8s %8s %8s %8s %6s"
        % ("t", "steer", "accel", "gear", "speed", "yaw_deg", "x", "y", "pulses")
    )

    t = 0.0
    for i in range(n_steps + 1):
        steer, accel, gear = _controls_at(t)
        ctrl.set_controls(steer, accel, gear)
        if i > 0:
            state = ctrl.step(dt)
            t = i * dt
        else:
            state = ctrl.get_state()

        ctrl.apply_to_world(world, z=0.0)
        xs.append(state.x)
        ys.append(state.y)
        pulses = state.wheel_pulse_count
        pulse_sum = sum(pulses)

        if i % log_every == 0 or i == n_steps:
            print(
                "  %-6.2f %7.1f %7.2f %4s %8.3f %8.1f %8.3f %8.3f %6d"
                % (
                    t,
                    state.steer_wheel_deg,
                    accel,
                    state.gear,
                    state.speed_mps,
                    math.degrees(state.yaw),
                    state.x,
                    state.y,
                    pulse_sum,
                )
            )

        rows.append(
            {
                "t_s": "%.4f" % t,
                "steer_wheel_deg": "%.3f" % state.steer_wheel_deg,
                "road_wheel_deg": "%.3f" % state.road_wheel_deg,
                "accel_cmd_mps2": "%.3f" % accel,
                "gear": state.gear,
                "speed_mps": "%.6f" % state.speed_mps,
                "speed_signal_mps": "%.6f" % state.speed_signal_mps,
                "yaw_rad": "%.6f" % state.yaw,
                "yaw_deg": "%.4f" % math.degrees(state.yaw),
                "x_m": "%.6f" % state.x,
                "y_m": "%.6f" % state.y,
                "pulse_fl": pulses[0],
                "pulse_fr": pulses[1],
                "pulse_rl": pulses[2],
                "pulse_rr": pulses[3],
                "wspeed_fl": "%.6f" % state.wheel_speed_mps[0],
                "wspeed_fr": "%.6f" % state.wheel_speed_mps[1],
                "wspeed_rl": "%.6f" % state.wheel_speed_mps[2],
                "wspeed_rr": "%.6f" % state.wheel_speed_mps[3],
            }
        )

        if i % god_every == 0 or i == n_steps:
            god = world.render_god_view()
            _save_png(out_dir / ("god_view_t%04.1fs.png" % t), god)

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("  wrote %s (%d rows)" % (csv_path, len(rows)))

    _plot_trajectory(xs, ys, out_dir / "trajectory.png")

    # Quick sanity asserts for success criteria (demo proof)
    final = ctrl.get_state()
    assert sum(final.wheel_pulse_count) > 0, "pulses should increase with motion"
    # Path should have curved (y moved) during D steer segment
    assert max(abs(y) for y in ys) > 0.05, "steering should curve path (y displacement)"
    # R segment should have driven vx negative at some point after t=6.3
    had_reverse = any(float(r["speed_mps"]) < -0.05 for r in rows if float(r["t_s"]) >= 6.3)
    assert had_reverse, "R gear should produce negative body vx"

    world.release()
    print("  output:", out_dir)
    print("  OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
