#!/usr/bin/env python3
"""Legacy CLI: render 4 fisheye views via the Phase 0 World API.

Kept for backward compatibility. Prefer:
  scripts/run_phase0_smoke.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from scene.world import World

OUT_DIR = ROOT / "output"
N_BENCH_FRAMES = 5


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    world = World.from_config(ROOT / "configs" / "default.yaml")
    world.set_ego_pose(0.0, 0.0, 0.0, 0.0)

    # Warm-up + short bench
    _ = world.render_cameras()
    t0 = time.perf_counter()
    last = {}
    for _ in range(N_BENCH_FRAMES):
        last = world.render_cameras()
    elapsed = time.perf_counter() - t0
    fps = N_BENCH_FRAMES / elapsed if elapsed > 0 else 0.0
    print("GL backend via World API")
    print(
        "Bench: %d frames x 4 cams in %.3fs (≈%.1f surround-FPS)"
        % (N_BENCH_FRAMES, elapsed, fps)
    )

    w, h = world.rig.image_size
    for name, arr in last.items():
        path = OUT_DIR / ("%s.png" % name)
        Image.fromarray(arr).save(path)
        print("Wrote %s (%d bytes) shape=%s" % (path, path.stat().st_size, arr.shape))

    collage = Image.new("RGB", (w * 2, h * 2))
    collage.paste(Image.fromarray(last["front"]), (0, 0))
    collage.paste(Image.fromarray(last["right"]), (w, 0))
    collage.paste(Image.fromarray(last["left"]), (0, h))
    collage.paste(Image.fromarray(last["rear"]), (w, h))
    collage_path = OUT_DIR / "collage.png"
    collage.save(collage_path)
    print("Wrote %s (%d bytes)" % (collage_path, collage_path.stat().st_size))

    world.release()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
