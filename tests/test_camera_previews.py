"""Run with the simulator Python: python tests/test_camera_previews.py."""
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from view_render_worker import camera_previews

w, h = 64, 48
for channels in (3, 4):
    grid = np.random.default_rng(42).integers(0, 256, (h*2, w*2, channels), dtype=np.uint8)
    reference = np.ascontiguousarray(grid[::-1, :, :3]) if channels == 4 else grid
    previews = camera_previews(grid, w, h, bottom_up_rgba=channels == 4)
    for name, (row, col) in dict(front=(0, 0), left=(0, 1), right=(1, 0), rear=(1, 1)).items():
        old = cv2.resize(reference[row*h:(row+1)*h, col*w:(col+1)*w],
                         (w//2, h//2), interpolation=cv2.INTER_AREA)
        assert np.array_equal(previews[name], old), name
        assert previews[name].flags.c_contiguous
print("Camera preview pixels and orientation match")
