"""Run headless with the simulator Python; requires the actual GL backend."""
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sensors.moderngl_backend import ModernGLFisheyeBackend

w, h = 64, 48
b = ModernGLFisheyeBackend(width=w, height=h, pinhole_capture_scale=1)
try:
    for channels in (3, 4):
        for seed in range(3):
            raw = np.random.default_rng(seed).integers(0, 256, (2*h, 2*w, channels), dtype=np.uint8)
            tex = b.ctx.texture((2*w, 2*h), channels, raw.tobytes())
            b._grid_fbo = b.ctx.framebuffer(color_attachments=[tex])
            preview = b.read_grid_preview()
            expected = cv2.resize(raw, (w, h), interpolation=cv2.INTER_AREA)[::-1, :, :3]
            assert np.array_equal(preview, expected), (channels, seed)
            b._grid_fbo.release()
            tex.release()
    print("GPU preview matches OpenCV pixels and row direction")
finally:
    b.release()
