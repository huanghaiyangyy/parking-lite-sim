"""RGBA pipe writes must preserve bytes even when the kernel writes partially."""
import queue
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from stream_h264_adb import JetsonEncoder

class ShortInput:
    def __init__(self):
        self.data = bytearray()

    def write(self, data):
        assert data.ndim == 1
        n = min(11, len(data))
        self.data.extend(data[:n])
        return n

for channels in (3, 4):
    raw = np.random.default_rng(5).integers(0, 256, (5, 7, channels), dtype=np.uint8)
    encoder = JetsonEncoder.__new__(JetsonEncoder)
    encoder.height, encoder.width, encoder.error = 5, 7, None
    encoder.pending = queue.Queue()
    encoder.proc = SimpleNamespace(stdin=ShortInput())
    encoder.push(23, 123456, raw)
    expected = raw
    if channels == 3:
        expected = np.full((5, 7, 4), 255, dtype=np.uint8)
        expected[:, :, :3] = raw[::-1]
    assert encoder.proc.stdin.data == expected.tobytes()
    assert encoder.pending.get_nowait() == (23, 123456)
print("Jetson partial pipe writes preserve frame bytes and order")
