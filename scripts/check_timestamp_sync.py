"""Run: python scripts/check_timestamp_sync.py (no network, rendering, or CAN)."""
from collections import deque
from pathlib import Path
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from board_clock import BoardClock
from live_vehicle import LiveVehicle


class OneSampleBatch:
    stopped = False

    def is_set(self):
        return self.stopped

    def wait(self, seconds):
        self.stopped = True


def clock_with_samples(rtts):
    clock = BoardClock.__new__(BoardClock)
    clock._stop = OneSampleBatch()
    clock._ready = threading.Event()
    clock._lock = threading.Lock()
    clock._error = None
    clock.tx2_us = clock.board_us = clock.rtt_us = 0
    clock.resample_s, clock.max_rtt_us = 5, 40000
    clock._forward = lambda: None
    samples = iter((rtt, 1000000+i, 5000000+i) for i, rtt in enumerate((rtts * 20)[:20]))
    clock._read_sample = lambda: next(samples)
    clock._tx2_us = lambda: 1100000
    clock._loop()
    return clock


def main():
    clock = clock_with_samples([45000, 18000, 8000, 12000, 35000])
    assert clock._ready.is_set() and clock.rtt_us == 8000
    capture = 1050000
    pts = clock.to_board_us(capture)
    assert pts == 5050000
    clock._tx2_us = lambda: 1300000  # 200 ms of rendering/queuing elapsed.
    assert clock.to_board_us(capture) == pts, "render latency must not change capture PTS"
    assert clock.now_us() - pts == 250000

    clock._tx2_us = lambda: 18000000
    try:
        clock.to_board_us(capture)
    except RuntimeError as exc:
        assert "stale" in str(exc)
    else:
        raise AssertionError("a lost clock must fail instead of inventing a timestamp")

    bad = clock_with_samples([45000, 50000, 46000, 48000, 60000])
    assert not bad._ready.is_set() and bad.board_us == 0
    assert "RTT" in str(bad._error)

    vehicle = LiveVehicle.__new__(LiveVehicle)
    vehicle._lock = threading.Lock()
    pose = object()
    vehicle._history = deque([(1.25, pose)])
    assert vehicle.timed_state() == (1.25, pose)
    print("PASS: physics pose/source pairing, render-delay invariance, RTT rejection, stale-clock rejection")


if __name__ == "__main__":
    main()
