#!/usr/bin/env python3
"""Compare CAN bytes with the original bitwise encoder; never opens CAN."""
import random
from statistics import median
import time
from unittest.mock import patch

import apa_can_bridge as can


def reference_crc(data, counter, table):
    crc = 0xFF
    for byte in list(data[1:]) + [table[counter & 15]]:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) & 255) ^ (0x2F if crc & 128 else 0)
    return crc ^ 255


def reference_put(buf, start, length, value):
    for bit in range(length):
        pos = start + bit
        mask = 1 << (pos % 8)
        buf[pos // 8] = (buf[pos // 8] & ~mask) | (((value >> bit) & 1) * mask)


def reference_get(buf, start, length):
    return sum(((buf[(start + i) // 8] >> ((start + i) % 8)) & 1) << i
               for i in range(length))


def main():
    rng = random.Random(19020)
    for _ in range(3000):
        buf = bytearray(rng.getrandbits(8) for _ in range(rng.randint(1, 64)))
        start = rng.randrange(len(buf) * 8)
        length = rng.randint(0, len(buf) * 8 - start)
        value = rng.getrandbits(length + 2) * rng.choice((-1, 0, 1))
        expected = buf[:]
        reference_put(expected, start, length, value)
        can.put_bits(buf, start, length, value)
        assert buf == expected
        assert can.get_bits(buf, start, length) == reference_get(buf, start, length)
        table = rng.choice([f[4] for f in can.FRAMES if f[4]] + [can.ADAS_07_CRC])
        counter = rng.randrange(256)
        assert can.crc8(buf, counter, table) == reference_crc(buf, counter, table)

    state, adas = can.ChassisState(), can.Adas07()
    for index in range(128):
        state.update(speed_kph=rng.uniform(0, 30), steer_deg=rng.uniform(-500, 500),
                     yaw_rate_dps=rng.uniform(-40, 40), long_accel=rng.uniform(-5, 5),
                     pulses=[rng.randrange(100000) for _ in range(4)],
                     gear=rng.randint(1, 4), brake_pressed=bool(index & 1),
                     standstill_confirmed=bool(index & 2))
        adas.stamp, adas.crc_ok = time.monotonic(), True
        adas.eps_active, adas.vlc_req = index % 3, (index // 3) % 3
        counters = {f[0]: index & 15 for f in can.FRAMES}
        original_counts = counters.copy()
        actual = can.encode_frames(state, adas, counters, can.FRAMES)
        with patch.multiple(can, put_bits=reference_put, get_bits=reference_get,
                            crc8=reference_crc):
            expected = can.encode_frames(state, adas, original_counts, can.FRAMES)
        assert actual == expected and counters == original_counts

    def benchmark():
        samples = []
        for _ in range(5):
            started = time.perf_counter()
            for _ in range(300):
                can.encode_frames(state, adas, counters, can.FRAMES)
            samples.append((time.perf_counter() - started) * 1e6 / 300)
        return median(samples)

    # Keep handshake freshness identical across both benchmark phases.
    adas.stamp = time.monotonic() + 60
    optimized = benchmark()
    with patch.multiple(can, put_bits=reference_put, get_bits=reference_get,
                        crc8=reference_crc):
        original = benchmark()
    print("CAN bytes match: 3000 randomized bit/CRC cases, 128 full batches.")
    print("18-frame encode median: original %.1f us, optimized %.1f us (%.2fx)" %
          (original, optimized, original / optimized))


if __name__ == '__main__':
    main()
