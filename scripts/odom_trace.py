"""Decode J05 original-library diagnostics without changing odometry state."""
import csv
import json
import math
from pathlib import Path
import struct
import time

HEADER = struct.Struct('<IHHHHIII')
SAMPLE = struct.Struct('<IIQQIIHHhhBBBBffffffddddfffIBBBB')
FIELDS = ('sequence flags mcu_us chassis_us actual_dt_us integral_exec_us '
          'count_rl count_rr delta_rl delta_rr speed_dir gear yaw_rate_valid imu_flag '
          'yaw_rate_input_dps last_yaw_rate_dps yaw_rate_bias_dps distance_per_pulse_m '
          'pulse_distance_m still_time_s origin_before_rad origin_after_rad '
          'filter_before_rad filter_after_rad wheel_x_m wheel_y_m wheel_yaw_rad '
          'error_code odometry_function wheel_valid bias_solved reserved').split()


def decode(payload):
    """Topic body = Android dispatch timestamp (us) + unchanged MCU batch."""
    if len(payload) < 8 + HEADER.size:
        raise ValueError('short ODOM trace header')
    dispatch_us, = struct.unpack_from('<Q', payload)
    magic, version, hs, ss, count, latest, ring, reserved = HEADER.unpack_from(payload, 8)
    if (magic != 0x3152544F or version != 1 or hs != HEADER.size or
            ss != SAMPLE.size or not 1 <= count <= 8 or ring != 32 or
            len(payload) != 8 + hs + count * ss):
        raise ValueError('invalid ODOM trace version/size/count')
    samples = [dict(zip(FIELDS, SAMPLE.unpack_from(payload, 8 + hs + i * ss)))
               for i in range(count)]
    if any(not 0 < r['sequence'] <= latest for r in samples):
        raise ValueError('invalid source sequence')
    if any(a['sequence'] >= b['sequence'] for a, b in zip(samples, samples[1:])):
        raise ValueError('unordered trace batch')
    return dispatch_us, samples


class TraceCsvWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.stream = self.path.open('w', newline='', buffering=65536)
        self.raw = self.path.with_suffix('.wire.log').open('wb', buffering=65536)
        self.writer = csv.writer(self.stream)
        self.writer.writerow(['rx_mono', 'rx_wall', 'soc_dispatch_us', 'relay_rx_us',
                              'epoch', 'sample_gap', 'nonfinite_fields'] + FIELDS)
        self.last = None
        self.epoch = 0
        self.seen = set()
        self.stats = dict(batches=0, samples=0, duplicates=0, stale=0,
                          missing_samples=0, invalid_batches=0, clock_regressions=0)

    def take(self, line, rx_mono=None, rx_wall=None):
        rx_mono = time.monotonic() if rx_mono is None else rx_mono
        rx_wall = time.time() if rx_wall is None else rx_wall
        self.raw.write(('%.9f %.6f ' % (rx_mono, rx_wall)).encode() + line + b'\n')
        try:
            tag, relay_us, encoded = line.split()
            if tag != b'ODT1' or len(encoded) > 1984:
                raise ValueError('invalid diagnostic line')
            relay_us = int(relay_us)
            dispatch_us, samples = decode(bytes.fromhex(encoded.decode('ascii')))
        except (ValueError, UnicodeError):
            self.stats['invalid_batches'] += 1
            return
        self.stats['batches'] += 1
        newest = samples[-1]
        newest_key = (newest['sequence'], newest['mcu_us'])
        if (self.last and newest['sequence'] <= self.last['sequence'] and
                newest['mcu_us'] < self.last['mcu_us'] and newest_key not in self.seen):
            self.epoch += 1
            self.last = None
            self.seen.clear()
        for sample in samples:
            key = (sample['sequence'], sample['mcu_us'])
            if key in self.seen:
                self.stats['duplicates'] += 1
                continue
            if self.last and sample['sequence'] <= self.last['sequence']:
                self.stats['stale'] += 1
                continue
            gap = 0 if self.last is None else sample['sequence'] - self.last['sequence'] - 1
            if self.last and sample['mcu_us'] < self.last['mcu_us']:
                self.stats['clock_regressions'] += 1
            self.stats['missing_samples'] += gap
            nonfinite = ','.join(k for k, v in sample.items()
                                 if isinstance(v, float) and not math.isfinite(v))
            self.writer.writerow(['%.9f' % rx_mono, '%.6f' % rx_wall, dispatch_us,
                                  relay_us, self.epoch, gap, nonfinite] +
                                 [sample[k] for k in FIELDS])
            self.stats['samples'] += 1
            self.seen.add(key)
            self.last = sample
        # Only recent overlap needs deduplication; source sequence rejects old data.
        if len(self.seen) > 4096:
            self.seen = {(r['sequence'], r['mcu_us']) for r in samples}

    def close(self):
        self.stream.close()
        self.raw.close()
        self.path.with_suffix('.stats.json').write_text(json.dumps(dict(
            self.stats, epochs=self.epoch + 1, last_sample=self.last,
            time_alignment='MCU, Android monotonic and TX2 clocks remain separate'),
            indent=2) + '\n')
