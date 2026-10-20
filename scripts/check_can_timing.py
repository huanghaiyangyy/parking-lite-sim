#!/usr/bin/env python3
"""Passively capture selected CAN frames or summarize candump -L timestamps."""
import argparse
import json
from pathlib import Path
import re
import statistics
import subprocess

FRAMES = {0x190: ('Airbag_02', 10), 0x20C: ('iBRS_04', 20),
          0x0A0: ('iBRS_01', 10), 0x214: ('iBRS_05', 20)}
LINE = re.compile(r'^\((\d+\.\d+)\)\s+\S+\s+([0-9A-Fa-f]+)##?[0-9A-Fa-f]+')


def percentile(values, percent):
    values = sorted(values)
    index = (len(values) - 1) * percent / 100
    low = int(index)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (index - low)


def summarize(path):
    stamps = {can_id: [] for can_id in FRAMES}
    for line in Path(path).read_text().splitlines():
        match = LINE.match(line)
        if match:
            stamp, can_id = float(match[1]), int(match[2], 16)
            if can_id in stamps:
                stamps[can_id].append(stamp)
    result = {}
    for can_id, times in stamps.items():
        name, period = FRAMES[can_id]
        dt = [(b-a)*1000 for a, b in zip(times, times[1:])]
        if not dt:
            result[name] = dict(can_id=hex(can_id), frames=len(times), error='fewer than two frames')
            continue
        result[name] = dict(can_id=hex(can_id), frames=len(times), target_ms=period,
            mean_ms=statistics.mean(dt), min_ms=min(dt), max_ms=max(dt),
            p50_ms=percentile(dt, 50), p95_ms=percentile(dt, 95), p99_ms=percentile(dt, 99),
            abs_jitter_p99_ms=percentile([abs(v-period) for v in dt], 99),
            gaps_over_1_5_periods=sum(v > period*1.5 for v in dt))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='existing candump -L file')
    parser.add_argument('--iface', default='can0')
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error('--seconds must be positive')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    source = args.input
    if source is None:
        source = args.out.with_suffix('.candump')
        filters = args.iface + ''.join(',%03X:7FF' % can_id for can_id in FRAMES)
        with source.open('wb') as log:
            proc = subprocess.Popen(['candump', '-L', filters], stdout=log, stderr=subprocess.PIPE)
            try:
                proc.wait(timeout=args.seconds)
                raise RuntimeError('candump exited early: ' + proc.stderr.read().decode())
            except subprocess.TimeoutExpired:
                proc.send_signal(2)
                proc.wait(timeout=3)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=3)
    result = dict(source=str(source), timestamp='SocketCAN kernel echo/RX timestamp; not MCU arrival',
                  messages=summarize(source))
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
