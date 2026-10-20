#!/usr/bin/env python3
"""Cross-check MCU-produced bytes, CSV bookkeeping and the real TCP receiver."""
import csv
import json
from pathlib import Path
import socket
import struct
import tempfile
import time

from odom_compare import OdomCompare
from odom_trace import HEADER, SAMPLE, FIELDS, TraceCsvWriter, decode


def line(payload):
    return b'ODT1 8000100 ' + payload.hex().encode()


def main():
    # Produced by mcu_base_whitebox/tools/check_odom_trace.py, not by this parser.
    fixture = (Path(__file__).parent/'testdata/odom_trace_v1.bin').read_bytes()
    payload = struct.pack('<Q', 8000000) + fixture
    dispatch, samples = decode(payload)
    assert dispatch == 8000000 and len(samples) == 8
    assert [s['sequence'] for s in samples] == list(range(33, 41))
    assert samples[0]['mcu_us'] == 1330000 and samples[-1]['mcu_us'] == 1400000
    assert all(s['delta_rl'] == -2 and s['delta_rr'] == -3 for s in samples)
    assert all(s['actual_dt_us'] == 10000 and s['integral_exec_us'] == 7 for s in samples)
    assert all(s['yaw_rate_bias_dps'] == .125 and s['imu_flag'] == 1 for s in samples)

    def single(sequence, mcu_us, nan=False):
        sample = dict(samples[-1], sequence=sequence, mcu_us=mcu_us)
        if nan:
            sample['yaw_rate_input_dps'] = float('nan')
        header = HEADER.pack(0x3152544F, 1, 24, 120, 1, sequence, 32, 0)
        return struct.pack('<Q', 8000000) + header + SAMPLE.pack(*[sample[k] for k in FIELDS])

    for invalid in [payload[:20], payload[:-1], payload+b'\0',
                    payload[:12]+b'\xff\xff'+payload[14:]]:
        try:
            decode(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('malformed diagnostic accepted')

    with tempfile.TemporaryDirectory(prefix='tx2-odom-trace-') as directory:
        out = Path(directory)
        writer = TraceCsvWriter(out/'bookkeeping.csv')
        writer.take(line(payload))
        writer.take(line(payload))
        writer.take(line(single(43, 1430000)))
        writer.take(line(single(44, 1440000, nan=True)))
        writer.take(b'ODT1 invalid not-hex')
        writer.take(line(single(1, 1000)))
        writer.close()
        stats = json.loads((out/'bookkeeping.stats.json').read_text())
        assert stats['samples'] == 11 and stats['duplicates'] == 8
        assert stats['missing_samples'] == 2 and stats['epochs'] == 2
        assert stats['invalid_batches'] == 1
        with (out/'bookkeeping.csv').open() as f:
            recorded = list(csv.DictReader(f))
        assert recorded[-1]['epoch'] == '1'
        assert recorded[-2]['nonfinite_fields'] == 'yaw_rate_input_dps'

        probe = socket.socket()
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
        probe.close()
        receiver = OdomCompare(None, port)
        receiver.enable_trace(out/'odom_rx.csv')
        client = None
        try:
            deadline = time.monotonic() + 3
            while client is None and time.monotonic() < deadline:
                try:
                    client = socket.create_connection(('127.0.0.1', port), timeout=.2)
                except OSError:
                    time.sleep(.02)
            assert client is not None, receiver.status()
            data = b'1 0 0 .1 0 0 .1\n' + line(payload) + b'\n2 1 2 .2 1 2 .2\n'
            client.sendall(data[:23])
            client.sendall(data[23:])
            while time.monotonic() < deadline:
                sample, error = receiver.status()
                if sample and sample['stamp'] == 2:
                    break
                time.sleep(.02)
            assert sample['stamp'] == 2 and sample['sequence'] == 2, (sample, error)
        finally:
            if client is not None:
                client.close()
            receiver.close()
        stats = json.loads((out/'mcu_odom_trace.stats.json').read_text())
        assert stats['samples'] == 8 and stats['missing_samples'] == 0
        with (out/'odom_rx.csv').open() as f:
            assert len(list(csv.DictReader(f))) == 2
    print('PASS: C wire ABI, validation, overlap dedup, gaps, MCU restart, NaN evidence, multiplexed TCP')


if __name__ == '__main__':
    main()
