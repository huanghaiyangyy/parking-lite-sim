#!/usr/bin/env python3
"""Regression checks for live sensor timing; no CAN device or renderer needed."""
from pathlib import Path
import csv
import math
import socket
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from apa_can_bridge import (ADAS_07_CRC, CanLink, chassis_from_vehicle, crc8,
                            get_bits, put_bits)
from live_vehicle import LiveVehicle
from odom_compare import OdomCompare, pose_error
from vehicle import VehicleState


def active_command():
    frame = bytearray(8)
    put_bits(frame, 53, 3, 2)  # reverse
    put_bits(frame, 17, 14, 5800)  # straight
    put_bits(frame, 31, 2, 2)
    put_bits(frame, 36, 2, 2)
    put_bits(frame, 40, 8, 160)  # +1 m/s2 body acceleration
    put_bits(frame, 8, 4, 1)
    put_bits(frame, 0, 8, crc8(frame, 1, ADAS_07_CRC))
    return frame


def wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError('condition timed out')


def main():
    # The current yaw rate must remain correct on the first frame and when
    # the preceding UI interval had a different duration or crossed +/-pi.
    current = VehicleState(yaw=math.radians(.2), yaw_rate_rps=0.)
    previous = VehicleState(yaw=0., yaw_rate_rps=math.radians(10.))
    assert chassis_from_vehicle(current, previous, .2)['yaw_rate_dps'] == 0.
    assert abs(chassis_from_vehicle(previous)['yaw_rate_dps'] - 10.) < 1e-12

    with tempfile.TemporaryDirectory(prefix='tx2-live-check-') as temp:
        out = Path(temp)
        link = CanLink(dry_run=True)
        runtime = LiveVehicle(ROOT / 'configs/j01_vehicle.yaml', dict(x=0., y=0., yaw=0.),
                              link, out / 'deterministic')
        try:
            with patch('live_vehicle.time.monotonic', return_value=10.):
                runtime.reset(0., 0., math.pi - .001, speed=1.)
            runtime.set_controls(380., 0., 'D')
            state, _, dt = runtime.tick(10.01)
            half_stamp, half = runtime.pose_at(10.005)
            assert half_stamp == 10.005
            assert abs(half.yaw - (math.pi - .001 + .5 * state.yaw_rate_rps * .01)) < 1e-10
            # Variable real elapsed time is integrated in small substeps.
            state, _, dt = runtime.tick(10.21)
            assert abs(dt - .2) < 1e-12
            assert abs((state.wheel_speed_mps[2] + state.wheel_speed_mps[3]) / 2 - 1.) < 1e-12
            assert state.wheel_pulse_count[3] > state.wheel_pulse_count[2]
            before = state.wheel_pulse_count[:]
            reset = runtime.reset(0., 0., 0.)
            assert reset.wheel_pulse_count == before and max(before) > 0
        finally:
            runtime.close()

        # Deliberately stop all UI publication for 350 ms. Physics and sensor
        # frames must continue, including fresh ADAS selection and expiry.
        link = CanLink(dry_run=True)
        runtime = LiveVehicle(ROOT / 'configs/j01_vehicle.yaml', dict(x=0., y=0., yaw=0.),
                              link, out / 'threaded')
        try:
            runtime.reset(0., 0., 0., speed=1.)
            runtime.set_controls(90., 0., 'D')
            runtime.start()
            wait_for(lambda: link.sent['Airbag_02'] >= 2)
            start = runtime.get_state()
            time.sleep(.35)
            state = runtime.get_state()
            assert math.hypot(state.x - start.x, state.y - start.y) > .25
            assert link.sent['Airbag_02'] >= 15 and runtime.status()['sequence'] >= 15
            runtime.set_controls(0., 0., 'P')
            link.inject(active_command())
            wait_for(lambda: runtime.get_state().gear == 'R')
            assert runtime.status()['source'] == 'CAN'
            runtime.set_controls(0., -4., 'D', brake_pressed=True)
            wait_for(lambda: link.chassis.snapshot()['brake_pressed'])
            assert runtime.get_state().gear == 'D'
            runtime.set_controls(0., 0., 'P')
            wait_for(lambda: runtime.status()['source'] == 'keys' and runtime.get_state().gear == 'P')
            assert link.error is None, link.error
        finally:
            runtime.close()
        assert not link._tx.is_alive()
        truth = list(csv.DictReader((out / 'threaded/sim_truth.csv').open()))
        sent = list(csv.DictReader((out / 'threaded/can_tx.csv').open()))
        assert len(truth) >= 15
        for row in sent:
            if row['name'] != 'Airbag_02':
                continue
            raw = get_bits(bytes.fromhex(row['payload_hex']), 104, 16)
            actual = raw * .01 - 327.68
            ref = next(r for r in reversed(truth) if float(r['sim_mono']) <= float(row['tx_mono']))
            assert abs(actual - float(ref['yaw_rate_dps'])) <= .005001
        assert all(float(b['sim_mono']) >= float(a['sim_mono']) for a, b in zip(truth, truth[1:]))

        # Real TCP parsing preserves both source and receive timestamps;
        # invalid sensor values cannot become a comparison baseline.
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        odom = OdomCompare('', port)
        odom.enable_trace(out / 'odom_rx.csv')
        connection = None
        try:
            def connect():
                nonlocal connection
                try:
                    connection = socket.create_connection(('127.0.0.1', port), timeout=.1)
                    return True
                except OSError:
                    return False
            wait_for(connect)
            connection.sendall(b'1 nan 0 0 0 0 0\n2 10 21 1.570796 0 0 0\n')
            wait_for(lambda: odom.status()[0] is not None)
            sample, error = odom.status()
            assert error is None and sample['stamp'] == 2. and sample['sequence'] == 1
            assert sample['received_monotonic'] > 0 and sample['received_wall'] > 0
            origin = dict(x=10., y=20., yaw=math.pi / 2, sim_x=0., sim_y=0., sim_yaw=0.)
            err = pose_error(1., 0., 0., sample, origin)
            assert abs(err['err_dx']) < 1e-12 and abs(err['err_dy']) < 1e-12
        finally:
            if connection is not None:
                connection.close()
            odom.close()
        assert not odom._thread.is_alive()
        assert len(list(csv.DictReader((out / 'odom_rx.csv').open()))) == 1
    print('PASS: instantaneous yaw rate, rear-wheel mean, pulse-preserving reset, yaw interpolation')
    print('PASS: physics/CAN continue through UI stalls; ADAS takeover, brake override, expiry')
    print('PASS: sent CAN yaw matches truth; session traces and timestamped ODOM reception')


if __name__ == '__main__':
    main()
