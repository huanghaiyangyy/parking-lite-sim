#!/usr/bin/env python3
"""APA chassis CAN bridge.

Sends the chassis frames the J01 MCU reads and receives ADAS_07 (0x226).
Each message uses its own CRC-8 table. ChassisState.update() is the
simulator hook; this process does not move the vehicle by itself.

    python3 scripts/apa_can_bridge.py --check
    python3 scripts/apa_can_bridge.py --dry-run
    python3 scripts/apa_can_bridge.py --iface can0          # TX2 only
    python3 scripts/apa_can_bridge.py --sim --seconds 2     # local model, no socket

Stop sentry-cansend.service, sentry-awake.service and other CAN senders first.
Do not start this while sentry_cansend.py owns the overlapping ids
(0x0A9, 0x2B3, 0x2B6, 0x400, 0x620, 0x632).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CAN_FD_BRS = 0x01
CAN_FD_ESI = 0x02
CAN_FD_FDF = 0x04
CAN_FRAME_LEN = 72

GEAR_P, GEAR_R, GEAR_N, GEAR_D = 1, 2, 3, 4
GEAR_FROM_SIM = {"P": GEAR_P, "R": GEAR_R, "N": GEAR_N, "D": GEAR_D}
GEAR_TO_SIM = {GEAR_P: "P", GEAR_R: "R", GEAR_N: "N", GEAR_D: "D"}
DIR_FORWARD, DIR_BACKWARD, DIR_STOP = 1, 2, 3


def _crc8_lut():
    values = []
    for crc in range(256):
        for _ in range(8):
            crc = ((crc << 1) & 0xFF) ^ (0x2F if crc & 0x80 else 0)
        values.append(crc)
    return tuple(values)


CRC8_LUT = _crc8_lut()


def crc8(data, counter, table):
    """CRC-8/2F plus the per-counter table from the Cockpit DBC."""
    crc = 0xFF
    for byte in data[1:]:
        crc = CRC8_LUT[crc ^ byte]
    return CRC8_LUT[crc ^ table[counter & 0x0F]] ^ 0xFF


def put_bits(buf, start, length, value):
    mask = (1 << length) - 1
    packed = int.from_bytes(buf, "little")
    packed = (packed & ~(mask << start)) | ((value & mask) << start)
    buf[:] = packed.to_bytes(len(buf), "little")


def get_bits(buf, start, length):
    return (int.from_bytes(buf, "little") >> start) & ((1 << length) - 1)


def raw_linear(physical, factor, offset, length):
    return int(round((physical - offset) / factor)) & ((1 << length) - 1)


# name, can id, dlc, period ms, crc table
FRAMES = (
    ("iBRS_01", 0x0A0, 16, 10,
     (0x6B, 0x74, 0xD7, 0x6F, 0x96, 0x66, 0xF6, 0x07,
      0x5B, 0x97, 0xF0, 0xF4, 0x1C, 0x23, 0x8B, 0x51)),
    ("iBRS_03", 0x213, 16, 20,
     (0xF2, 0xA5, 0xC2, 0x5A, 0xA7, 0x88, 0x45, 0x77,
      0xAD, 0x50, 0x4C, 0x5F, 0x43, 0xEF, 0x90, 0xEA)),
    ("iBRS_04", 0x20C, 16, 20,
     (0x25, 0xF8, 0x32, 0xEC, 0x4F, 0x53, 0xB4, 0xC4,
      0xF9, 0x30, 0x04, 0x58, 0x3E, 0x86, 0x63, 0xF6)),
    ("iBRS_05", 0x214, 8, 20,
     (0xDB, 0xE8, 0x8E, 0x40, 0x9F, 0x60, 0x85, 0xDE,
      0x80, 0xC2, 0x0F, 0xAC, 0x07, 0xE3, 0x14, 0xA4)),
    ("iBRS_11", 0x217, 8, 20,
     (0x68, 0x0E, 0x46, 0x1E, 0x0A, 0xBF, 0xF7, 0x6C,
      0xB0, 0x7F, 0x51, 0xFA, 0x82, 0xC0, 0x3B, 0x14)),
    ("EPS_Fr01", 0x0C0, 8, 10,
     (0x27, 0xAF, 0x0D, 0x0F, 0x31, 0x28, 0x91, 0xD1,
      0xB8, 0xC5, 0x62, 0x6B, 0x76, 0xCA, 0x64, 0x63)),
    ("EPS_Fr02", 0x0C1, 12, 10,
     (0x72, 0xB1, 0x5A, 0xB8, 0x2B, 0x77, 0x32, 0xDF,
      0x5F, 0xED, 0x1D, 0xEA, 0xE8, 0x53, 0x3F, 0xD7)),
    ("VCU_VehInfo", 0x0A9, 8, 10,
     (0x03, 0x17, 0xDC, 0x0B, 0x91, 0xBC, 0x96, 0xF5,
      0xAB, 0xCA, 0xCD, 0xCC, 0x93, 0x97, 0xEC, 0x35)),
    ("VCU_ADASInfo", 0x23C, 8, 20,
     (0x57, 0x39, 0x72, 0x0D, 0x51, 0xB1, 0x25, 0x28,
      0x5A, 0x52, 0x85, 0xB8, 0x38, 0xF8, 0x2B, 0x6B)),
    ("Airbag_02", 0x190, 20, 10,
     (0x0F, 0xD1, 0x6B, 0x63, 0x9E, 0x74, 0x22, 0x94,
      0xD7, 0xA1, 0xE7, 0x6F, 0x9C, 0xCF, 0x96, 0xDD)),
    ("BDU_0", 0x400, 4, 100,
     (0x5D, 0x96, 0x82, 0x97, 0xB6, 0x8B, 0x24, 0x02,
      0xEB, 0x69, 0x20, 0xBB, 0x77, 0x54, 0x86, 0xE6)),
    ("LDC_01", 0x2B3, 16, 20,
     (0xDA, 0x27, 0x39, 0xAF, 0xE5, 0x0D, 0xA0, 0x0F,
      0xB1, 0x31, 0x3E, 0x28, 0xF0, 0x91, 0x52, 0xD1)),
    ("RDC_01", 0x2B6, 8, 20, None),
    ("Airbag_01", 0x210, 8, 40,
     (0x0C, 0xF3, 0xCF, 0x5E, 0x06, 0xC9, 0x1E, 0xB7,
      0x23, 0x8F, 0x4E, 0xBE, 0xBC, 0x6C, 0x10, 0x26)),
    ("BDU_60", 0x620, 12, 1000, None),
    ("BDU_DspyInfo02", 0x632, 8, 1000, None),
    ("VCU_Chrg", 0x308, 12, 50,
     (0x7A, 0x29, 0x83, 0x15, 0x39, 0x9C, 0x3A, 0x45,
      0xA6, 0x0D, 0x12, 0xD6, 0x10, 0xDC, 0xB1, 0xEF)),
    # No E2E. Shift-stalk intent must stay not_verified during APA control.
    ("BDU_73", 0x171, 16, 10, None),
)

ADAS_07_ID = 0x226
ADAS_07_CRC = (
    0xC7, 0xF2, 0x9F, 0xA5, 0xA8, 0xC2, 0x15, 0x5A,
    0x14, 0xA7, 0x6D, 0x88, 0x13, 0x45, 0xD9, 0x77,
)


class ChassisState:
    """Values the simulator overwrites. Defaults are a stopped D-gear bench."""

    def __init__(self):
        self._lock = threading.Lock()
        self.gear = GEAR_D
        self.pt_st = 3
        self.accel_pedal = 0.0
        self.speed_kph = 0.0
        self.wheel_speed_kph = (0.0, 0.0, 0.0, 0.0)  # FL FR RL RR
        self.direction = DIR_STOP
        self.pulses = [0, 0, 0, 0]
        self.brake_pressed = False
        self.brake_pos = 0.0
        self.standstill = True
        self.standstill_confirmed = False
        self.epb = 0
        self.steer_deg = 0.0
        self.steer_rate_dps = 0.0
        self.torsion_nm = 0.0
        self.eps_ctrl = 0
        self.eps_assist_pct = 100.0
        self.eps_motor_nm = 0.0
        self.vlc_rsp = 0
        self.vlc_st = 0
        self.accel_target = 0.0
        self.long_accel = 0.0
        self.yaw_rate_dps = 0.0
        self.slope_deg = 0.0
        self.power_mode = 3
        self.auto_handshake = True

    def update(self, **fields):
        with self._lock:
            unknown = set(fields) - set(self.__dict__)
            if unknown:
                raise KeyError(sorted(unknown))
            self.__dict__.update(fields)

    def snapshot(self):
        with self._lock:
            data = {k: v for k, v in self.__dict__.items() if k != "_lock"}
            data["pulses"] = list(self.pulses)
            data["wheel_speed_kph"] = tuple(self.wheel_speed_kph)
            return data


def chassis_from_vehicle(state, prev=None, dt=0.02, command=None):
    """Map a VehicleState into ChassisState fields.

    Wheel pulses stay cumulative; the frame packer wraps each to 12 bits.
    Body acceleration uses the same sign as ADAS_APAVLCAxvCvAim. Positive
    accelerates in the selected gear, including reverse.
    """
    speed = float(state.speed_mps)
    kph = abs(speed) * 3.6
    wheels = tuple(abs(v) * 3.6 for v in state.wheel_speed_mps)
    if abs(speed) < 0.02:
        direction = DIR_STOP
    elif state.gear == "R" or speed < 0:
        direction = DIR_BACKWARD
    else:
        direction = DIR_FORWARD
    model_yaw_rate = getattr(state, "yaw_rate_rps", None)
    yaw_rate = 0.0 if model_yaw_rate is None else math.degrees(model_yaw_rate)
    long_accel = 0.0
    steer_rate = 0.0
    if prev is not None and dt > 0:
        dyaw = (state.yaw - prev.yaw + math.pi) % (2.0 * math.pi) - math.pi
        if model_yaw_rate is None:
            yaw_rate = math.degrees(dyaw / dt)
        long_accel = (state.speed_mps - prev.speed_mps) / dt
        steer_rate = (state.steer_wheel_deg - prev.steer_wheel_deg) / dt
    brake_pressed = bool(getattr(command, "brake_pressed", False)) if command is not None else False
    fields = dict(
        gear=GEAR_FROM_SIM[state.gear],
        speed_kph=kph,
        wheel_speed_kph=wheels,
        direction=direction,
        pulses=list(state.wheel_pulse_count),
        standstill=abs(speed) < 0.05,
        # IBRS_StandStillConfirmed 1 = pressure built. MCU will not release
        # the hold or command drive-off until this is 1 while standstill is 1.
        standstill_confirmed=abs(speed) < 0.05,
        steer_deg=float(state.steer_wheel_deg),
        steer_rate_dps=steer_rate,
        long_accel=long_accel,
        yaw_rate_dps=yaw_rate,
        accel_target=0.0 if command is None else float(command.accel),
        brake_pressed=brake_pressed,
        # iBRS pedal position is 0.4 %/LSB. Hold maps to a clearly pressed pedal.
        brake_pos=40.0 if brake_pressed else 0.0,
    )
    return fields


def _direction(state):
    if state.direction:
        return state.direction
    if state.gear == GEAR_R:
        return DIR_BACKWARD
    if state.gear == GEAR_D:
        return DIR_FORWARD
    return DIR_STOP


def handshake_status(request):
    """J01 actuator answer to an ADAS_07 handshake request.

    EPS_PrkgCtrlSts / VCU_APAVLCRspSt are not a copy of the request:
    Not_Request -> available (0), Ready_Request -> ready (1),
    Active_Request -> active (2). Echoing 1 would report temporary failure.
    """
    if request in (1, 2):
        return request
    return 0


def build_payload(name, state, counter, adas):
    if isinstance(adas, Adas07):
        adas = argparse.Namespace(**adas.view())
    if state.auto_handshake and adas is not None and not adas.stale and adas.crc_ok:
        eps_ctrl = handshake_status(adas.eps_active)
        vlc_rsp = handshake_status(adas.vlc_req)
        # VCU_stAPAVLC: 2 APA_control_Ready, 3 APA_control_Active. 4 is takeover.
        vlc_st = {1: 2, 2: 3}.get(adas.vlc_req, 0)
        # Keep the displayed chassis state equal to the bytes just transmitted.
        if hasattr(state, "update"):
            state.update(eps_ctrl=eps_ctrl, vlc_rsp=vlc_rsp, vlc_st=vlc_st)
    else:
        eps_ctrl = state.eps_ctrl
        vlc_rsp = state.vlc_rsp
        vlc_st = state.vlc_st
    direction = _direction(state)
    wheels = state.wheel_speed_kph
    if all(v == 0.0 for v in wheels) and state.speed_kph:
        wheels = (state.speed_kph,) * 4
    pulses = [p & 0xFFF for p in state.pulses]
    # DBC: 0 Not_Pressed, 1 Pressed_QM, 2 Pressed_D. MCU treats both 1 and 2
    # as the driver pedal, so a released pedal must be 0.
    brake = 1 if state.brake_pressed else 0

    if name == "iBRS_01":
        buf = bytearray(16)
        put_bits(buf, 12, 2, 1)  # VehSpdVD: 1 valid
        put_bits(buf, 14, 2, 1 if state.standstill_confirmed else 0)
        for start, speed in zip((16, 32, 80, 96), wheels):
            put_bits(buf, start, 15, raw_linear(abs(speed), 0.01, 0, 15))
        put_bits(buf, 48, 15, raw_linear(abs(state.speed_kph), 0.01, 0, 15))
        return buf
    if name == "iBRS_03":
        buf = bytearray(16)
        put_bits(buf, 46, 2, 1 if state.standstill else 0)
        put_bits(buf, 56, 2, brake)
        put_bits(buf, 62, 1, 1)  # APA available
        put_bits(buf, 73, 8, raw_linear(state.brake_pos, 0.4, 0, 8))
        return buf
    if name == "iBRS_04":
        buf = bytearray(16)
        for start, pulse in zip((16, 28, 40, 52), pulses):
            put_bits(buf, start, 12, pulse)
        return buf
    if name == "iBRS_05":
        buf = bytearray(8)
        for start in (16, 19, 24, 27):
            put_bits(buf, start, 2, direction)
        put_bits(buf, 32, 2, direction)
        return buf
    if name == "iBRS_11":
        buf = bytearray(8)
        put_bits(buf, 16, 2, state.epb & 3)
        return buf
    if name == "EPS_Fr01":
        buf = bytearray(8)
        put_bits(buf, 18, 1, 0)  # EPS_SteeringAngleValid: 0 valid
        put_bits(buf, 19, 1, 0)  # EPS_SteerAngleSpdValid: 0 valid
        put_bits(buf, 32, 14, raw_linear(state.steer_deg, 0.1, -780, 14))
        put_bits(buf, 48, 14, raw_linear(state.steer_rate_dps, 0.125, -1024, 14))
        # DBC places torsion torque past this 8-byte frame. The MCU snapshot
        # still reads it, so pack the 0 Nm raw at bit 20.
        put_bits(buf, 20, 11, raw_linear(state.torsion_nm, 0.01, -10.24, 11))
        return buf
    if name == "EPS_Fr02":
        buf = bytearray(12)
        put_bits(buf, 12, 3, eps_ctrl & 7)
        put_bits(buf, 16, 11, raw_linear(state.eps_motor_nm, 0.01, -8, 11))
        put_bits(buf, 27, 1, 0)
        put_bits(buf, 48, 8, raw_linear(state.eps_assist_pct, 0.5, 0, 8))
        return buf
    if name == "VCU_VehInfo":
        buf = bytearray(8)
        put_bits(buf, 19, 3, state.pt_st & 7)
        put_bits(buf, 24, 8, raw_linear(state.accel_pedal, 0.4, 0, 8))
        put_bits(buf, 32, 4, state.gear & 0xF)
        put_bits(buf, 36, 2, 0)
        return buf
    if name == "VCU_ADASInfo":
        buf = bytearray(8)
        put_bits(buf, 16, 3, vlc_rsp & 7)
        put_bits(buf, 22, 3, vlc_st & 7)
        put_bits(buf, 32, 8, raw_linear(state.slope_deg, 1, -90, 8))
        put_bits(buf, 40, 8, raw_linear(state.accel_target, 0.05, -7, 8))
        return buf
    if name == "Airbag_02":
        buf = bytearray(20)
        put_bits(buf, 40, 16, raw_linear(state.long_accel, 0.005, -163.84, 16))
        put_bits(buf, 104, 16, raw_linear(state.yaw_rate_dps, 0.01, -327.68, 16))
        return buf
    if name == "BDU_0":
        buf = bytearray(4)
        put_bits(buf, 16, 3, state.power_mode & 7)
        return buf
    if name == "LDC_01":
        buf = bytearray(16)
        for start in (24, 34):
            put_bits(buf, start, 10, raw_linear(0, 0.1, -30, 10))
        return buf
    if name == "RDC_01":
        buf = bytearray(8)
        for start in (8, 18):
            put_bits(buf, start, 10, raw_linear(0, 0.1, -30, 10))
        return buf
    if name == "Airbag_01":
        return bytearray(8)
    if name == "BDU_60":
        buf = bytearray(12)
        put_bits(buf, 31, 1, 1)
        return buf
    if name == "BDU_DspyInfo02":
        return bytearray(8)
    if name == "VCU_Chrg":
        return bytearray(12)
    if name == "BDU_73":
        # Driver is not touching the stalk. 4 = not_verified; anything else
        # (including the 0 Init default) is a gear intervention once APA is active.
        # VCU_TqActFA raw 16382 is 0 Nm (factor 1, offset -16382).
        buf = bytearray(16)
        put_bits(buf, 44, 15, 16382)
        put_bits(buf, 59, 4, 4)
        return buf
    raise KeyError(name)


def seal(buf, counter, table, counter_bit=8):
    if counter_bit is not None:
        put_bits(buf, counter_bit, 4, counter & 0x0F)
    if table is not None:
        put_bits(buf, 0, 8, crc8(buf, counter, table))
    return buf


class Adas07:
    def __init__(self):
        self._lock = threading.Lock()
        self.stamp = 0.0
        self.counter = None
        self.crc_ok = False
        self.gear_req = 0
        self.steer_deg = 0.0
        self.eps_active = 0
        self.accel = 0.0
        self.vlc_req = 0
        self.drive_off = 0
        self.standstill_req = 0
        self.decel_to_stop = 0
        self.epb_req = 0
        self.emergency = 0
        self.mode = 0
        self.raw = b""

    @property
    def stale(self):
        return self.view()["stale"]

    def parse(self, payload):
        if len(payload) != 8:
            return
        counter = get_bits(payload, 8, 4)
        expect = crc8(payload, counter, ADAS_07_CRC)
        with self._lock:
            self.stamp = time.monotonic()
            self.counter = counter
            self.crc_ok = payload[0] == expect
            self.gear_req = get_bits(payload, 53, 3)
            self.steer_deg = get_bits(payload, 17, 14) * 0.1 - 580
            self.eps_active = get_bits(payload, 31, 2)
            self.accel = get_bits(payload, 40, 8) * 0.05 - 7
            self.vlc_req = get_bits(payload, 36, 2)
            self.drive_off = get_bits(payload, 48, 1)
            self.standstill_req = get_bits(payload, 49, 1)
            self.decel_to_stop = get_bits(payload, 50, 1)
            self.epb_req = get_bits(payload, 51, 2)
            self.emergency = get_bits(payload, 15, 2)
            self.mode = get_bits(payload, 12, 3)
            self.raw = bytes(payload[:8])

    def view(self):
        with self._lock:
            view = {k: v for k, v in self.__dict__.items() if k != "_lock"}
            view["stale"] = self.stamp == 0.0 or time.monotonic() - self.stamp > 0.1
            return view


def controls_from_adas(adas, current_gear):
    """ADAS_07 to VehicleController inputs. None when the axes are not active."""
    adas = argparse.Namespace(**adas.view())
    if adas.stale or not adas.crc_ok:
        return None
    if adas.vlc_req != 2 and adas.eps_active != 2:
        return None
    gear = GEAR_TO_SIM.get(adas.gear_req, current_gear) if adas.gear_req else current_gear
    steer = adas.steer_deg if adas.eps_active == 2 else 0.0
    # ADAS_APAVLCAxvCvAim is body acceleration. J01 already negates it in R,
    # so a reverse drive-off arrives positive. The vehicle pedal uses that
    # sign directly: positive increases speed in the selected gear.
    accel = adas.accel if adas.vlc_req == 2 else 0.0
    if adas.emergency or adas.decel_to_stop or adas.standstill_req:
        accel = -4.0
    return steer, accel, gear


def can_frame(can_id, payload):
    flags = CAN_FD_FDF | CAN_FD_BRS
    return struct.pack("=IBB2x64s", can_id, len(payload), flags, bytes(payload).ljust(64, b"\0"))


def open_can(iface):
    sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FD_FRAMES, 1)
    # Match the 11-bit ID only. Masking CAN_ERR_FLAG drops every standard
    # frame on this kernel, so the window stays at "no ADAS_07".
    sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER,
                   struct.pack("=II", ADAS_07_ID, 0x7FF))
    sock.bind((iface,))
    sock.settimeout(0.05)
    return sock


def receive_loop(sock, adas, stop):
    while not stop.is_set():
        try:
            frame = sock.recv(CAN_FRAME_LEN)
        except socket.timeout:
            continue
        except OSError:
            break
        if len(frame) < 8:
            continue
        can_id, length = struct.unpack_from("=IB", frame)
        if can_id == ADAS_07_ID and length == 8:
            adas.parse(frame[8:8 + length])


def encode_frames(state, adas, counters, frames):
    """Pack a batch from one chassis and command snapshot."""
    view = argparse.Namespace(**state.snapshot())
    command = argparse.Namespace(**adas.view())
    due = []
    for name, can_id, dlc, _period, table in frames:
        # These DBC messages have no counter: bit 8 is signal data.
        counter_bit = None if name in ("RDC_01", "BDU_DspyInfo02", "BDU_73") else 8
        payload = seal(build_payload(name, view, counters[name], command),
                       counters[name], table, counter_bit=counter_bit)
        if len(payload) != dlc:
            raise RuntimeError("%s dlc %s != %s" % (name, len(payload), dlc))
        due.append((name, can_id, payload))
        counters[name] = (counters[name] + 1) & 0x0F
    return due


def encode_due(state, adas, counters, tick_ms=0):
    """Payloads whose period lands on this 10 ms tick; tick zero sends all."""
    return encode_frames(state, adas, counters,
                         (frame for frame in FRAMES if tick_ms % frame[3] == 0))


def set_tx_thread_nice(pid, tid, nice):
    """Raise only the CAN TX thread; RTKit supports an unprivileged desktop."""
    try:
        os.setpriority(os.PRIO_PROCESS, tid, nice)
    except PermissionError:
        result = subprocess.run([
            "gdbus", "call", "--system", "--dest", "org.freedesktop.RealtimeKit1",
            "--object-path", "/org/freedesktop/RealtimeKit1", "--method",
            "org.freedesktop.RealtimeKit1.MakeThreadHighPriorityWithPID",
            "--", str(pid), str(tid), str(nice),
        ], capture_output=True, text=True, timeout=3)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or "RTKit rejected CAN TX priority")
    actual = os.getpriority(os.PRIO_PROCESS, tid)
    if actual != nice:
        raise RuntimeError("CAN TX nice is %d, requested %d" % (actual, nice))
    return actual


class CanLink:
    """Chassis TX and ADAS_07 RX sharing the window's vehicle.

    A state provider can advance the vehicle on the same 10 ms clock as TX.
    Legacy push callers publish snapshots. dry_run never opens a socket.
    """

    def __init__(self, iface=None, dry_run=True):
        self.chassis = ChassisState()
        self.adas = Adas07()
        self.counters = {name: 0 for name, *_ in FRAMES}
        self.sent = {name: 0 for name, *_ in FRAMES}
        self.prev = None
        self._prev_t = None
        self.dry_run = bool(dry_run) or not iface
        self.iface = None if self.dry_run else iface
        self.sock = None
        self.error = None
        self._stop = threading.Event()
        self._rx = None
        self._tx = None
        self._state_provider = None
        self._trace = None
        self._trace_writer = None
        # Preserve DBC periods; prioritize the shared sensor/physics TX thread.
        self.tx_nice = int(os.environ.get("TX2_CAN_TX_NICE", "0" if self.dry_run else "-10"))
        if not -20 <= self.tx_nice <= 19:
            raise ValueError("TX2_CAN_TX_NICE must be between -20 and 19")
        self._tx_priority = None
        if not self.dry_run:
            self.sock = open_can(iface)
            self._rx = threading.Thread(
                target=receive_loop, args=(self.sock, self.adas, self._stop), daemon=True,
            )
            self._rx.start()

    def enable_trace(self, path):
        """Record actual sensor-frame send times and bytes before starting TX."""
        if self._tx is not None:
            raise RuntimeError("enable CAN trace before starting TX")
        self._trace = Path(path).open("w", newline="", buffering=65536)
        self._trace_writer = csv.writer(self._trace)
        self._trace_writer.writerow(("tx_mono", "tx_wall", "dry_run", "name", "can_id", "payload_hex"))

    def start(self, state_provider=None):
        """Start periodic TX after the caller has supplied its first state."""
        if self._stop.is_set():
            raise RuntimeError("CAN link is closed")
        if state_provider is not None:
            if self._tx is not None:
                raise RuntimeError("install state provider before starting TX")
            self._state_provider = state_provider
        if self._tx is None:
            if not self.dry_run:
                # A high-priority TX thread still waits for Python's GIL.
                sys.setswitchinterval(min(sys.getswitchinterval(), 0.001))
            self._tx = threading.Thread(target=self._transmit, name="can-tx", daemon=True)
            self._tx.start()

    def _transmit(self):
        tid = threading.get_native_id()
        priority = dict(pid=os.getpid(), tid=tid, requested_nice=self.tx_nice, error=None,
                        gil_switch_interval_s=sys.getswitchinterval())
        try:
            priority["nice"] = set_tx_thread_nice(os.getpid(), tid, self.tx_nice)
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            priority.update(nice=os.getpriority(os.PRIO_PROCESS, tid), error=str(exc))
        self._tx_priority = priority
        if self._trace is not None:
            Path(self._trace.name).with_name("can_thread.json").write_text(
                json.dumps(priority, indent=2) + "\n")
        if not self.dry_run:
            print("CAN TX thread: " + json.dumps(priority), file=sys.stderr, flush=True)
        # Queue odometry sensors before other frames from the same snapshot.
        sensor_order = {"Airbag_02": 0, "iBRS_04": 1, "iBRS_01": 2, "iBRS_05": 3}
        tx_frames = sorted(FRAMES, key=lambda frame: sensor_order.get(frame[0], 4))
        deadlines = {name: time.monotonic() for name, *_ in FRAMES}
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                frames = [frame for frame in tx_frames if now >= deadlines[frame[0]]]
                if self._state_provider is not None and any(f[0] == "Airbag_02" for f in frames):
                    vehicle, command, dt = self._state_provider(now)
                    self.push(vehicle, command, dt=dt)
                for name, can_id, payload in encode_frames(
                    self.chassis, self.adas, self.counters, frames,
                ):
                    if self.sock is not None:
                        self.sock.send(can_frame(can_id, payload))
                    self.sent[name] += 1
                    if self._trace_writer is not None and name in ("Airbag_02", "iBRS_04", "iBRS_05"):
                        self._trace_writer.writerow(("%.9f" % time.monotonic(), "%.6f" % time.time(),
                                                     int(self.dry_run), name, hex(can_id), payload.hex()))
                # Skip missed periods instead of replaying bursts after a stall.
                finished = time.monotonic()
                for name, _can_id, _dlc, period_ms, _table in frames:
                    period = period_ms / 1000.0
                    skipped = int((finished - deadlines[name]) / period) + 1
                    deadlines[name] += skipped * period
                self._stop.wait(max(0, min(deadlines.values()) - time.monotonic()))
        except Exception as exc:
            if not self._stop.is_set():
                self.error = str(exc)
                self._stop.set()

    def close(self):
        self._stop.set()
        if self._tx is not None:
            self._tx.join(timeout=1.0)
        sock, self.sock = self.sock, None
        if sock is not None:
            sock.close()
        if self._rx is not None:
            self._rx.join(timeout=0.2)
        if self._trace is not None:
            self._trace.close()
            self._trace = self._trace_writer = None

    def inject(self, payload):
        """Parse one ADAS_07 the way the socket thread would."""
        self.adas.parse(payload)

    def controls(self, gear):
        return controls_from_adas(self.adas, gear)

    def sync(self, vehicle):
        """Forget the previous sample after a pose reset."""
        self.prev = vehicle
        self._prev_t = time.monotonic()

    def push(self, vehicle, command=None, dt=None):
        """Publish the video vehicle's latest state; TX runs independently.

        A missing CAN peer fills the socket queue and times out. Keep the
        window and video running; status() surfaces the error.
        """
        if self.error is not None:
            return
        now = time.monotonic()
        if dt is None:
            dt = 0.01 if self._prev_t is None else max(1e-3, now - self._prev_t)
        self._prev_t = now
        self.chassis.update(**chassis_from_vehicle(vehicle, self.prev, dt, command))
        self.prev = vehicle
        self.start()

    def status(self):
        got = self.adas.view()
        snap = self.chassis.snapshot()
        return {
            "dry": self.dry_run,
            "iface": self.iface,
            "error": self.error,
            "sent": sum(self.sent.values()),
            "tx_thread": self._tx_priority,
            "stamp": got["stamp"],
            "crc": got["crc_ok"],
            "stale": got["stale"],
            "gear": got["gear_req"],
            "steer": got["steer_deg"],
            "acc": got["accel"],
            "vlc": got["vlc_req"],
            "eps": got["eps_active"],
            "emergency": got["emergency"],
            "standstill_req": got["standstill_req"],
            "decel_to_stop": got["decel_to_stop"],
            "drive_off": got["drive_off"],
            "epb_req": got["epb_req"],
            "mode": got["mode"],
            "counter": got["counter"],
            "tx_gear": snap["gear"],
            "tx_brake": 1 if snap["brake_pressed"] else 0,
            "tx_brake_pos": snap["brake_pos"],
            "tx_vlc": snap["vlc_rsp"],
            "tx_vlc_st": snap["vlc_st"],
            "tx_eps": snap["eps_ctrl"],
        }


def run(iface, state, dry_run, status_s):
    if dry_run:
        counters = {name: 0 for name, *_ in FRAMES}
        for name, can_id, payload in encode_due(state, Adas07(), counters):
            print("%s %03X %s" % (name, can_id, payload.hex()))
        return
    link = CanLink(iface, dry_run=False)
    link.chassis = state
    try:
        link.start()
        while not link._stop.wait(status_s if status_s > 0 else 0.1):
            if status_s > 0:
                view = argparse.Namespace(**state.snapshot())
                got = link.adas.view()
                print(json.dumps({
                    "gear": view.gear, "kph": view.speed_kph, "steer": view.steer_deg,
                    "adas_crc": got["crc_ok"], "adas_stale": got["stale"],
                    "adas_gear": got["gear_req"], "adas_acc": round(got["accel"], 3),
                    "adas_steer": round(got["steer_deg"], 1),
                    "vlc": got["vlc_req"], "eps": got["eps_active"],
                }, ensure_ascii=False), flush=True)
        if link.error is not None:
            raise RuntimeError("CAN TX failed: " + link.error)
    except KeyboardInterrupt:
        pass
    finally:
        link.close()


def simulate(seconds, steer_deg, accel, gear):
    """Step the J01 model and pack the frames it would send. No CAN socket."""
    from vehicle import VehicleController

    ctrl = VehicleController(ROOT / "configs/j01_vehicle.yaml")
    ctrl.reset(0.5, 1.5, 0.0, 0.0)
    ctrl.set_controls(steer_deg, accel, gear)
    chassis = ChassisState()
    adas = Adas07()
    counters = {name: 0 for name, *_ in FRAMES}
    prev = None
    dt = ctrl.dt_s
    steps = max(1, int(round(seconds / dt)))
    sent = {name: 0 for name, *_ in FRAMES}
    elapsed_ms = 0
    for _ in range(steps):
        state = ctrl.step(dt)
        chassis.update(**chassis_from_vehicle(state, prev, dt))
        prev = state
        elapsed_ms += int(round(dt * 1000))
        # Control dt is 20 ms; 10 ms frames are emitted on both edges.
        for stamp in range(elapsed_ms - int(round(dt * 1000)) + 10, elapsed_ms + 1, 10):
            for name, _can_id, _payload in encode_due(chassis, adas, counters, stamp):
                sent[name] += 1
    return state, chassis.snapshot(), sent


def self_check():
    state = ChassisState()
    state.speed_kph = 3.6
    state.steer_deg = 0
    state.gear = GEAR_D
    adas = Adas07()
    view = argparse.Namespace(**state.snapshot())
    for name, _can_id, dlc, _period, table in FRAMES:
        payload = seal(build_payload(name, view, 1, adas), 1, table)
        assert len(payload) == dlc, name
        if table is not None:
            assert payload[0] == crc8(payload, 1, table), name
    zero = seal(build_payload("EPS_Fr01", view, 0, adas), 0, None)
    assert get_bits(zero, 32, 14) == 7800
    assert get_bits(zero, 48, 14) == 8192
    speed = build_payload("iBRS_01", view, 0, adas)
    assert get_bits(speed, 48, 15) == 360  # MCU confirmed 3.6 km/h
    released = build_payload("iBRS_03", view, 0, adas)
    assert get_bits(released, 56, 2) == 0
    moving = build_payload("iBRS_01", view, 0, adas)
    assert get_bits(moving, 14, 2) == 0
    view.standstill = True
    view.standstill_confirmed = True
    stopped = build_payload("iBRS_01", view, 0, adas)
    assert get_bits(stopped, 14, 2) == 1
    view.standstill = False
    view.standstill_confirmed = False
    view.brake_pressed = True
    view.brake_pos = 40.0
    pressed = build_payload("iBRS_03", view, 0, adas)
    assert get_bits(pressed, 56, 2) == 1
    view.brake_pressed = False
    view.brake_pos = 0.0
    view.steer_deg = 123.4
    angle = build_payload("EPS_Fr01", view, 0, adas)
    assert get_bits(angle, 32, 14) == 9034  # MCU confirmed +123.4 degrees
    gear = seal(build_payload("VCU_VehInfo", view, 0, adas), 0, None)
    assert get_bits(gear, 32, 4) == GEAR_D
    assert get_bits(gear, 19, 3) == 3
    pulses = seal(build_payload("iBRS_04", argparse.Namespace(
        **dict(state.snapshot(), pulses=[1, 4097, 100, 4095])), 0, adas), 0, None)
    assert get_bits(pulses, 16, 12) == 1
    assert get_bits(pulses, 28, 12) == 1  # 4097 wraps
    sample = bytearray.fromhex("000011FE01000000")
    assert crc8(sample, 0, FRAMES[7][4]) == 0xE7
    shift = None
    for counter in range(16):
        counts = {name: counter for name, *_ in FRAMES}
        encoded = {name: payload for name, _, payload in
                   encode_frames(state, adas, counts, FRAMES)}
        # Idle mirror availability must not toggle as other frames count.
        assert encoded["BDU_DspyInfo02"] == bytes(8)
        # Shift intent must stay not_verified; the frame has no rolling counter.
        assert get_bits(encoded["BDU_73"], 59, 4) == 4
        assert get_bits(encoded["BDU_73"], 44, 15) == 16382
        assert shift is None or encoded["BDU_73"] == shift
        shift = encoded["BDU_73"]
        for name, starts in (("LDC_01", (24, 34)), ("RDC_01", (8, 18))):
            assert all(get_bits(encoded[name], start, 10) == 300 for start in starts)
        assert get_bits(encoded["BDU_60"], 8, 4) == counter
        for name, _id, _dlc, _period, table in FRAMES:
            if table is not None:
                assert encoded[name][0] == crc8(encoded[name], counter, table), name
    vehicle, snap, sent = simulate(1.0, 90.0, 1.0, "D")
    assert vehicle.gear == "D" and vehicle.speed_mps > 0.5
    assert snap["pulses"][3] > snap["pulses"][2]  # right rear travels farther on a left turn
    assert snap["steer_deg"] == vehicle.steer_wheel_deg
    assert sent["iBRS_01"] > sent["iBRS_03"] > 0
    # One meter at the MCU R17 pulse distance.
    from vehicle.kinematics import KinematicState, bicycle_step
    from vehicle.config import load_vehicle_config
    params = load_vehicle_config(ROOT / "configs/j01_vehicle.yaml").vehicle
    ks = KinematicState(vx=1.0, gear="D")
    bicycle_step(ks, 0.0, 0.0, "D", 1.0, params)
    assert abs(ks.wheel_pulse_count[2] - round(1.0 / params.pulse_distance_m)) <= 1
    command = bytearray(8)
    put_bits(command, 53, 3, GEAR_R)
    put_bits(command, 17, 14, 5800)  # 0 deg target
    put_bits(command, 31, 2, 2)
    put_bits(command, 36, 2, 2)
    put_bits(command, 40, 8, raw_linear(-0.5, 0.05, -7, 8))
    put_bits(command, 8, 4, 1)
    put_bits(command, 0, 8, crc8(command, 1, ADAS_07_CRC))
    request = Adas07()
    request.parse(command)
    steer, pedal, gear_name = controls_from_adas(request, "D")
    assert gear_name == "R" and abs(pedal + 0.5) < 1e-6 and steer == 0.0
    response = build_payload("VCU_ADASInfo", view, 0, request)
    assert get_bits(response, 16, 3) == 2 and get_bits(response, 22, 3) == 3
    eps = build_payload("EPS_Fr02", view, 0, request)
    assert get_bits(eps, 12, 3) == 2
    # Ready is a distinct answer. Echoing the request value 1 is temporary failure.
    ready = bytearray(command)
    put_bits(ready, 31, 2, 1)
    put_bits(ready, 36, 2, 1)
    put_bits(ready, 0, 8, crc8(ready, 1, ADAS_07_CRC))
    request.parse(ready)
    assert handshake_status(1) == 1 and handshake_status(0) == 0
    assert get_bits(build_payload("VCU_ADASInfo", view, 0, request), 16, 3) == 1
    assert get_bits(build_payload("VCU_ADASInfo", view, 0, request), 22, 3) == 2
    assert get_bits(build_payload("EPS_Fr02", view, 0, request), 12, 3) == 1
    assert controls_from_adas(request, "D") is None
    angle = build_payload("EPS_Fr01", view, 0, request)
    assert get_bits(angle, 18, 1) == 0 and get_bits(angle, 19, 1) == 0
    damaged = bytearray(command)
    damaged[0] ^= 1
    request.parse(damaged)
    assert controls_from_adas(request, "D") is None
    response = build_payload("VCU_ADASInfo", view, 0, request)
    assert get_bits(response, 16, 3) == 0 and get_bits(response, 22, 3) == 0
    request.parse(command)
    request.stamp = time.monotonic() - 0.2
    assert controls_from_adas(request, "D") is None
    assert get_bits(build_payload("EPS_Fr02", view, 0, request), 12, 3) == 0
    # Exercise the SocketCAN receive boundary without opening a real bus.
    rx_stop = threading.Event()
    incoming = [can_frame(ADAS_07_ID | flag, command)
                for flag in (0x40000000, 0x80000000, 0x20000000)]
    incoming += [can_frame(ADAS_07_ID, command[:7])]

    class Inbox:
        def recv(self, size):
            assert size == CAN_FRAME_LEN
            if incoming:
                return incoming.pop(0)
            rx_stop.set()
            raise socket.timeout()

    received = Adas07()
    receive_loop(Inbox(), received, rx_stop)
    assert received.stamp == 0.0
    rx_stop.clear()
    incoming.append(can_frame(ADAS_07_ID, command))
    receive_loop(Inbox(), received, rx_stop)
    assert controls_from_adas(received, "D") == (0.0, -0.5, "R")
    packet = can_frame(ADAS_07_ID, command)
    assert len(packet) == 72 and packet[5] == CAN_FD_FDF | CAN_FD_BRS
    # A full second contains each frame's nominal number of slots, including tick zero.
    counts = {name: 0 for name, *_ in FRAMES}
    counters = dict(counts)
    for tick_ms in range(0, 1000, 10):
        for name, _can_id, _payload in encode_due(state, adas, counters, tick_ms):
            counts[name] += 1
    assert counts == {name: 1000 // period for name, _id, _dlc, period, _crc in FRAMES}
    # Same order as the window: Active ADAS replaces keys, then chassis TX.
    from vehicle import VehicleController

    ctrl = VehicleController(ROOT / "configs/j01_vehicle.yaml")
    ctrl.reset(0.5, 1.5, 0.0, 0.0)
    link = CanLink(dry_run=True)
    try:
        assert link.sock is None and link.controls("D") is None
        keys = (12.0, 0.2, "D")
        chosen = link.controls(ctrl.get_state().gear) or keys
        assert chosen == keys
        ctrl.set_controls(*chosen)
        stepped = ctrl.step(ctrl.dt_s)
        link.push(stepped, argparse.Namespace(accel=chosen[1]))
        # Deliberately stop publishing poses as a slow video renderer would.
        time.sleep(0.12)
        assert link.sent["iBRS_01"] >= 3
        assert link.sent["iBRS_01"] > link.sent["iBRS_03"] > 0
        assert link.sent["BDU_60"] == 1
        link.inject(command)
        chosen = link.controls("D")
        assert chosen is not None and chosen[2] == "R" and abs(chosen[1] + 0.5) < 1e-6
        assert chosen != keys
        # A positive body accel is the J01 reverse drive-off request.
        put_bits(command, 40, 8, raw_linear(0.5, 0.05, -7, 8))
        put_bits(command, 0, 8, crc8(command, 1, ADAS_07_CRC))
        link.inject(command)
        chosen = link.controls("D")
        assert chosen == (0.0, 0.5, "R")
        # Dry-run does not transmit, so update the displayed reply explicitly.
        build_payload("VCU_ADASInfo", link.chassis, 0, link.adas)
        build_payload("EPS_Fr02", link.chassis, 0, link.adas)
        reported = link.status()
        assert reported["tx_vlc"] == 2 and reported["tx_vlc_st"] == 3
        assert reported["tx_eps"] == 2
        ctrl.set_controls(*chosen)
        stepped = ctrl.step(0.2)
        link.push(stepped, argparse.Namespace(accel=chosen[1]))
        assert stepped.gear == "R" and stepped.speed_mps < 0
        assert abs(link.chassis.snapshot()["accel_target"] - chosen[1]) < 1e-9
        # Emergency reverse braking must report the acceleration actually applied.
        emergency = bytearray(command)
        put_bits(emergency, 49, 1, 1)
        put_bits(emergency, 0, 8, crc8(emergency, 1, ADAS_07_CRC))
        link.inject(emergency)
        chosen = link.controls("R")
        assert chosen == (0.0, -4.0, "R")
        ctrl.set_controls(*chosen)
        stepped = ctrl.step(ctrl.dt_s)
        link.push(stepped, argparse.Namespace(accel=-chosen[1]))
        assert link.chassis.snapshot()["accel_target"] == 4.0
    finally:
        link.close()
    assert not link._tx.is_alive()


def main():
    parser = argparse.ArgumentParser(description="APA chassis CAN bridge")
    parser.add_argument("--iface", default="can0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", type=float, default=1.0)
    parser.add_argument("--gear", type=int, choices=tuple(GEAR_TO_SIM), default=GEAR_D)
    parser.add_argument("--speed-kph", type=float, default=0.0)
    parser.add_argument("--steer-deg", type=float, default=0.0)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--sim", action="store_true", help="step the local model and print one status")
    parser.add_argument("--seconds", type=float, default=1.0)
    args = parser.parse_args()
    if not 0.0 <= args.speed_kph <= 327.67:
        parser.error("--speed-kph must be between 0 and 327.67")
    if not -500.0 <= args.steer_deg <= 500.0:
        parser.error("--steer-deg must be between -500 and 500")
    if not math.isfinite(args.status) or args.status < 0:
        parser.error("--status must be finite and nonnegative")
    self_check()
    if args.check:
        print("crc, packing, ADAS validity, periodic TX, and vehicle feedback ok")
        return
    if args.sim:
        vehicle, snap, sent = simulate(args.seconds, args.steer_deg, 0.5, GEAR_TO_SIM[args.gear])
        print(json.dumps({
            "speed_mps": round(vehicle.speed_mps, 3),
            "steer_deg": round(vehicle.steer_wheel_deg, 2),
            "pulses": snap["pulses"],
            "frames": sent,
        }))
        return
    state = ChassisState()
    state.gear = args.gear
    state.speed_kph = args.speed_kph
    state.steer_deg = args.steer_deg
    state.standstill = args.speed_kph == 0.0
    state.direction = (DIR_STOP if state.standstill else
                       DIR_BACKWARD if args.gear == GEAR_R else DIR_FORWARD)
    run(args.iface, state, args.dry_run, 0 if args.dry_run else args.status)


if __name__ == "__main__":
    main()
