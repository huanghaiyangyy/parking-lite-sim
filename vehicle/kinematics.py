"""Bicycle-model kinematic step (no dynamics / physics engine).

Conventions (documented for callers):
- Pose in world FLU: +x forward, +y left, +z up; yaw about +z (rad).
- Body longitudinal speed ``vx`` is **signed**: gear D drives positive vx,
  gear R drives negative vx. ``speed_signal_mps`` published as ``abs(vx)``.
- Road-wheel steer δ: positive = turn left (toward +y), matching yaw_rate =
  vx * tan(δ) / L so that +δ with +vx increases yaw (CCW from above).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Tuple

from .config import VehicleParams

WHEEL_ORDER = ("fl", "fr", "rl", "rr")

_DEFAULT_GEAR_MAP: Dict[str, int] = {"D": 1, "R": -1, "N": 0, "P": 0}


@dataclass
class KinematicState:
    """Internal mutable kinematic / sensor state."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    vx: float = 0.0  # signed body longitudinal speed (m/s)
    steer_wheel_deg: float = 0.0
    road_wheel_deg: float = 0.0
    gear: str = "P"
    yaw_rate_rps: float = 0.0
    wheel_speed_mps: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0]
    )
    wheel_pulse_count: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    wheel_pulse_rate_hz: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0]
    )
    # Fractional tick accumulator so low speeds still accumulate pulses.
    _pulse_frac: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0], repr=False
    )


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def gear_sign(gear: str, gear_map: Mapping[str, int] | None = None) -> int:
    g = str(gear).upper()
    m = gear_map if gear_map is not None else _DEFAULT_GEAR_MAP
    return int(m.get(g, 0))


def steer_wheel_to_road(
    steer_wheel_deg: float, params: VehicleParams
) -> Tuple[float, float]:
    """Clamp steer wheel, convert via ratio, clamp road wheel.

    Returns (steer_wheel_deg, road_wheel_deg) after clamps. Steer wheel is
    re-mapped from the clamped road angle so the pair stays consistent.
    """
    sw = clamp(
        float(steer_wheel_deg),
        -params.max_steer_wheel_deg,
        params.max_steer_wheel_deg,
    )
    ratio = params.steering_ratio if params.steering_ratio != 0 else 1.0
    road = clamp(sw / ratio, -params.max_road_wheel_deg, params.max_road_wheel_deg)
    sw = clamp(
        road * ratio,
        -params.max_steer_wheel_deg,
        params.max_steer_wheel_deg,
    )
    return sw, road


def clamp_accel(accel_mps2: float, params: VehicleParams) -> float:
    return clamp(float(accel_mps2), -params.max_decel_mps2, params.max_accel_mps2)


def bicycle_step(
    state: KinematicState,
    steer_wheel_deg: float,
    accel_mps2: float,
    gear: str,
    dt: float,
    params: VehicleParams,
) -> KinematicState:
    """Advance kinematic state by ``dt`` seconds. Mutates and returns ``state``."""
    if dt <= 0.0:
        return state

    gear_u = str(gear).upper()
    if gear_u not in params.gear_map:
        raise ValueError(
            "unknown gear %r; expected one of %s" % (gear, list(params.gear_map))
        )

    sw, road_deg = steer_wheel_to_road(steer_wheel_deg, params)
    a_cmd = clamp_accel(accel_mps2, params)
    gsign = gear_sign(gear_u, params.gear_map)

    state.steer_wheel_deg = sw
    state.road_wheel_deg = road_deg
    state.gear = gear_u

    # N/P: ignore accel for motion; bleed speed toward 0 (simple parking model).
    if gsign == 0:
        if abs(state.vx) < 1e-6:
            state.vx = 0.0
        else:
            dv = -math.copysign(params.max_decel_mps2 * dt, state.vx)
            if abs(dv) >= abs(state.vx):
                state.vx = 0.0
            else:
                state.vx += dv
    else:
        # Positive a_cmd increases |speed| along gear direction:
        # D + accel → +vx, R + accel → -vx. Negative a_cmd brakes toward 0;
        # clamp prevents crossing into the opposite gear without shifting.
        state.vx += a_cmd * gsign * dt
        if gsign > 0:
            state.vx = clamp(state.vx, 0.0, params.max_speed_mps)
        else:
            state.vx = clamp(state.vx, -params.max_speed_mps, 0.0)

    vx = state.vx
    delta = math.radians(road_deg)
    L = params.wheelbase_m
    yaw_rate = 0.0 if abs(L) < 1e-9 else vx * math.tan(delta) / L
    state.yaw_rate_rps = yaw_rate

    state.yaw = state.yaw + yaw_rate * dt
    state.yaw = (state.yaw + math.pi) % (2.0 * math.pi) - math.pi
    state.x += vx * math.cos(state.yaw) * dt
    state.y += vx * math.sin(state.yaw) * dt

    # The average rear speed is the rear-axle speed even while turning.
    cos_d = math.cos(delta)
    if abs(cos_d) < 1e-3:
        cos_d = math.copysign(1e-3, cos_d if cos_d != 0 else 1.0)
    v_front = vx / cos_d
    half_track = 0.5 * params.track_width_m
    speeds = [
        v_front - yaw_rate * half_track,
        v_front + yaw_rate * half_track,
        vx - yaw_rate * half_track,
        vx + yaw_rate * half_track,
    ]  # fl, fr, rl, rr
    state.wheel_speed_mps = list(speeds)

    pulse_distance = float(getattr(params, "pulse_distance_m", 0.0) or 0.0)
    if pulse_distance <= 0.0:
        r = params.wheel_radius_m
        if r <= 0:
            raise ValueError("wheel_radius_m must be > 0")
        pulse_distance = 2.0 * math.pi * r / float(params.pulse_ticks_per_wheel_rev)
    rates: List[float] = []
    for i, v_w in enumerate(speeds):
        rate_hz = abs(v_w) / pulse_distance
        rates.append(rate_hz)
        state._pulse_frac[i] += rate_hz * dt
        whole = int(state._pulse_frac[i])
        if whole:
            state.wheel_pulse_count[i] += whole
            state._pulse_frac[i] -= float(whole)
    state.wheel_pulse_rate_hz = rates
    return state


def reset_state(
    state: KinematicState,
    x: float,
    y: float,
    yaw: float,
    speed: float = 0.0,
) -> KinematicState:
    state.x = float(x)
    state.y = float(y)
    state.yaw = float(yaw)
    state.vx = float(speed)
    state.steer_wheel_deg = 0.0
    state.road_wheel_deg = 0.0
    state.gear = "P"
    state.yaw_rate_rps = 0.0
    state.wheel_speed_mps = [0.0, 0.0, 0.0, 0.0]
    state.wheel_pulse_count = [0, 0, 0, 0]
    state.wheel_pulse_rate_hz = [0.0, 0.0, 0.0, 0.0]
    state._pulse_frac = [0.0, 0.0, 0.0, 0.0]
    return state


__all__ = [
    "WHEEL_ORDER",
    "KinematicState",
    "bicycle_step",
    "reset_state",
    "steer_wheel_to_road",
    "clamp_accel",
    "gear_sign",
    "clamp",
]
