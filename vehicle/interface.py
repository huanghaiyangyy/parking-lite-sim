"""Public VehicleController API — kinematic control + state feedback."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from .config import VehicleConfig, load_vehicle_config
from .kinematics import (
    WHEEL_ORDER,
    KinematicState,
    bicycle_step,
    reset_state,
    steer_wheel_to_road,
)

PathLike = Union[str, Path]


@dataclass
class VehicleState:
    """Snapshot of vehicle pose, controls, and wheel feedback.

    Units / conventions:
    - x, y: world FLU meters (+x forward, +y left)
    - yaw: radians about +z (up)
    - speed_mps: signed body longitudinal vx (D ≥ 0, R ≤ 0)
    - speed_signal_mps: published vehicle speed = abs(speed_mps) (non-negative)
    - steer / road wheel: degrees; positive road wheel = turn left (+y)
    - wheel_* arrays ordered fl, fr, rl, rr (see WHEEL_ORDER)
    - wheel_pulse_count: cumulative ABS-like ticks (int)
    - wheel_pulse_rate_hz: instantaneous pulse rate
    """

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    speed_mps: float = 0.0
    steer_wheel_deg: float = 0.0
    road_wheel_deg: float = 0.0
    gear: str = "P"
    wheel_speed_mps: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0]
    )
    wheel_pulse_count: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    wheel_pulse_rate_hz: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0]
    )
    speed_signal_mps: float = 0.0

    # Convenience named aliases (same values as wheel_speed_mps[i])
    @property
    def fl(self) -> float:
        return self.wheel_speed_mps[0]

    @property
    def fr(self) -> float:
        return self.wheel_speed_mps[1]

    @property
    def rl(self) -> float:
        return self.wheel_speed_mps[2]

    @property
    def rr(self) -> float:
        return self.wheel_speed_mps[3]


def _to_vehicle_state(ks: KinematicState) -> VehicleState:
    return VehicleState(
        x=ks.x,
        y=ks.y,
        yaw=ks.yaw,
        speed_mps=ks.vx,
        steer_wheel_deg=ks.steer_wheel_deg,
        road_wheel_deg=ks.road_wheel_deg,
        gear=ks.gear,
        wheel_speed_mps=list(ks.wheel_speed_mps),
        wheel_pulse_count=list(ks.wheel_pulse_count),
        wheel_pulse_rate_hz=list(ks.wheel_pulse_rate_hz),
        speed_signal_mps=abs(ks.vx),
    )


class VehicleController:
    """Configurable kinematic vehicle controller.

    Parameters are driven entirely by external YAML (see ``configs/vehicle.yaml``).
    Independent of the renderer — pair with ``World.set_ego_pose`` in demos.
    """

    def __init__(self, config_path: PathLike):
        self._cfg: VehicleConfig = load_vehicle_config(config_path)
        self._params = self._cfg.vehicle
        self._dt_default = float(self._cfg.control.dt_s)
        self._state = KinematicState()
        self._cmd_steer = 0.0
        self._cmd_accel = 0.0
        self._cmd_gear = "P"

    @property
    def config(self) -> VehicleConfig:
        return self._cfg

    @property
    def config_path(self) -> Optional[Path]:
        return self._cfg.source_path

    @property
    def dt_s(self) -> float:
        return self._dt_default

    def reset(self, x: float, y: float, yaw: float, speed: float = 0.0) -> VehicleState:
        """Reset pose / speed; clears pulses and sets gear to P, controls to 0."""
        reset_state(self._state, x, y, yaw, speed=speed)
        self._cmd_steer = 0.0
        self._cmd_accel = 0.0
        self._cmd_gear = "P"
        return self.get_state()

    def set_controls(
        self,
        steer_wheel_deg: float,
        accel_mps2: float,
        gear: str,
    ) -> None:
        """Latch driver inputs for subsequent ``step`` calls.

        gear in {"D","R","N","P"} (case-insensitive). Accel is longitudinal
        pedal intent (m/s^2); positive accelerates in the selected gear sense.
        """
        g = str(gear).upper()
        if g not in self._params.gear_map:
            raise ValueError(
                "unknown gear %r; expected one of %s"
                % (gear, list(self._params.gear_map))
            )
        self._cmd_steer = float(steer_wheel_deg)
        self._cmd_accel = float(accel_mps2)
        self._cmd_gear = g
        # Reflect latched steer/gear in snapshot immediately (motion applies on step).
        sw, road = steer_wheel_to_road(self._cmd_steer, self._params)
        self._state.steer_wheel_deg = sw
        self._state.road_wheel_deg = road
        self._state.gear = g

    def step(self, dt: Optional[float] = None) -> VehicleState:
        """Integrate one kinematic step; returns updated ``VehicleState``."""
        dt_use = self._dt_default if dt is None else float(dt)
        bicycle_step(
            self._state,
            self._cmd_steer,
            self._cmd_accel,
            self._cmd_gear,
            dt_use,
            self._params,
        )
        return self.get_state()

    def get_state(self) -> VehicleState:
        return _to_vehicle_state(self._state)

    def apply_to_world(self, world, z: float = 0.0) -> VehicleState:
        """Light-touch helper: push current pose into ``World.set_ego_pose``.

        Does not render; caller decides when to call ``render_*``.
        """
        st = self.get_state()
        world.set_ego_pose(st.x, st.y, z, st.yaw)
        return st


__all__ = ["VehicleController", "VehicleState", "WHEEL_ORDER"]
