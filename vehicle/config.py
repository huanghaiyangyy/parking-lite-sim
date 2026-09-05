"""Load vehicle control YAML into dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Union

import yaml

PathLike = Union[str, Path]


@dataclass(frozen=True)
class VehicleParams:
    wheelbase_m: float = 2.7
    track_width_m: float = 1.55
    steering_ratio: float = 16.0
    max_steer_wheel_deg: float = 500.0
    max_road_wheel_deg: float = 35.0
    max_accel_mps2: float = 2.0
    max_decel_mps2: float = 4.0
    max_speed_mps: float = 5.0
    wheel_radius_m: float = 0.33
    pulse_ticks_per_wheel_rev: int = 96
    gear_map: Mapping[str, int] = field(
        default_factory=lambda: {"D": 1, "R": -1, "N": 0, "P": 0}
    )


@dataclass(frozen=True)
class ControlParams:
    dt_s: float = 0.02


@dataclass(frozen=True)
class VehicleConfig:
    vehicle: VehicleParams
    control: ControlParams
    source_path: Optional[Path] = None


def _as_float(d: Mapping, key: str, default: float) -> float:
    v = d.get(key, default)
    return float(v)


def _as_int(d: Mapping, key: str, default: int) -> int:
    v = d.get(key, default)
    return int(v)


def load_vehicle_config(path: PathLike) -> VehicleConfig:
    """Load configs/vehicle.yaml (or any path with vehicle: / control: sections)."""
    p = Path(path)
    if not p.is_file():
        root = Path(__file__).resolve().parent.parent
        alt = root / path
        if alt.is_file():
            p = alt
        else:
            raise FileNotFoundError("vehicle config not found: %s" % path)

    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("vehicle config root must be a mapping: %s" % p)

    vraw = raw.get("vehicle") or {}
    craw = raw.get("control") or {}
    if not isinstance(vraw, dict):
        raise ValueError("vehicle: must be a mapping")
    if not isinstance(craw, dict):
        raise ValueError("control: must be a mapping")

    gear_raw = vraw.get("gear_map") or {"D": 1, "R": -1, "N": 0, "P": 0}
    gear_map: Dict[str, int] = {str(k).upper(): int(v) for k, v in dict(gear_raw).items()}

    vehicle = VehicleParams(
        wheelbase_m=_as_float(vraw, "wheelbase_m", 2.7),
        track_width_m=_as_float(vraw, "track_width_m", 1.55),
        steering_ratio=_as_float(vraw, "steering_ratio", 16.0),
        max_steer_wheel_deg=_as_float(vraw, "max_steer_wheel_deg", 500.0),
        max_road_wheel_deg=_as_float(vraw, "max_road_wheel_deg", 35.0),
        max_accel_mps2=_as_float(vraw, "max_accel_mps2", 2.0),
        max_decel_mps2=_as_float(vraw, "max_decel_mps2", 4.0),
        max_speed_mps=_as_float(vraw, "max_speed_mps", 5.0),
        wheel_radius_m=_as_float(vraw, "wheel_radius_m", 0.33),
        pulse_ticks_per_wheel_rev=_as_int(vraw, "pulse_ticks_per_wheel_rev", 96),
        gear_map=gear_map,
    )
    control = ControlParams(dt_s=_as_float(craw, "dt_s", 0.02))
    return VehicleConfig(vehicle=vehicle, control=control, source_path=p)


__all__ = [
    "VehicleParams",
    "ControlParams",
    "VehicleConfig",
    "load_vehicle_config",
]
