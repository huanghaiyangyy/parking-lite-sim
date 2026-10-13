"""World: config-backed ego SE3 + camera render / AVM entrypoints."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import yaml

from avm.stitcher import stitch_avm
from sensors.cameras import CameraRig

PathLike = Union[str, Path]


class World:
    """Lightweight parking sim world (Phase 1: lot + posed ego + god view)."""

    def __init__(self, config: Mapping[str, Any], config_path: Optional[Path] = None):
        self.config: Dict[str, Any] = dict(config)
        self.config_path = config_path
        self.rig = CameraRig.from_config(self.config)
        ego = (self.config.get("vehicle") or {}).get("ego_pose") or {}
        self._ego = {
            "x": float(ego.get("x", 0.0)),
            "y": float(ego.get("y", 0.0)),
            "z": float(ego.get("z", 0.0)),
            "yaw": math.radians(float(ego.get("yaw_deg", 0.0))),
        }
        scene_cfg = self.config.get("scene") or {}
        self.scene = {
            "type": scene_cfg.get("type", "procedural_lot"),
            "placeholder": bool(scene_cfg.get("placeholder", False)),
            "assets": scene_cfg.get("assets_root", "assets"),
        }
        self._god_cfg = dict(self.config.get("god_view") or {})
        # Keep backend ego mesh in sync with initial pose
        self.rig.sync_ego_pose(self.ego_xyz_yaw())

    @classmethod
    def from_config(cls, path: PathLike) -> "World":
        p = Path(path)
        if not p.is_file():
            # Resolve relative to project root (parent of scene/)
            root = Path(__file__).resolve().parent.parent
            alt = root / path
            if alt.is_file():
                p = alt
            else:
                raise FileNotFoundError("config not found: %s" % path)
        with p.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError("config root must be a mapping: %s" % p)
        return cls(cfg, config_path=p)

    def set_ego_pose(self, x: float, y: float, z: float, yaw: float) -> None:
        """Set ego pose in world FLU. yaw is radians about up-z."""
        self._ego = {
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "yaw": float(yaw),
        }
        self.rig.sync_ego_pose(self.ego_xyz_yaw())

    @property
    def ego_pose(self) -> Dict[str, float]:
        return dict(self._ego)

    def ego_xyz_yaw(self) -> Tuple[float, float, float, float]:
        e = self._ego
        return (e["x"], e["y"], e["z"], e["yaw"])

    def render_cameras(self) -> Dict[str, np.ndarray]:
        """Render surround fisheyes → dict[str, HxWx3 uint8]."""
        return self.rig.render_cameras(self.ego_xyz_yaw())

    def render_camera_grid(self) -> np.ndarray:
        """Render front|left / right|rear directly on GPU for video input."""
        return self.rig.render_grid(self.ego_xyz_yaw())

    def render_god_view(
        self,
        eye_flu: Optional[Sequence[float]] = None,
        target_flu: Optional[Sequence[float]] = None,
        up_flu: Optional[Sequence[float]] = None,
        fov_y_deg: Optional[float] = None,
    ) -> np.ndarray:
        """Pinhole overview camera (world-fixed by default from config god_view)."""
        g = self._god_cfg
        eye = eye_flu if eye_flu is not None else g.get("eye_flu", [18.0, -22.0, 28.0])
        target = (
            target_flu if target_flu is not None else g.get("target_flu", [0.0, 0.0, 0.0])
        )
        up = up_flu if up_flu is not None else g.get("up_flu", [0.0, 0.0, 1.0])
        fov = float(
            fov_y_deg if fov_y_deg is not None else g.get("fov_y_deg", 55.0)
        )
        return self.rig.render_god_view(
            self.ego_xyz_yaw(),
            eye_flu=eye,
            target_flu=target,
            up_flu=up,
            fov_y_deg=fov,
        )

    def stitch_avm(self, images: Mapping[str, np.ndarray]) -> np.ndarray:
        """AVM BEV stitch (Phase 3): IPM with GT K/D/extrinsics + feather blend."""
        return stitch_avm(
            images,
            bev_cfg=self.config.get("bev"),
            cameras=self.rig.cameras,
            vehicle_cfg=self.config.get("vehicle"),
        )

    def release(self) -> None:
        self.rig.release()
