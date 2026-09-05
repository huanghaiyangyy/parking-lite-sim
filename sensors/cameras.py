"""Camera models and surround CameraRig (config-driven K/D/extrinsics)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Vehicle FLU: x forward, y left, z up
# Demo/OpenGL scene: x right, y up, z forward  (gl = flu_to_gl)


def flu_to_gl(p: Sequence[float]) -> Tuple[float, float, float]:
    """Map vehicle/world FLU point to ModernGL scene coords (Y-up, +Z forward)."""
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    return (-y, z, x)


def gl_to_flu(p: Sequence[float]) -> Tuple[float, float, float]:
    gx, gy, gz = float(p[0]), float(p[1]), float(p[2])
    return (gz, -gx, gy)


def rpy_deg_to_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Rotation vehicle→camera optical frame from intrinsic XYZ (roll, pitch, yaw) degrees.

    Optical +X looks along the camera forward axis in vehicle FLU after this R.
    """
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    return Rz @ Ry @ Rx


def yaw_matrix(yaw_rad: float) -> np.ndarray:
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


@dataclass
class CameraModel:
    """Single fisheye camera: K, D, and vehicle-frame extrinsics."""

    name: str
    translation_m: np.ndarray  # (3,) vehicle FLU
    rotation_deg: np.ndarray  # (3,) roll, pitch, yaw
    K: np.ndarray  # 3x3
    D: np.ndarray  # (4,) k1..k4 OpenCV fisheye / Kannala–Brandt GT

    @classmethod
    def from_dict(cls, name: str, d: Mapping[str, Any]) -> "CameraModel":
        return cls(
            name=name,
            translation_m=np.asarray(d["translation_m"], dtype=np.float64).reshape(3),
            rotation_deg=np.asarray(d["rotation_deg"], dtype=np.float64).reshape(3),
            K=np.asarray(d["K"], dtype=np.float64).reshape(3, 3),
            D=np.asarray(d["D"], dtype=np.float64).reshape(4),
        )

    def rotation_matrix(self) -> np.ndarray:
        r, p, y = self.rotation_deg.tolist()
        return rpy_deg_to_matrix(r, p, y)

    def optical_axes_vehicle(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (forward, up) unit vectors in vehicle FLU."""
        R = self.rotation_matrix()
        forward = R @ np.array([1.0, 0.0, 0.0])
        up = R @ np.array([0.0, 0.0, 1.0])
        forward = forward / (np.linalg.norm(forward) + 1e-12)
        up = up / (np.linalg.norm(up) + 1e-12)
        return forward, up

    def look_at_vehicle(self, look_dist: float = 6.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """eye, target, up in vehicle FLU."""
        forward, up = self.optical_axes_vehicle()
        eye = self.translation_m.copy()
        target = eye + forward * look_dist
        return eye, target, up


@dataclass
class CameraRig:
    """Four surround cameras loaded from config; optional ModernGL render backend."""

    cameras: Dict[str, CameraModel]
    image_size: Tuple[int, int]  # width, height
    render_cfg: Dict[str, Any] = field(default_factory=dict)
    vehicle_cfg: Dict[str, Any] = field(default_factory=dict)
    assets_root: Optional[Path] = None
    order: Tuple[str, ...] = ("front", "rear", "left", "right")
    _backend: Any = field(default=None, repr=False)
    _ego_xyz_yaw: Tuple[float, float, float, float] = field(
        default=(0.0, 0.0, 0.0, 0.0), repr=False
    )

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "CameraRig":
        img = cfg["image"]
        cams_cfg = cfg["cameras"]
        cameras = {
            name: CameraModel.from_dict(name, cams_cfg[name])
            for name in ("front", "rear", "left", "right")
            if name in cams_cfg
        }
        scene_cfg = cfg.get("scene") or {}
        assets = scene_cfg.get("assets_root")
        root = Path(__file__).resolve().parents[1]
        assets_root = (root / assets) if assets else (root / "assets")
        return cls(
            cameras=cameras,
            image_size=(int(img["width"]), int(img["height"])),
            render_cfg=dict(cfg.get("render") or {}),
            vehicle_cfg=dict(cfg.get("vehicle") or {}),
            assets_root=assets_root,
        )

    @property
    def names(self) -> List[str]:
        return [n for n in self.order if n in self.cameras]

    def ensure_backend(self):
        if self._backend is None:
            from .moderngl_backend import ModernGLFisheyeBackend

            self._backend = ModernGLFisheyeBackend(
                width=self.image_size[0],
                height=self.image_size[1],
                pinhole_fov_y_deg=float(self.render_cfg.get("pinhole_fov_y_deg", 165.0)),
                fisheye_fov_deg=float(self.render_cfg.get("fisheye_fov_deg", 190.0)),
                vehicle_cfg=self.vehicle_cfg,
                assets_root=self.assets_root,
                fisheye_mode=str(self.render_cfg.get("fisheye_mode", "opencv")),
                near=float(self.render_cfg.get("near", 0.05)),
                far=float(self.render_cfg.get("far", 80.0)),
            )
            # Apply last known ego pose so mesh is never stuck at stale origin
            x, y, z, yaw = self._ego_xyz_yaw
            self._backend.set_ego_pose(x, y, z, yaw)
        return self._backend

    def release(self) -> None:
        if self._backend is not None:
            self._backend.release()
            self._backend = None

    def sync_ego_pose(self, ego_xyz_yaw: Tuple[float, float, float, float]) -> None:
        """Push ego pose into the GL backend (dynamic ego mesh)."""
        self._ego_xyz_yaw = (
            float(ego_xyz_yaw[0]),
            float(ego_xyz_yaw[1]),
            float(ego_xyz_yaw[2]),
            float(ego_xyz_yaw[3]),
        )
        if self._backend is not None:
            x, y, z, yaw = self._ego_xyz_yaw
            self._backend.set_ego_pose(x, y, z, yaw)

    def camera_look_gl(
        self,
        name: str,
        ego_xyz_yaw: Tuple[float, float, float, float],
        look_dist: float = 6.0,
    ) -> Dict[str, Tuple[float, float, float]]:
        """World-space look-at in ModernGL coords for one camera given ego pose.

        ego_xyz_yaw: (x, y, z, yaw_rad) in world FLU.
        """
        cam = self.cameras[name]
        eye_v, target_v, up_v = cam.look_at_vehicle(look_dist=look_dist)
        ex, ey, ez, yaw = ego_xyz_yaw
        R = yaw_matrix(yaw)
        t = np.array([ex, ey, ez], dtype=np.float64)

        def to_world(p: np.ndarray) -> np.ndarray:
            return R @ p + t

        eye_w = to_world(eye_v)
        target_w = to_world(target_v)
        up_w = R @ up_v  # direction only
        return {
            "eye": flu_to_gl(eye_w),
            "target": flu_to_gl(target_w),
            "up": flu_to_gl(up_w),
        }

    def render_cameras(
        self,
        ego_xyz_yaw: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    ) -> Dict[str, np.ndarray]:
        """Render all cameras → dict[name, HxWx3 uint8] using each cam's K/D."""
        self.sync_ego_pose(ego_xyz_yaw)
        backend = self.ensure_backend()
        out: Dict[str, np.ndarray] = {}
        for name in self.names:
            cam = self.cameras[name]
            look = self.camera_look_gl(name, ego_xyz_yaw)
            out[name] = backend.render_camera(
                look["eye"],
                look["target"],
                look["up"],
                K=cam.K,
                D=cam.D,
            )
        return out

    def render_god_view(
        self,
        ego_xyz_yaw: Tuple[float, float, float, float],
        eye_flu: Sequence[float],
        target_flu: Sequence[float],
        up_flu: Sequence[float] = (0.0, 0.0, 1.0),
        fov_y_deg: float = 55.0,
    ) -> np.ndarray:
        """Render a world-fixed pinhole overview (god / orbit view)."""
        self.sync_ego_pose(ego_xyz_yaw)
        backend = self.ensure_backend()
        return backend.render_pinhole(
            flu_to_gl(eye_flu),
            flu_to_gl(target_flu),
            flu_to_gl(up_flu),
            fov_y_deg=fov_y_deg,
        )
