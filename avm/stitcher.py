"""AVM (around-view monitor) stitcher — Phase 3 BEV IPM + feather blend.

Pipeline per camera (same OpenCV K/D + YAML extrinsics as Phase 2):
  BEV ground grid (vehicle FLU, z=0)
    → camera-local frame (R, t from config)
    → OpenCV optical (X right, Y down, Z forward)
    → Kannala–Brandt fisheye project (cv2.fisheye-compatible)
    → cv2.remap from fisheye RGB
  Overlaps blended with distance-transform feathering; vehicle ROI masked.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple

import cv2
import numpy as np

from sensors.cameras import CameraModel

# Cache: BEV→fisheye maps per (cam name, K, D, t, R, bev extents, img size)
_IPM_CACHE: Dict[tuple, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def bev_grid_shape(bev_cfg: Mapping[str, Any]) -> Tuple[int, int, float]:
    """Return (height_px, width_px, meters_per_px) for the BEV canvas."""
    res = float(bev_cfg.get("resolution_m_per_px", 0.02))
    fwd = float(bev_cfg.get("forward_m", 8.0))
    back = float(bev_cfg.get("back_m", 8.0))
    left = float(bev_cfg.get("left_m", 6.0))
    right = float(bev_cfg.get("right_m", 6.0))
    height_px = int(round((fwd + back) / res))
    width_px = int(round((left + right) / res))
    return height_px, width_px, res


def bev_pixel_to_vehicle_xy(
    bev_cfg: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Meshgrid of ground (x, y) in vehicle FLU for each BEV pixel center.

    Image layout (looking down −z): top = +x forward, left = +y left.
    """
    h, w, res = bev_grid_shape(bev_cfg)
    fwd = float(bev_cfg.get("forward_m", 8.0))
    left = float(bev_cfg.get("left_m", 6.0))
    # row 0 → x = +forward; col 0 → y = +left
    rows = np.arange(h, dtype=np.float64)
    cols = np.arange(w, dtype=np.float64)
    xs = fwd - (rows + 0.5) * res  # (H,)
    ys = left - (cols + 0.5) * res  # (W,)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")  # (H, W)
    return xx, yy


def vehicle_roi_mask(
    bev_cfg: Mapping[str, Any],
    vehicle_cfg: Optional[Mapping[str, Any]] = None,
    margin_m: float = 0.50,
) -> np.ndarray:
    """Boolean mask True where BEV pixels fall inside the ego body rectangle."""
    vehicle_cfg = vehicle_cfg or {}
    length = float(vehicle_cfg.get("length_m", 4.4))
    width = float(vehicle_cfg.get("width_m", 1.9))
    xx, yy = bev_pixel_to_vehicle_xy(bev_cfg)
    half_l = 0.5 * length + margin_m
    half_w = 0.5 * width + margin_m
    return (np.abs(xx) <= half_l) & (np.abs(yy) <= half_w)


def _vehicle_to_opencv(
    P_v: np.ndarray,
    t: np.ndarray,
    R: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map vehicle-FLU points (..., 3) → OpenCV cam coords (X right, Y down, Z fwd).

    CameraModel.rotation_matrix() maps camera-local → vehicle with:
      cam X = optical forward, cam Y = left, cam Z = up
    (same convention as sensors.cameras look-at). OpenCV needs Z-forward.
    """
    P_rel = P_v - t.reshape(1, 1, 3)
    # vehicle → cam_local: R^T
    # (H,W,3) @ R  is wrong; use einsum
    P_cam = np.einsum("ij,...j->...i", R.T, P_rel)
    X = -P_cam[..., 1]  # right = -left
    Y = -P_cam[..., 2]  # down = -up
    Z = P_cam[..., 0]  # forward
    return X, Y, Z


def _fisheye_project_kb(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kannala–Brandt project matching cv2.fisheye.distortPoints.

    Returns u, v, valid (Z>eps and finite).
    """
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    D = np.asarray(D, dtype=np.float64).reshape(4)
    k1, k2, k3, k4 = [float(x) for x in D]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    valid = Z > 1e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(valid, X / Z, 0.0)
        b = np.where(valid, Y / Z, 0.0)
    r = np.sqrt(a * a + b * b)
    # θ = atan(r); for r≈0, θ≈0 and θ_d/r → 1
    theta = np.arctan(r)
    t2 = theta * theta
    t4 = t2 * t2
    t6 = t4 * t2
    t8 = t4 * t4
    theta_d = theta * (1.0 + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8)
    # scale = θ_d / r ; when r→0 use 1
    near0 = r < 1e-12
    scale = np.where(near0, 1.0, theta_d / np.maximum(r, 1e-12))
    x_d = scale * a
    y_d = scale * b
    u = fx * x_d + cx
    v = fy * y_d + cy
    finite = np.isfinite(u) & np.isfinite(v)
    return u, v, valid & finite


def build_ipm_maps(
    cam: CameraModel,
    bev_cfg: Mapping[str, Any],
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build remap maps (map_x, map_y) and a soft weight mask for one camera.

    map_* are float32 HxW_bev; invalid samples set to -1.
    weight is float32 in [0,1]: distance-to-border feathering inside valid region.
    """
    width, height = int(image_size[0]), int(image_size[1])
    h, w, res = bev_grid_shape(bev_cfg)
    R = cam.rotation_matrix()
    t = cam.translation_m.astype(np.float64)
    K = cam.K
    D = cam.D

    key = (
        cam.name,
        h,
        w,
        res,
        width,
        height,
        float(bev_cfg.get("forward_m", 8.0)),
        float(bev_cfg.get("back_m", 8.0)),
        float(bev_cfg.get("left_m", 6.0)),
        float(bev_cfg.get("right_m", 6.0)),
        K.tobytes(),
        D.tobytes(),
        t.tobytes(),
        R.tobytes(),
    )
    cached = _IPM_CACHE.get(key)
    if cached is not None:
        return cached

    xx, yy = bev_pixel_to_vehicle_xy(bev_cfg)
    zz = np.zeros_like(xx)
    P = np.stack([xx, yy, zz], axis=-1)  # (H, W, 3)

    X, Y, Z = _vehicle_to_opencv(P, t, R)
    u, v, valid = _fisheye_project_kb(X, Y, Z, K, D)

    in_img = (
        valid
        & (u >= 0.0)
        & (u <= width - 1.0)
        & (v >= 0.0)
        & (v <= height - 1.0)
    )
    map_x = np.where(in_img, u, -1.0).astype(np.float32)
    map_y = np.where(in_img, v, -1.0).astype(np.float32)

    # Soft weight: distance to invalid border (feather seams)
    valid_u8 = in_img.astype(np.uint8)
    if int(valid_u8.max()) == 0:
        weight = np.zeros((h, w), dtype=np.float32)
    else:
        dist = cv2.distanceTransform(valid_u8, cv2.DIST_L2, 5)
        # Feather width ~0.4 m in BEV pixels
        feather_px = max(1.0, 0.4 / res)
        weight = np.clip(dist / feather_px, 0.0, 1.0).astype(np.float32)
        # Prefer viewing direction: boost pixels in front of each cam optical axis
        fwd = R @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
        # Horizontal preference in vehicle XY
        view_xy = fwd[:2]
        nrm = float(np.linalg.norm(view_xy)) + 1e-12
        view_xy = view_xy / nrm
        # Vector from cam to ground point (horizontal)
        dx = xx - t[0]
        dy = yy - t[1]
        align = dx * view_xy[0] + dy * view_xy[1]
        # Mild half-plane preference (keep overlap bright for feathering)
        pref = 0.65 + 0.35 * np.tanh(align / 2.5)
        weight = weight * pref.astype(np.float32)
        # Floor inside valid region so seams do not collapse to black
        weight = np.where(in_img, np.maximum(weight, 0.05), 0.0).astype(np.float32)

    _IPM_CACHE[key] = (map_x, map_y, weight)
    return map_x, map_y, weight


def clear_ipm_cache() -> None:
    _IPM_CACHE.clear()


def warp_camera_to_bev(
    image: np.ndarray,
    cam: CameraModel,
    bev_cfg: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp one fisheye RGB to BEV. Returns (rgb float32 HxWx3, weight float32 HxW)."""
    h_img, w_img = image.shape[:2]
    map_x, map_y, weight = build_ipm_maps(cam, bev_cfg, (w_img, h_img))
    warped = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return warped.astype(np.float32), weight


def stitch_avm(
    images: Mapping[str, np.ndarray],
    bev_cfg: dict | None = None,
    cameras: Mapping[str, CameraModel] | None = None,
    vehicle_cfg: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Stitch surround fisheye images into a bird's-eye (BEV) mosaic.

    Args:
        images: mapping camera name → HxWx3 uint8 RGB (fisheye)
        bev_cfg: BEV extent/resolution from config ``bev:``
        cameras: CameraModel map (K/D/extrinsics). Required for real stitch.
        vehicle_cfg: optional vehicle size for ROI mask

    Returns:
        BEV image as HxWx3 uint8
    """
    if bev_cfg is None:
        bev_cfg = {
            "forward_m": 8.0,
            "back_m": 8.0,
            "left_m": 6.0,
            "right_m": 6.0,
            "resolution_m_per_px": 0.02,
        }
    if cameras is None:
        raise ValueError(
            "stitch_avm requires cameras (CameraModel map with K/D/extrinsics); "
            "pass via World.stitch_avm or cameras="
        )

    h, w, _res = bev_grid_shape(bev_cfg)
    acc = np.zeros((h, w, 3), dtype=np.float64)
    wsum = np.zeros((h, w), dtype=np.float64)

    order = ("front", "rear", "left", "right")
    for name in order:
        if name not in images or name not in cameras:
            continue
        img = images[name]
        cam = cameras[name]
        warped, weight = warp_camera_to_bev(img, cam, bev_cfg)
        w_f = weight.astype(np.float64)
        acc += warped.astype(np.float64) * w_f[..., None]
        wsum += w_f

    # Avoid div0
    mask = wsum > 1e-6
    out = np.zeros((h, w, 3), dtype=np.float64)
    out[mask] = acc[mask] / wsum[mask, None]
    out = np.clip(out, 0.0, 255.0).astype(np.uint8)

    # Mask vehicle body ROI to dark gray (no fake under-car texture)
    roi = vehicle_roi_mask(bev_cfg, vehicle_cfg)
    out[roi] = (40, 42, 48)

    return out


def draw_slot_overlay_bev(
    bev: np.ndarray,
    ego_xyz_yaw: Tuple[float, float, float, float],
    bev_cfg: Mapping[str, Any],
    color: Tuple[int, int, int] = (255, 220, 60),
) -> np.ndarray:
    """Optional: draw procedural lot slot rectangles into BEV (ego frame).

    Phase 3 helper for visualization; Phase 4 can compare against GT paint.
    """
    from scene.lot import (
        N_SLOTS,
        SLOT_D,
        SLOT_ORIGIN_X,
        SLOT_ORIGIN_Z_NORTH,
        SLOT_ORIGIN_Z_SOUTH,
        SLOT_W,
    )
    from sensors.cameras import gl_to_flu, yaw_matrix

    out = bev.copy()
    h, w, res = bev_grid_shape(bev_cfg)
    fwd = float(bev_cfg.get("forward_m", 8.0))
    left = float(bev_cfg.get("left_m", 6.0))
    ex, ey, _ez, yaw = ego_xyz_yaw
    R = yaw_matrix(yaw)
    t = np.array([ex, ey, 0.0], dtype=np.float64)

    def world_flu_to_bev_px(px: float, py: float) -> Optional[Tuple[int, int]]:
        # world → ego
        p = R.T @ (np.array([px, py, 0.0]) - t)
        x_e, y_e = float(p[0]), float(p[1])
        col = int(round((left - y_e) / res - 0.5))
        row = int(round((fwd - x_e) / res - 0.5))
        if 0 <= row < h and 0 <= col < w:
            return col, row  # cv2 uses (x,y)=(col,row)
        return None

    def slot_corners_flu(origin_x_gl: float, origin_z_gl: float, i: int):
        # Slot rect in GL XZ → FLU
        x0 = origin_x_gl + i * SLOT_W
        x1 = x0 + SLOT_W
        z0 = origin_z_gl
        z1 = origin_z_gl + SLOT_D
        corners_gl = [
            (x0, 0.0, z0),
            (x1, 0.0, z0),
            (x1, 0.0, z1),
            (x0, 0.0, z1),
        ]
        return [gl_to_flu(c) for c in corners_gl]

    for oz in (SLOT_ORIGIN_Z_SOUTH, SLOT_ORIGIN_Z_NORTH):
        for i in range(N_SLOTS):
            corners = slot_corners_flu(SLOT_ORIGIN_X, oz, i)
            pts = []
            for c in corners:
                pq = world_flu_to_bev_px(c[0], c[1])
                if pq is not None:
                    pts.append(pq)
            if len(pts) >= 2:
                arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(out, [arr], isClosed=True, color=color, thickness=1)

    # Ego forward axis tick
    mid = world_flu_to_bev_px(ex + math.cos(yaw) * 2.0, ey + math.sin(yaw) * 2.0)
    origin = world_flu_to_bev_px(ex, ey)
    if mid is not None and origin is not None:
        cv2.arrowedLine(out, origin, mid, (80, 220, 255), 1, tipLength=0.2)

    return out
