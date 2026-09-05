"""OpenCV Kannala–Brandt (cv2.fisheye) LUT helpers for synthetic fisheye.

Pipeline (Phase 2):
  1. Render a wide-FOV pinhole RGB image (ModernGL), top-left origin after flip.
  2. For each output fisheye pixel, invert the OpenCV fisheye model to a 3D ray
     (θ from θ_d via Newton; matches cv2.fisheye poly D=[k1,k2,k3,k4]).
  3. Project the ray into the pinhole image with a known K_pinhole and cv2.remap.

Approximation: a single pinhole cannot cover θ ≥ 90° (FOV ≥ 180°). Those rays
(and any ray outside the pinhole frustum) sample as black. Cube faces would
extend coverage; not required for Phase 2 ground-plane AVM prep.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# Cache LUTs keyed by (w, h, K bytes, D bytes, fov_y, pin_w, pin_h)
_LUT_CACHE: Dict[tuple, Tuple[np.ndarray, np.ndarray]] = {}


def pinhole_K_from_fov(
    width: int,
    height: int,
    fov_y_deg: float,
) -> np.ndarray:
    """Intrinsics matching glm.perspective(fov_y) on a width×height framebuffer.

    OpenCV convention after vertical flip of the GL readback: +X right, +Y down.
    """
    fov_y = math.radians(float(fov_y_deg))
    fy = height / (2.0 * math.tan(0.5 * fov_y))
    fx = fy  # square pixels; horizontal FOV follows aspect
    cx = 0.5 * (width - 1)
    cy = 0.5 * (height - 1)
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _theta_from_theta_d(theta_d: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Vectorized Newton: θ_d = θ (1 + k1 θ² + k2 θ⁴ + k3 θ⁶ + k4 θ⁸)."""
    k1, k2, k3, k4 = [float(x) for x in np.asarray(D, dtype=np.float64).reshape(4)]
    theta = np.array(theta_d, dtype=np.float64, copy=True)
    # Cap initial guess slightly below π to keep poly stable
    theta = np.clip(theta, 0.0, math.pi - 1e-4)
    for _ in range(25):
        t2 = theta * theta
        t4 = t2 * t2
        t6 = t4 * t2
        t8 = t4 * t4
        scale = 1.0 + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8
        f = theta * scale - theta_d
        dscale = (
            2.0 * k1 * theta
            + 4.0 * k2 * theta * t2
            + 6.0 * k3 * theta * t4
            + 8.0 * k4 * theta * t6
        )
        df = scale + theta * dscale
        step = np.where(np.abs(df) > 1e-12, f / df, 0.0)
        theta = np.clip(theta - step, 0.0, math.pi - 1e-4)
        if float(np.max(np.abs(f))) < 1e-10:
            break
    return theta


def fisheye_pixels_to_rays(
    width: int,
    height: int,
    K: np.ndarray,
    D: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Invert OpenCV fisheye for every pixel → unit rays (X right, Y down, Z fwd).

    Returns map_x components as ray_x, ray_y, ray_z (HxW) and valid mask.
    """
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    D = np.asarray(D, dtype=np.float64).reshape(4)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    us = np.arange(width, dtype=np.float64)
    vs = np.arange(height, dtype=np.float64)
    uu, vv = np.meshgrid(us, vs)
    x = (uu - cx) / fx
    y = (vv - cy) / fy
    theta_d = np.sqrt(x * x + y * y)

    theta = _theta_from_theta_d(theta_d, D)
    # φ from distorted normalized coords (same as undistorted direction azimuth)
    phi = np.arctan2(y, x)
    # Center pixel / θ_d≈0
    near0 = theta_d < 1e-12
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    ray_x = sin_t * np.cos(phi)
    ray_y = sin_t * np.sin(phi)
    ray_z = cos_t
    ray_x = np.where(near0, 0.0, ray_x)
    ray_y = np.where(near0, 0.0, ray_y)
    ray_z = np.where(near0, 1.0, ray_z)

    valid = ray_z > 1e-6  # in front of a pinhole (+Z)
    return ray_x, ray_y, ray_z, valid


def build_fisheye_from_pinhole_maps(
    width: int,
    height: int,
    K: np.ndarray,
    D: np.ndarray,
    pinhole_fov_y_deg: float,
    pinhole_width: Optional[int] = None,
    pinhole_height: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build cv2.remap maps: fisheye (u,v) → sample coords in wide pinhole image.

    Maps are float32 HxW. Invalid samples use (-1,-1) so BORDER_CONSTANT → black.
    """
    pw = int(pinhole_width or width)
    ph = int(pinhole_height or height)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    D = np.asarray(D, dtype=np.float64).reshape(4)

    key = (
        width,
        height,
        pw,
        ph,
        float(pinhole_fov_y_deg),
        K.tobytes(),
        D.tobytes(),
    )
    cached = _LUT_CACHE.get(key)
    if cached is not None:
        return cached

    ray_x, ray_y, ray_z, valid = fisheye_pixels_to_rays(width, height, K, D)
    Kp = pinhole_K_from_fov(pw, ph, pinhole_fov_y_deg)
    fxp, fyp = float(Kp[0, 0]), float(Kp[1, 1])
    cxp, cyp = float(Kp[0, 2]), float(Kp[1, 2])

    # Perspective divide into pinhole pixels
    with np.errstate(divide="ignore", invalid="ignore"):
        u_p = fxp * (ray_x / ray_z) + cxp
        v_p = fyp * (ray_y / ray_z) + cyp

    in_frame = (
        valid
        & np.isfinite(u_p)
        & np.isfinite(v_p)
        & (u_p >= 0.0)
        & (u_p <= pw - 1.0)
        & (v_p >= 0.0)
        & (v_p <= ph - 1.0)
    )
    map_x = np.where(in_frame, u_p, -1.0).astype(np.float32)
    map_y = np.where(in_frame, v_p, -1.0).astype(np.float32)

    _LUT_CACHE[key] = (map_x, map_y)
    return map_x, map_y


def remap_pinhole_to_fisheye(
    pinhole_rgb: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    pinhole_fov_y_deg: float,
    out_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Apply OpenCV-fisheye LUT remap. pinhole_rgb is HxWx3 uint8 (top-left)."""
    ph, pw = pinhole_rgb.shape[:2]
    if out_size is None:
        width, height = pw, ph
    else:
        width, height = int(out_size[0]), int(out_size[1])

    map_x, map_y = build_fisheye_from_pinhole_maps(
        width,
        height,
        K,
        D,
        pinhole_fov_y_deg=pinhole_fov_y_deg,
        pinhole_width=pw,
        pinhole_height=ph,
    )
    return cv2.remap(
        pinhole_rgb,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def verify_kd_roundtrip(
    K: np.ndarray,
    D: np.ndarray,
    n: int = 64,
) -> float:
    """Max |reproj| px for random rays with θ < 80° via cv2.fisheye.distortPoints."""
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    D = np.asarray(D, dtype=np.float64).reshape(4, 1)
    rng = np.random.default_rng(0)
    theta = rng.uniform(0.0, math.radians(80.0), size=n)
    phi = rng.uniform(-math.pi, math.pi, size=n)
    # Normalized plane a=X/Z, b=Y/Z
    a = np.tan(theta) * np.cos(phi)
    b = np.tan(theta) * np.sin(phi)
    obj = np.stack([a, b], axis=-1).reshape(-1, 1, 2).astype(np.float64)
    dist = cv2.fisheye.distortPoints(obj, K, D)
    und = cv2.fisheye.undistortPoints(dist, K, D)
    # Back to pixels via our ray path vs distort
    # Compare distort(undistort(dist)) ≈ dist
    redist = cv2.fisheye.distortPoints(und, K, D)
    err = np.linalg.norm(redist.reshape(-1, 2) - dist.reshape(-1, 2), axis=1)
    return float(np.max(err))


def clear_lut_cache() -> None:
    _LUT_CACHE.clear()
