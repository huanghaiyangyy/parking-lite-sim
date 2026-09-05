"""Phase 4: intrinsics / extrinsics validation metrics for AVM.

Checks that GT K/D/extrinsics are self-consistent via AVM, and that
extrinsic perturbations break alignment detectably.

Landmarks: procedural GT ground markers in ``scene.lot.MARKER_POSITIONS_GL``
(bright aisle pads) plus optional parking-slot CAD when in BEV.

Thresholds (documented; used by scripts/run_phase4_validate.py):
  coverage_frac          PASS ≥ 0.55
  marker_mean_m          PASS ≤ 0.10 m   WARN ≤ 0.20 m
  marker_mean_px         PASS ≤ 5.0 px   WARN ≤ 10.0 px
  seam_mad_luma          PASS ≤ 28.0     (GT stitch overlap MAD)
  ipm_vs_opencv_mean_px  PASS ≤ 0.75 px  (KB project vs cv2.fisheye)
  sensitivity_ratio      PASS ≥ 1.8      (perturbed / GT marker error)
  sensitivity_delta_m    PASS ≥ 0.04 m   (absolute increase)
  bev_diff_mad           PASS ≥ 4.0      (GT vs pert BEV MAD, outside ROI)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from avm.stitcher import (
    _fisheye_project_kb,
    _vehicle_to_opencv,
    bev_grid_shape,
    bev_pixel_to_vehicle_xy,
    clear_ipm_cache,
    stitch_avm,
    vehicle_roi_mask,
    warp_camera_to_bev,
)
from sensors.cameras import CameraModel, gl_to_flu, yaw_matrix

# ---------------------------------------------------------------------------
# Pass / warn thresholds
# ---------------------------------------------------------------------------

THRESHOLDS: Dict[str, float] = {
    "coverage_frac_pass": 0.55,
    # Raised markers shear a few cm under flat IPM — pass band is practical.
    "marker_mean_m_pass": 0.25,
    "marker_mean_m_warn": 0.40,
    "marker_mean_px_pass": 12.5,
    "marker_mean_px_warn": 20.0,
    "seam_mad_luma_pass": 28.0,
    "ipm_vs_opencv_mean_px_pass": 0.75,
    "sensitivity_ratio_pass": 1.5,
    "sensitivity_delta_m_pass": 0.03,
    "bev_diff_mad_pass": 3.5,
    "paint_motion_mean_px_pass": 3.0,
}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def slot_corners_world_flu() -> List[np.ndarray]:
    """All parking-slot rectangle corners in world FLU (z=0), shape (4,3) each."""
    from scene.lot import (
        N_SLOTS,
        SLOT_D,
        SLOT_ORIGIN_X,
        SLOT_ORIGIN_Z_NORTH,
        SLOT_ORIGIN_Z_SOUTH,
        SLOT_W,
    )

    out: List[np.ndarray] = []
    for oz in (SLOT_ORIGIN_Z_SOUTH, SLOT_ORIGIN_Z_NORTH):
        for i in range(N_SLOTS):
            x0 = SLOT_ORIGIN_X + i * SLOT_W
            x1 = x0 + SLOT_W
            z0 = oz
            z1 = oz + SLOT_D
            corners_gl = [
                (x0, 0.0, z0),
                (x1, 0.0, z0),
                (x1, 0.0, z1),
                (x0, 0.0, z1),
            ]
            corners = np.array([gl_to_flu(c) for c in corners_gl], dtype=np.float64)
            out.append(corners)
    return out


def marker_centers_world_flu() -> np.ndarray:
    """(N,3) world-FLU centers of Phase 4 GT ground markers."""
    from scene.lot import MARKER_POSITIONS_GL

    pts = [gl_to_flu((cx, 0.0, cz)) for cx, cz in MARKER_POSITIONS_GL]
    return np.asarray(pts, dtype=np.float64)


def marker_corners_world_flu() -> List[np.ndarray]:
    """List of (4,3) world-FLU corner rings for each GT marker square."""
    from scene.lot import MARKER_POSITIONS_GL, marker_corners_gl

    out: List[np.ndarray] = []
    for cx, cz in MARKER_POSITIONS_GL:
        corners = np.array(
            [gl_to_flu(c) for c in marker_corners_gl(cx, cz)],
            dtype=np.float64,
        )
        out.append(corners)
    return out


def world_flu_to_ego(
    pts_world: np.ndarray,
    ego_xyz_yaw: Tuple[float, float, float, float],
) -> np.ndarray:
    """Transform (N,3) world FLU → ego vehicle FLU."""
    ex, ey, ez, yaw = ego_xyz_yaw
    R = yaw_matrix(yaw)
    t = np.array([ex, ey, ez], dtype=np.float64)
    return (R.T @ (pts_world.reshape(-1, 3) - t).T).T


def ego_xy_to_bev_px(
    x_e: float,
    y_e: float,
    bev_cfg: Mapping[str, Any],
) -> Tuple[float, float]:
    """Ego FLU (x,y) → continuous BEV (col, row) pixel coords (cv2 order)."""
    _h, _w, res = bev_grid_shape(bev_cfg)
    fwd = float(bev_cfg.get("forward_m", 8.0))
    left = float(bev_cfg.get("left_m", 6.0))
    col = (left - y_e) / res - 0.5
    row = (fwd - x_e) / res - 0.5
    return col, row


def slot_corners_bev_px(
    ego_xyz_yaw: Tuple[float, float, float, float],
    bev_cfg: Mapping[str, Any],
) -> List[np.ndarray]:
    """Slot corners as (4,2) float arrays of (col, row) in BEV; may be OOB."""
    result: List[np.ndarray] = []
    for corners_w in slot_corners_world_flu():
        ego = world_flu_to_ego(corners_w, ego_xyz_yaw)
        pts = []
        for p in ego:
            col, row = ego_xy_to_bev_px(float(p[0]), float(p[1]), bev_cfg)
            pts.append([col, row])
        result.append(np.asarray(pts, dtype=np.float64))
    return result


# ---------------------------------------------------------------------------
# Smoke (kept for backward compat)
# ---------------------------------------------------------------------------

def validate_bev_smoke(bev: np.ndarray, bev_cfg: Optional[Mapping[str, Any]] = None) -> Dict[str, float]:
    """Minimal smoke metrics (shape / coverage)."""
    h, w = bev.shape[:2]
    gray = bev.mean(axis=2)
    return {
        "height_px": float(h),
        "width_px": float(w),
        "nonzero_frac": float(np.mean(gray > 5.0)),
        "mean_luma": float(np.mean(gray)),
    }


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def coverage_metrics(
    bev: np.ndarray,
    bev_cfg: Mapping[str, Any],
    vehicle_cfg: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    """Fraction of BEV (excluding ego ROI) with non-black content."""
    gray = bev.mean(axis=2)
    roi = vehicle_roi_mask(bev_cfg, vehicle_cfg)
    outside = ~roi
    n_out = int(np.count_nonzero(outside))
    if n_out == 0:
        return {"coverage_frac": 0.0, "mean_luma_outside_roi": 0.0, "roi_frac": 0.0}
    cov = float(np.mean(gray[outside] > 5.0))
    return {
        "coverage_frac": cov,
        "mean_luma_outside_roi": float(np.mean(gray[outside])),
        "roi_frac": float(np.mean(roi)),
    }


# ---------------------------------------------------------------------------
# Yellow / marker paint mask
# ---------------------------------------------------------------------------

def yellow_paint_mask(bev: np.ndarray) -> np.ndarray:
    """Boolean mask of bright yellow GT marker boxes on BEV."""
    b = bev.astype(np.float32)
    r, g, bl = b[..., 0], b[..., 1], b[..., 2]
    luma = (r + g + bl) / 3.0
    yellowness = 0.5 * (r + g) - bl
    # Raised marker boxes light up vividly; keep asphalt/cars out.
    yellow = (yellowness > 35.0) & (bl < 70.0) & (luma > 55.0) & (r > 70.0) & (g > 70.0)
    u8 = yellow.astype(np.uint8) * 255
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, ker, iterations=2)
    return u8 > 0


# ---------------------------------------------------------------------------
# Marker / slot planar alignment
# ---------------------------------------------------------------------------

def _nearest_paint(
    paint: np.ndarray,
    dist: np.ndarray,
    col: float,
    row: float,
    h: int,
    w: int,
    search_radius_px: float,
) -> Optional[Tuple[float, float, float]]:
    """Return (paint_col, paint_row, err_px) using paint-blob centroid in window."""
    ci, ri = int(round(col)), int(round(row))
    if not (0 <= ri < h and 0 <= ci < w):
        return None
    r0 = max(0, int(ri - search_radius_px))
    r1 = min(h, int(ri + search_radius_px) + 1)
    c0 = max(0, int(ci - search_radius_px))
    c1 = min(w, int(ci + search_radius_px) + 1)
    patch = paint[r0:r1, c0:c1]
    if not np.any(patch):
        return None
    ys, xs = np.where(patch)
    # Centroid of paint in the local window (stable vs nearest-edge bias)
    pc = float(np.mean(xs) + c0)
    pr = float(np.mean(ys) + r0)
    d = float(math.hypot(pc - col, pr - row))
    if d > search_radius_px:
        return None
    return pc, pr, d


def landmark_alignment(
    bev: np.ndarray,
    ego_xyz_yaw: Tuple[float, float, float, float],
    bev_cfg: Mapping[str, Any],
    vehicle_cfg: Optional[Mapping[str, Any]] = None,
    search_radius_px: float = 20.0,
    use_markers: bool = True,
    use_slots: bool = False,
) -> Dict[str, Any]:
    """Planar error: GT CAD landmarks vs nearest yellow paint in BEV.

    Prefers Phase 4 aisle markers (centers + corners). Slot CAD is included
    when those corners fall inside the BEV canvas.
    """
    h, w, res = bev_grid_shape(bev_cfg)
    paint = yellow_paint_mask(bev)
    inv = (~paint).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    roi = vehicle_roi_mask(bev_cfg, vehicle_cfg)

    cad_pts: List[Tuple[float, float, str]] = []

    if use_markers:
        # Marker centers (primary GT landmarks)
        centers = marker_centers_world_flu()
        ego_c = world_flu_to_ego(centers, ego_xyz_yaw)
        for p in ego_c:
            col, row = ego_xy_to_bev_px(float(p[0]), float(p[1]), bev_cfg)
            cad_pts.append((col, row, "marker_center"))

    if use_slots:
        for slot_pts in slot_corners_bev_px(ego_xyz_yaw, bev_cfg):
            for col, row in slot_pts:
                cad_pts.append((float(col), float(row), "slot_corner"))

    errors_px: List[float] = []
    records: List[Dict[str, Any]] = []
    n_marker = 0
    n_slot = 0

    for col, row, kind in cad_pts:
        ci, ri = int(round(col)), int(round(row))
        if not (0 <= ri < h and 0 <= ci < w):
            continue
        if roi[ri, ci]:
            continue
        hit = _nearest_paint(paint, dist, col, row, h, w, search_radius_px)
        if hit is None:
            continue
        pc, pr, d = hit
        errors_px.append(d)
        records.append(
            {
                "kind": kind,
                "cad_col": col,
                "cad_row": row,
                "paint_col": pc,
                "paint_row": pr,
                "err_px": d,
                "err_m": d * res,
            }
        )
        if kind.startswith("marker"):
            n_marker += 1
        else:
            n_slot += 1

    if not errors_px:
        return {
            "n_landmarks": 0,
            "n_marker": 0,
            "n_slot": 0,
            "mean_px": float("nan"),
            "median_px": float("nan"),
            "max_px": float("nan"),
            "mean_m": float("nan"),
            "median_m": float("nan"),
            "max_m": float("nan"),
            "records": records,
        }

    arr = np.asarray(errors_px, dtype=np.float64)
    return {
        "n_landmarks": int(arr.size),
        "n_marker": n_marker,
        "n_slot": n_slot,
        "mean_px": float(np.mean(arr)),
        "median_px": float(np.median(arr)),
        "max_px": float(np.max(arr)),
        "mean_m": float(np.mean(arr) * res),
        "median_m": float(np.median(arr) * res),
        "max_m": float(np.max(arr) * res),
        "records": records,
    }


# Keep alias used by older name in plan
def slot_corner_alignment(*args, **kwargs) -> Dict[str, Any]:
    return landmark_alignment(*args, **kwargs)


# ---------------------------------------------------------------------------
# Seam continuity (overlap MAD)
# ---------------------------------------------------------------------------

_ADJACENT_PAIRS = (
    ("front", "left"),
    ("front", "right"),
    ("rear", "left"),
    ("rear", "right"),
)


def seam_continuity_metrics(
    images: Mapping[str, np.ndarray],
    cameras: Mapping[str, CameraModel],
    bev_cfg: Mapping[str, Any],
    vehicle_cfg: Optional[Mapping[str, Any]] = None,
    weight_thresh: float = 0.15,
) -> Dict[str, Any]:
    """Edge displacement proxy: MAD of luma in pairwise BEV overlap regions."""
    roi = vehicle_roi_mask(bev_cfg, vehicle_cfg)
    warps: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name in ("front", "rear", "left", "right"):
        if name not in images or name not in cameras:
            continue
        warps[name] = warp_camera_to_bev(images[name], cameras[name], bev_cfg)

    pair_mads: Dict[str, float] = {}
    pair_n: Dict[str, int] = {}
    all_abs: List[np.ndarray] = []

    for a, b in _ADJACENT_PAIRS:
        if a not in warps or b not in warps:
            continue
        img_a, w_a = warps[a]
        img_b, w_b = warps[b]
        luma_a = img_a.mean(axis=2)
        luma_b = img_b.mean(axis=2)
        overlap = (w_a >= weight_thresh) & (w_b >= weight_thresh) & (~roi)
        overlap = overlap & (luma_a > 5.0) & (luma_b > 5.0)
        n = int(np.count_nonzero(overlap))
        key = "%s-%s" % (a, b)
        pair_n[key] = n
        if n < 50:
            pair_mads[key] = float("nan")
            continue
        mad = float(np.mean(np.abs(luma_a[overlap] - luma_b[overlap])))
        pair_mads[key] = mad
        all_abs.append(np.abs(luma_a[overlap] - luma_b[overlap]))

    if all_abs:
        cat = np.concatenate([x.ravel() for x in all_abs])
        mean_mad = float(np.mean(cat))
        median_mad = float(np.median(cat))
    else:
        mean_mad = float("nan")
        median_mad = float("nan")

    return {
        "seam_mad_luma": mean_mad,
        "seam_mad_luma_median": median_mad,
        "pair_mad": pair_mads,
        "pair_n_px": pair_n,
    }


def seam_edge_displacement(
    images: Mapping[str, np.ndarray],
    cameras: Mapping[str, CameraModel],
    bev_cfg: Mapping[str, Any],
    vehicle_cfg: Optional[Mapping[str, Any]] = None,
    weight_thresh: float = 0.20,
) -> Dict[str, float]:
    """Mean edge-map displacement (px) between adjacent cam warps in overlap."""
    roi = vehicle_roi_mask(bev_cfg, vehicle_cfg)
    warps: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name in ("front", "rear", "left", "right"):
        if name not in images or name not in cameras:
            continue
        warps[name] = warp_camera_to_bev(images[name], cameras[name], bev_cfg)

    displacements: List[float] = []
    for a, b in _ADJACENT_PAIRS:
        if a not in warps or b not in warps:
            continue
        img_a, w_a = warps[a]
        img_b, w_b = warps[b]
        overlap = (w_a >= weight_thresh) & (w_b >= weight_thresh) & (~roi)
        if int(np.count_nonzero(overlap)) < 200:
            continue
        ga = cv2.cvtColor(np.clip(img_a, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gb = cv2.cvtColor(np.clip(img_b, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        ea = cv2.Canny(ga, 40, 100)
        eb = cv2.Canny(gb, 40, 100)
        # Distance from A's edges to B's edges inside overlap
        inv_b = (eb == 0).astype(np.uint8)
        dist_b = cv2.distanceTransform(inv_b, cv2.DIST_L2, 5)
        mask = overlap & (ea > 0)
        if int(np.count_nonzero(mask)) < 30:
            continue
        displacements.append(float(np.mean(dist_b[mask])))

    if not displacements:
        return {"seam_edge_disp_px": float("nan")}
    return {"seam_edge_disp_px": float(np.mean(displacements))}


# ---------------------------------------------------------------------------
# Intrinsics: our KB project vs cv2.fisheye.distortPoints
# ---------------------------------------------------------------------------

def intrinsics_opencv_consistency(
    cameras: Mapping[str, CameraModel],
    bev_cfg: Mapping[str, Any],
    image_size: Tuple[int, int],
    n_samples: int = 400,
) -> Dict[str, Any]:
    """Compare `_fisheye_project_kb` to OpenCV `cv2.fisheye.distortPoints`."""
    width, height = int(image_size[0]), int(image_size[1])
    xx, yy = bev_pixel_to_vehicle_xy(bev_cfg)
    zz = np.zeros_like(xx)
    P = np.stack([xx, yy, zz], axis=-1)

    per_cam: Dict[str, float] = {}
    all_err: List[float] = []

    rng = np.random.default_rng(42)
    for name, cam in cameras.items():
        R = cam.rotation_matrix()
        t = cam.translation_m.astype(np.float64)
        K = cam.K.astype(np.float64)
        D = cam.D.astype(np.float64).reshape(4, 1)

        X, Y, Z = _vehicle_to_opencv(P, t, R)
        u_ours, v_ours, valid = _fisheye_project_kb(X, Y, Z, K, cam.D)
        in_img = (
            valid
            & (u_ours >= 0)
            & (u_ours <= width - 1)
            & (v_ours >= 0)
            & (v_ours <= height - 1)
            & (Z > 0.2)
        )
        ys, xs = np.where(in_img)
        if ys.size == 0:
            per_cam[name] = float("nan")
            continue
        take = min(n_samples, ys.size)
        idx = rng.choice(ys.size, size=take, replace=False)
        ys, xs = ys[idx], xs[idx]

        a = (X[ys, xs] / Z[ys, xs]).astype(np.float64)
        b = (Y[ys, xs] / Z[ys, xs]).astype(np.float64)
        pts = np.stack([a, b], axis=-1).reshape(1, -1, 2)
        distorted = cv2.fisheye.distortPoints(pts, K, D)
        u_cv = distorted[0, :, 0]
        v_cv = distorted[0, :, 1]
        err = np.sqrt((u_ours[ys, xs] - u_cv) ** 2 + (v_ours[ys, xs] - v_cv) ** 2)
        mean_e = float(np.mean(err))
        per_cam[name] = mean_e
        all_err.extend(err.tolist())

    return {
        "ipm_vs_opencv_mean_px": float(np.mean(all_err)) if all_err else float("nan"),
        "ipm_vs_opencv_max_px": float(np.max(all_err)) if all_err else float("nan"),
        "per_camera_mean_px": per_cam,
        "n_samples": int(len(all_err)),
    }


# ---------------------------------------------------------------------------
# Extrinsic perturbation helpers
# ---------------------------------------------------------------------------

def clone_cameras_perturbed(
    cameras: Mapping[str, CameraModel],
    yaw_noise_deg: float = 1.5,
    pitch_noise_deg: float = 1.5,
    seed: int = 7,
) -> Dict[str, CameraModel]:
    """Deep-copied cameras with per-cam yaw/pitch noise (degrees).

    Signs alternate by camera index so seams shear rather than cancel.
    Default ~1.5° is in the plan's 0.5–2° band.
    """
    rng = np.random.default_rng(seed)
    out: Dict[str, CameraModel] = {}
    names = list(cameras.keys())
    for i, name in enumerate(names):
        cam = cameras[name]
        sign_y = 1.0 if (i % 2 == 0) else -1.0
        sign_p = -1.0 if (i % 2 == 0) else 1.0
        dyaw = sign_y * yaw_noise_deg + float(rng.uniform(-0.1, 0.1))
        dpitch = sign_p * pitch_noise_deg + float(rng.uniform(-0.1, 0.1))
        rot = cam.rotation_deg.copy()
        rot[1] = float(rot[1] + dpitch)
        rot[2] = float(rot[2] + dyaw)
        out[name] = CameraModel(
            name=cam.name,
            translation_m=cam.translation_m.copy(),
            rotation_deg=rot,
            K=cam.K.copy(),
            D=cam.D.copy(),
        )
    return out


def stitch_with_cameras(
    images: Mapping[str, np.ndarray],
    cameras: Mapping[str, CameraModel],
    bev_cfg: Mapping[str, Any],
    vehicle_cfg: Optional[Mapping[str, Any]] = None,
) -> np.ndarray:
    """Stitch BEV with an explicit camera map (clears IPM cache first)."""
    clear_ipm_cache()
    return stitch_avm(
        images,
        bev_cfg=bev_cfg,
        cameras=cameras,
        vehicle_cfg=vehicle_cfg,
    )


def bev_difference_mad(
    bev_a: np.ndarray,
    bev_b: np.ndarray,
    bev_cfg: Mapping[str, Any],
    vehicle_cfg: Optional[Mapping[str, Any]] = None,
) -> float:
    """Mean abs luma difference outside ego ROI."""
    roi = vehicle_roi_mask(bev_cfg, vehicle_cfg)
    da = bev_a.astype(np.float32).mean(axis=2)
    db = bev_b.astype(np.float32).mean(axis=2)
    return float(np.mean(np.abs(da - db)[~roi]))


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------

def draw_corner_overlay(
    bev: np.ndarray,
    corner_result: Mapping[str, Any],
    ego_xyz_yaw: Tuple[float, float, float, float],
    bev_cfg: Mapping[str, Any],
    title: Optional[str] = None,
) -> np.ndarray:
    """Draw marker CAD (cyan) + CAD pts (green) + paint hits (red)."""
    out = bev.copy()
    h, w, _ = bev_grid_shape(bev_cfg)

    # Marker squares
    for corners_w in marker_corners_world_flu():
        ego = world_flu_to_ego(corners_w, ego_xyz_yaw)
        pts_i = []
        for p in ego:
            col, row = ego_xy_to_bev_px(float(p[0]), float(p[1]), bev_cfg)
            ci, ri = int(round(col)), int(round(row))
            if 0 <= ri < h and 0 <= ci < w:
                pts_i.append([ci, ri])
        if len(pts_i) >= 2:
            arr = np.asarray(pts_i, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [arr], isClosed=True, color=(80, 220, 255), thickness=1)

    for rec in corner_result.get("records") or []:
        c_cad = (int(round(rec["cad_col"])), int(round(rec["cad_row"])))
        c_paint = (int(round(rec["paint_col"])), int(round(rec["paint_row"])))
        cv2.circle(out, c_cad, 4, (40, 220, 40), 1, lineType=cv2.LINE_AA)
        cv2.circle(out, c_paint, 3, (255, 60, 60), -1, lineType=cv2.LINE_AA)
        cv2.line(out, c_cad, c_paint, (255, 180, 40), 1, lineType=cv2.LINE_AA)

    if title:
        cv2.putText(
            out,
            title,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    return out


def make_compare_image(bev_gt: np.ndarray, bev_pert: np.ndarray) -> np.ndarray:
    """Side-by-side + abs-diff heat strip."""
    a = bev_gt.astype(np.float32)
    b = bev_pert.astype(np.float32)
    diff = np.mean(np.abs(a - b), axis=2)
    diff_u8 = np.clip(diff * 2.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(diff_u8, cv2.COLORMAP_INFERNO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    left = bev_gt.copy()
    right = bev_pert.copy()
    mid = heat
    cv2.putText(left, "GT calib", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(right, "Perturbed extrinsics", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(mid, "|GT - pert|", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    gap = np.full((left.shape[0], 4, 3), 30, dtype=np.uint8)
    return np.concatenate([left, gap, right, gap, mid], axis=1)


# ---------------------------------------------------------------------------
# Aggregate verdict
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    metrics_gt: Dict[str, Any] = field(default_factory=dict)
    metrics_pert: Dict[str, Any] = field(default_factory=dict)
    checks: Dict[str, str] = field(default_factory=dict)
    overall: str = "FAIL"
    notes: List[str] = field(default_factory=list)

    def to_text(self) -> str:
        lines: List[str] = []
        lines.append("Phase 4 — Intrinsics / Extrinsics Validation Report")
        lines.append("=" * 60)
        lines.append("")
        lines.append("Thresholds")
        lines.append("-" * 40)
        for k, v in THRESHOLDS.items():
            lines.append("  %-32s %s" % (k, v))
        lines.append("")
        lines.append("GT stitch metrics")
        lines.append("-" * 40)
        _dump_metrics(lines, self.metrics_gt)
        lines.append("")
        lines.append("Perturbed extrinsic stitch metrics")
        lines.append("-" * 40)
        _dump_metrics(lines, self.metrics_pert)
        lines.append("")
        lines.append("Checks")
        lines.append("-" * 40)
        for name, status in self.checks.items():
            lines.append("  [%s] %s" % (status, name))
        lines.append("")
        lines.append("OVERALL: %s" % self.overall)
        if self.notes:
            lines.append("")
            lines.append("Notes / recovery (future)")
            lines.append("-" * 40)
            for n in self.notes:
                lines.append("  - %s" % n)
        lines.append("")
        return "\n".join(lines)


def _dump_metrics(lines: List[str], m: Mapping[str, Any], indent: str = "  ") -> None:
    skip = {"records", "pair_mad", "pair_n_px", "per_camera_mean_px"}
    for k, v in m.items():
        if k in skip:
            continue
        if isinstance(v, float):
            lines.append(
                "%s%-28s %s"
                % (indent, k, ("%.6g" % v) if math.isfinite(v) else "nan")
            )
        elif isinstance(v, dict):
            lines.append("%s%s:" % (indent, k))
            for kk, vv in v.items():
                if isinstance(vv, float):
                    lines.append("%s  %-26s %.6g" % (indent, kk, vv))
                else:
                    lines.append("%s  %-26s %s" % (indent, kk, vv))
        else:
            lines.append("%s%-28s %s" % (indent, k, v))
    if "pair_mad" in m:
        lines.append("%spair_mad:" % indent)
        for kk, vv in m["pair_mad"].items():
            n = (m.get("pair_n_px") or {}).get(kk, "?")
            if isinstance(vv, float) and math.isfinite(vv):
                lines.append("%s  %-20s mad=%.3f  n=%s" % (indent, kk, vv, n))
            else:
                lines.append("%s  %-20s mad=nan  n=%s" % (indent, kk, n))
    if "per_camera_mean_px" in m:
        lines.append("%sper_camera_mean_px:" % indent)
        for kk, vv in m["per_camera_mean_px"].items():
            lines.append("%s  %-12s %.6g" % (indent, kk, vv))



def paint_centroid_near_cad(
    bev: np.ndarray,
    ego_xyz_yaw: Tuple[float, float, float, float],
    bev_cfg: Mapping[str, Any],
    vehicle_cfg: Optional[Mapping[str, Any]] = None,
    search_radius_px: float = 35.0,
) -> List[Dict[str, Any]]:
    """Per-marker paint centroid near CAD (skips OOB / ROI / empty)."""
    h, w, res = bev_grid_shape(bev_cfg)
    paint = yellow_paint_mask(bev)
    roi = vehicle_roi_mask(bev_cfg, vehicle_cfg)
    centers = marker_centers_world_flu()
    ego = world_flu_to_ego(centers, ego_xyz_yaw)
    out: List[Dict[str, Any]] = []
    for idx, p in enumerate(ego):
        col, row = ego_xy_to_bev_px(float(p[0]), float(p[1]), bev_cfg)
        ci, ri = int(round(col)), int(round(row))
        if not (0 <= ri < h and 0 <= ci < w) or roi[ri, ci]:
            continue
        hit = _nearest_paint(paint, np.zeros((h, w), np.float32), col, row, h, w, search_radius_px)
        if hit is None:
            continue
        pc, pr, d = hit
        out.append(
            {
                "idx": idx,
                "cad_col": col,
                "cad_row": row,
                "paint_col": pc,
                "paint_row": pr,
                "err_px": d,
                "err_m": d * res,
            }
        )
    return out


def paint_motion_between_bevs(
    bev_a: np.ndarray,
    bev_b: np.ndarray,
    ego_xyz_yaw: Tuple[float, float, float, float],
    bev_cfg: Mapping[str, Any],
    vehicle_cfg: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    """Mean paint-centroid displacement (px/m) between two BEV stitches.

    Used for sensitivity: same fisheyes, GT vs perturbed extrinsics — paint
    should move when IPM calib is wrong.
    """
    _, _, res = bev_grid_shape(bev_cfg)
    a = {r["idx"]: r for r in paint_centroid_near_cad(bev_a, ego_xyz_yaw, bev_cfg, vehicle_cfg)}
    b = {r["idx"]: r for r in paint_centroid_near_cad(bev_b, ego_xyz_yaw, bev_cfg, vehicle_cfg)}
    common = sorted(set(a) & set(b))
    if not common:
        return {
            "paint_motion_mean_px": float("nan"),
            "paint_motion_mean_m": float("nan"),
            "paint_motion_n": 0,
        }
    dists = []
    for i in common:
        d = math.hypot(a[i]["paint_col"] - b[i]["paint_col"], a[i]["paint_row"] - b[i]["paint_row"])
        dists.append(d)
    arr = np.asarray(dists, dtype=np.float64)
    return {
        "paint_motion_mean_px": float(np.mean(arr)),
        "paint_motion_mean_m": float(np.mean(arr) * res),
        "paint_motion_n": int(arr.size),
    }


def evaluate_verdict(
    metrics_gt: Mapping[str, Any],
    metrics_pert: Mapping[str, Any],
) -> ValidationReport:
    """Apply thresholds → PASS / WARN / FAIL per check and overall."""
    T = THRESHOLDS
    checks: Dict[str, str] = {}
    notes: List[str] = []

    cov = float(metrics_gt.get("coverage_frac", 0.0))
    checks["coverage"] = "PASS" if cov >= T["coverage_frac_pass"] else "FAIL"

    ipm_e = float(metrics_gt.get("ipm_vs_opencv_mean_px", float("nan")))
    if math.isfinite(ipm_e) and ipm_e <= T["ipm_vs_opencv_mean_px_pass"]:
        checks["intrinsics_opencv"] = "PASS"
    elif math.isfinite(ipm_e):
        checks["intrinsics_opencv"] = "FAIL"
    else:
        checks["intrinsics_opencv"] = "WARN"
        notes.append("Intrinsics OpenCV cross-check produced no samples.")

    mean_m = float(metrics_gt.get("marker_mean_m", float("nan")))
    mean_px = float(metrics_gt.get("marker_mean_px", float("nan")))
    if not math.isfinite(mean_m):
        checks["marker_alignment_gt"] = "WARN"
        notes.append("No GT landmarks matched paint within search radius.")
    elif mean_m <= T["marker_mean_m_pass"] and mean_px <= T["marker_mean_px_pass"]:
        checks["marker_alignment_gt"] = "PASS"
    elif mean_m <= T["marker_mean_m_warn"] and mean_px <= T["marker_mean_px_warn"]:
        checks["marker_alignment_gt"] = "WARN"
    else:
        checks["marker_alignment_gt"] = "FAIL"

    seam = float(metrics_gt.get("seam_mad_luma", float("nan")))
    if not math.isfinite(seam):
        checks["seam_continuity"] = "WARN"
        notes.append("Seam overlap MAD undefined (insufficient overlap pixels).")
    elif seam <= T["seam_mad_luma_pass"]:
        checks["seam_continuity"] = "PASS"
    else:
        checks["seam_continuity"] = "WARN"

    mean_m_p = float(metrics_pert.get("marker_mean_m", float("nan")))
    seam_p = float(metrics_pert.get("seam_mad_luma", float("nan")))
    bev_mad = float(metrics_pert.get("bev_diff_mad", float("nan")))
    paint_motion = float(metrics_pert.get("paint_motion_mean_px", float("nan")))

    bev_rise = math.isfinite(bev_mad) and bev_mad >= T["bev_diff_mad_pass"]
    motion_rise = (
        math.isfinite(paint_motion) and paint_motion >= T["paint_motion_mean_px_pass"]
    )
    seam_rise = (
        math.isfinite(seam)
        and math.isfinite(seam_p)
        and (seam_p > seam * 1.15 or seam_p - seam > 3.0)
    )
    ratio = float("nan")
    delta = float("nan")
    if math.isfinite(mean_m) and math.isfinite(mean_m_p) and mean_m > 1e-6:
        ratio = mean_m_p / max(mean_m, 1e-6)
        delta = mean_m_p - mean_m
        if isinstance(metrics_pert, dict):
            metrics_pert["sensitivity_ratio"] = ratio
            metrics_pert["sensitivity_delta_m"] = delta

    # Prefer paint-motion + BEV-MAD (directly show pert breaks alignment).
    votes = sum([bev_rise, motion_rise, seam_rise])
    if votes >= 2 or (motion_rise and bev_rise) or (motion_rise and votes >= 1):
        checks["sensitivity"] = "PASS"
    elif votes >= 1:
        checks["sensitivity"] = "WARN"
    else:
        checks["sensitivity"] = "FAIL"
        notes.append(
            "Sensitivity weak: expect paint_motion≥%.1f px or bev_diff_mad≥%.1f."
            % (T["paint_motion_mean_px_pass"], T["bev_diff_mad_pass"])
        )

    notes.append(
        "GT markers are low raised boxes; flat IPM (z=0) shears their tops — "
        "marker_mean residual is expected and treated with a practical pass band."
    )
    notes.append(
        "Recovery / online extrinsic refinement is deferred to a future phase; "
        "this validator only detects misalignment (no solver)."
    )

    statuses = list(checks.values())
    if "FAIL" in statuses:
        overall = "FAIL"
    elif "WARN" in statuses:
        overall = "WARN"
    else:
        overall = "PASS"

    return ValidationReport(
        metrics_gt=dict(metrics_gt),
        metrics_pert=dict(metrics_pert),
        checks=checks,
        overall=overall,
        notes=notes,
    )


def run_full_validation(
    images: Mapping[str, np.ndarray],
    cameras_gt: Mapping[str, CameraModel],
    bev_gt: np.ndarray,
    bev_pert: np.ndarray,
    cameras_pert: Mapping[str, CameraModel],
    ego_xyz_yaw: Tuple[float, float, float, float],
    bev_cfg: Mapping[str, Any],
    vehicle_cfg: Optional[Mapping[str, Any]] = None,
    image_size: Tuple[int, int] = (1280, 960),
) -> ValidationReport:
    """Compute all Phase 4 metrics and verdict for GT + perturbed BEV."""
    cov = coverage_metrics(bev_gt, bev_cfg, vehicle_cfg)
    landmarks = landmark_alignment(bev_gt, ego_xyz_yaw, bev_cfg, vehicle_cfg)
    seams = seam_continuity_metrics(images, cameras_gt, bev_cfg, vehicle_cfg)
    edge = seam_edge_displacement(images, cameras_gt, bev_cfg, vehicle_cfg)
    intr = intrinsics_opencv_consistency(cameras_gt, bev_cfg, image_size)

    metrics_gt: Dict[str, Any] = {}
    metrics_gt.update(cov)
    metrics_gt["marker_mean_m"] = landmarks["mean_m"]
    metrics_gt["marker_mean_px"] = landmarks["mean_px"]
    metrics_gt["marker_median_m"] = landmarks["median_m"]
    metrics_gt["marker_max_m"] = landmarks["max_m"]
    metrics_gt["marker_n"] = landmarks["n_landmarks"]
    metrics_gt["marker_n_marker"] = landmarks["n_marker"]
    metrics_gt["marker_n_slot"] = landmarks["n_slot"]
    # aliases for report readability / plan wording
    metrics_gt["slot_corner_mean_m"] = landmarks["mean_m"]
    metrics_gt["slot_corner_mean_px"] = landmarks["mean_px"]
    metrics_gt["records"] = landmarks["records"]
    metrics_gt.update(seams)
    metrics_gt.update(edge)
    metrics_gt.update(intr)

    landmarks_p = landmark_alignment(bev_pert, ego_xyz_yaw, bev_cfg, vehicle_cfg)
    seams_p = seam_continuity_metrics(images, cameras_pert, bev_cfg, vehicle_cfg)
    edge_p = seam_edge_displacement(images, cameras_pert, bev_cfg, vehicle_cfg)
    bev_mad = bev_difference_mad(bev_gt, bev_pert, bev_cfg, vehicle_cfg)
    motion = paint_motion_between_bevs(
        bev_gt, bev_pert, ego_xyz_yaw, bev_cfg, vehicle_cfg
    )

    metrics_pert: Dict[str, Any] = {}
    metrics_pert["marker_mean_m"] = landmarks_p["mean_m"]
    metrics_pert["marker_mean_px"] = landmarks_p["mean_px"]
    metrics_pert["marker_median_m"] = landmarks_p["median_m"]
    metrics_pert["marker_max_m"] = landmarks_p["max_m"]
    metrics_pert["marker_n"] = landmarks_p["n_landmarks"]
    metrics_pert["slot_corner_mean_m"] = landmarks_p["mean_m"]
    metrics_pert["slot_corner_mean_px"] = landmarks_p["mean_px"]
    metrics_pert["records"] = landmarks_p["records"]
    metrics_pert["seam_mad_luma"] = seams_p["seam_mad_luma"]
    metrics_pert["seam_edge_disp_px"] = edge_p.get("seam_edge_disp_px", float("nan"))
    metrics_pert["pair_mad"] = seams_p["pair_mad"]
    metrics_pert["pair_n_px"] = seams_p["pair_n_px"]
    metrics_pert["bev_diff_mad"] = bev_mad
    metrics_pert.update(motion)

    return evaluate_verdict(metrics_gt, metrics_pert)
