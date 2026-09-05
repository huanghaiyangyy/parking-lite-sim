"""Static parking-lot geometry (procedural; swappable via assets/ later).

Mesh is built in ModernGL coords (Y-up, +Z forward). Does NOT include the
ego vehicle — that lives in scene.ego_vehicle and is posed dynamically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .mesh_prims import (
    box_mesh,
    ground_plane,
    line_quad,
    parking_slot_lines,
)

# Slot layout constants (GL XZ). Documented so Phase 1 scripts can park in a slot.
SLOT_ORIGIN_X = -10.0
SLOT_ORIGIN_Z_SOUTH = -12.0  # south row (negative Z)
SLOT_ORIGIN_Z_NORTH = 7.0
SLOT_W = 2.5
SLOT_D = 5.0
N_SLOTS = 5

# Phase 4 GT ground markers (GL XZ centers). Bright pads in the aisle so they
# fall inside a typical BEV (±8 m fwd/back, ±6 m left/right) for common ego poses.
# Each marker is a square of side MARKER_SIZE_M on the ground plane.
MARKER_SIZE_M = 0.60
# Placed where surround IPM has good ground coverage (avoid ego under-body and
# forward/aft centerline blind wedges). Coordinates are GL XZ.
MARKER_POSITIONS_GL = (
    (2.5, 2.5),
    (-2.5, 2.5),
    (2.5, -2.5),
    (-2.5, -2.5),
    (3.5, 0.0),
    (-3.5, 0.0),
    (0.0, 3.5),
    (0.0, -3.5),
    (4.0, 2.0),
    (-4.0, 2.0),
    (4.0, -2.0),
    (-4.0, -2.0),
)


def marker_corners_gl(cx: float, cz: float, size: float = MARKER_SIZE_M):
    """Four ground corners (GL) of a marker square centered at (cx, cz)."""
    h = 0.5 * size
    return [
        (cx - h, 0.0, cz - h),
        (cx + h, 0.0, cz - h),
        (cx + h, 0.0, cz + h),
        (cx - h, 0.0, cz + h),
    ]


def slot_center_gl(row: str, index: int) -> tuple[float, float, float]:
    """Return GL (x, y, z) center of a marked parking slot (ground level y=0)."""
    if index < 0 or index >= N_SLOTS:
        raise ValueError("slot index out of range: %s" % index)
    oz = SLOT_ORIGIN_Z_SOUTH if row == "south" else SLOT_ORIGIN_Z_NORTH
    if row not in ("south", "north"):
        raise ValueError("row must be 'south' or 'north', got %r" % row)
    x = SLOT_ORIGIN_X + (index + 0.5) * SLOT_W
    z = oz + 0.5 * SLOT_D
    return (x, 0.0, z)


def build_static_lot(assets_root: Optional[Path] = None) -> np.ndarray:
    """Build interleaved float32 vertex buffer for the static lot.

    If ``assets_root/lot/lot.obj`` (or .gltf) exists in the future, loaders can
    replace this procedural path. For Phase 1 we always generate procedurally
    but keep the hook + directory layout ready.
    """
    _ = assets_root  # reserved for glTF/OBJ swap-in
    verts: list[float] = []

    verts += ground_plane(size=50.0, y=0.0, color=(0.32, 0.34, 0.36))
    verts += parking_slot_lines(
        SLOT_ORIGIN_X, SLOT_ORIGIN_Z_SOUTH, slot_w=SLOT_W, slot_d=SLOT_D, n_slots=N_SLOTS,
        color=(1.0, 0.95, 0.35),
    )
    verts += parking_slot_lines(
        SLOT_ORIGIN_X, SLOT_ORIGIN_Z_NORTH, slot_w=SLOT_W, slot_d=SLOT_D, n_slots=N_SLOTS,
        color=(1.0, 0.95, 0.35),
    )

    # Center aisle dashed yellow line
    for z in np.linspace(-15, 15, 16):
        verts += line_quad(
            -0.08,
            float(z),
            0.08,
            float(z) + 0.7,
            width=0.1,
            color=(0.95, 0.92, 0.3),
        )

    # Pillars / columns
    pillar_color = (0.55, 0.55, 0.58)
    for px, pz in [(-8, -4), (8, -4), (-8, 4), (8, 4), (-12, 0), (12, 0)]:
        verts += box_mesh(px, 1.5, pz, 0.6, 3.0, 0.6, pillar_color)

    # Boundary curbs / low walls
    curb = (0.45, 0.45, 0.48)
    wall = (0.42, 0.44, 0.48)
    verts += box_mesh(0, 0.15, -16, 30, 0.3, 0.4, curb)
    verts += box_mesh(0, 0.15, 16, 30, 0.3, 0.4, curb)
    # Side walls (taller, for depth cues in god-view)
    verts += box_mesh(-18, 1.2, 0, 0.4, 2.4, 34, wall)
    verts += box_mesh(18, 1.2, 0, 0.4, 2.4, 34, wall)

    # Phase 4 GT validation markers — low yellow boxes (visible to side cams;
    # flat pads z-fight into asphalt at grazing fisheye angles).
    marker_color = (1.0, 0.98, 0.05)
    for mx, mz in MARKER_POSITIONS_GL:
        verts += box_mesh(
            mx, 0.06, mz,
            MARKER_SIZE_M, 0.12, MARKER_SIZE_M,
            marker_color,
        )

    # Static parked cars (not ego)
    car_colors = [
        (0.15, 0.35, 0.75),
        (0.7, 0.15, 0.15),
        (0.2, 0.55, 0.25),
        (0.75, 0.6, 0.1),
    ]
    parked = [
        (-8.75, -9.5, car_colors[0]),
        (-6.25, -9.5, car_colors[1]),
        (-3.75, -9.5, car_colors[2]),
        (-1.25, -9.5, car_colors[3]),
        (1.25, -9.5, car_colors[0]),
        (-8.75, 9.5, car_colors[3]),
        (-6.25, 9.5, car_colors[1]),
        (-3.75, 9.5, car_colors[0]),
        (-1.25, 9.5, car_colors[2]),
        (1.25, 9.5, car_colors[1]),
        (-6.0, 2.5, car_colors[2]),
        (6.0, -2.0, car_colors[3]),
    ]
    for cx, cz, col in parked:
        verts += box_mesh(cx, 0.55, cz, 1.8, 1.1, 4.2, col)
        cabin = (col[0] * 0.7, col[1] * 0.7, col[2] * 0.7)
        verts += box_mesh(cx, 1.25, cz - 0.2, 1.6, 0.6, 2.2, cabin)

    return np.asarray(verts, dtype="f4")
