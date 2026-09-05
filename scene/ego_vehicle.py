"""Ego vehicle mesh at local origin (procedural; swappable via assets/vehicle/).

Local GL frame (matches cameras.flu_to_gl at identity pose):
  +Z forward (= vehicle FLU +X), +Y up (= FLU +Z), +X right (= FLU -Y).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .mesh_prims import box_mesh


def build_ego_mesh(
    length_m: float = 4.4,
    width_m: float = 1.9,
    height_m: float = 1.5,
    assets_root: Optional[Path] = None,
) -> np.ndarray:
    """Build ego body/cabin/wheels centered at local origin, facing +Z (GL forward).

    ``assets_root/vehicle/*.obj|gltf`` reserved for a future loader swap-in.
    """
    _ = assets_root
    verts: list[float] = []

    body = (0.12, 0.45, 0.75)
    cabin = (0.25, 0.55, 0.85)
    dark = (0.08, 0.08, 0.1)
    wheel = (0.05, 0.05, 0.05)
    accent = (0.95, 0.85, 0.15)  # nose stripe — readable in god-view

    body_h = min(0.9, height_m * 0.6)
    cabin_h = min(0.7, height_m * 0.45)
    verts += box_mesh(0.0, body_h * 0.5, 0.0, width_m, body_h, length_m, body)
    verts += box_mesh(
        0.0,
        body_h + cabin_h * 0.35,
        -0.15,
        width_m * 0.9,
        cabin_h,
        length_m * 0.55,
        cabin,
    )
    half_l = length_m * 0.5
    verts += box_mesh(0.0, 0.35, half_l - 0.05, width_m * 0.9, 0.35, 0.12, dark)
    verts += box_mesh(0.0, 0.35, -half_l + 0.05, width_m * 0.9, 0.35, 0.12, dark)
    verts += box_mesh(
        0.0, body_h + 0.02, half_l * 0.55, width_m * 0.35, 0.06, 0.35, accent
    )

    wx = width_m * 0.45
    wz = length_m * 0.32
    for sx, sz in [(-wx, wz), (wx, wz), (-wx, -wz), (wx, -wz)]:
        verts += box_mesh(sx, 0.28, sz, 0.28, 0.55, 0.55, wheel)

    return np.asarray(verts, dtype="f4")
