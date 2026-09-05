"""Ego vehicle mesh at local origin (procedural fallback; J01 OBJ if present).

Local GL frame (matches cameras.flu_to_gl at identity pose):
  +Z forward (= vehicle FLU +X), +Y up (= FLU +Z), +X right (= FLU -Y).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .mesh_prims import box_mesh

# Interior parts are huge and only visible with transparent chassis.
_SKIP_GROUPS = frozenset(
    {
        "interior",
        "door_fl_interior",
        "door_fr_interior",
        "door_bl_interior",
        "door_br_interior",
        "trunk_02",
        "trunk_03",
    }
)

def _color_for_group(group: str, mtl: str) -> np.ndarray:
    """Clay colours. Extracted EXT.png is raw ETC1 mis-decoded as RGB — unusable."""
    n = group.lower()
    if "wheel" in n:
        return np.array([0.08, 0.08, 0.09], dtype=np.float32)
    if any(k in n for k in ("glass", "windshield", "window", "sunroof")):
        return np.array([0.11, 0.13, 0.15], dtype=np.float32)
    if "mirror" in n:
        return np.array([0.28, 0.30, 0.32], dtype=np.float32)
    if "brake" in n or n.endswith("_back") or "fog" in n:
        return np.array([0.72, 0.07, 0.07], dtype=np.float32)
    if "reverse" in n:
        return np.array([0.88, 0.88, 0.86], dtype=np.float32)
    if "turn" in n:
        return np.array([0.92, 0.48, 0.06], dtype=np.float32)
    if "light" in n or mtl.startswith("lamp"):
        return np.array([0.82, 0.80, 0.72], dtype=np.float32)
    if mtl == "INT":
        return np.array([0.32, 0.26, 0.24], dtype=np.float32)
    # body / doors / hood / trunk paint — Jetta silver
    return np.array([0.74, 0.76, 0.80], dtype=np.float32)


def load_j01_obj_mesh(
    obj_path: Path,
    origin_forward_offset_m: float = 0.0,
    skip_interior: bool = True,
) -> np.ndarray:
    """Load J01 AVM OBJ → interleaved pos/normal/color float32 in ego GL frame.

    Source coords are millimetres, +X right, +Y forward, +Z up, origin at the
    vehicle geometric centre. We convert to metres and shift +Z so a rear-axle
    origin (``origin_forward_offset_m`` = length/2 − rear_overhang) matches
    ``configs/j01.yaml``.
    """
    obj_path = Path(obj_path)
    positions = []
    normals = []
    # (i0,i1,i2, group, mtl) 0-based
    tris = []
    group = ""
    mtl = "EXT"
    skip = False

    with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            if raw.startswith("v "):
                x, y, z = map(float, raw.split()[1:4])
                positions.append((x, y, z))
            elif raw.startswith("vn "):
                nx, ny, nz = map(float, raw.split()[1:4])
                normals.append((nx, ny, nz))
            elif raw.startswith("g "):
                group = raw.split(None, 1)[1].strip()
                skip = skip_interior and group in _SKIP_GROUPS
            elif raw.startswith("usemtl "):
                mtl = raw.split(None, 1)[1].strip()
            elif raw.startswith("f ") and not skip:
                corners = raw.split()[1:]
                if len(corners) < 3:
                    continue
                idxs = []
                for c in corners:
                    bits = c.split("/")
                    vi = int(bits[0]) - 1
                    ni = int(bits[2]) - 1 if len(bits) > 2 and bits[2] else vi
                    idxs.append((vi, ni))
                for k in range(1, len(idxs) - 1):
                    # reverse winding: AVM (x,y,z)→GL (x,z,y) swaps two axes
                    tris.append((idxs[0], idxs[k + 1], idxs[k], group, mtl))

    pos = np.asarray(positions, dtype=np.float64)
    nrm = np.asarray(normals, dtype=np.float64) if normals else np.zeros_like(pos)
    if len(nrm) != len(pos):
        nrm = np.zeros_like(pos)

    scale = 0.001
    ox = float(origin_forward_offset_m)
    verts = np.empty((len(tris) * 3, 9), dtype=np.float32)
    row = 0
    for a, b, c, grp, mtl_name in tris:
        rgb = _color_for_group(grp, mtl_name)
        for vi, ni in (a, b, c):
            x, y, z = pos[vi]
            nx, ny, nz = nrm[ni]
            # AVM mm: X right, Y forward, Z up → GL m: X right, Y up, Z forward
            px, py, pz = x * scale, z * scale, y * scale + ox
            gx, gy, gz = nx, nz, ny
            ln = (gx * gx + gy * gy + gz * gz) ** 0.5
            if ln > 1e-12:
                gx, gy, gz = gx / ln, gy / ln, gz / ln
            verts[row] = (px, py, pz, gx, gy, gz, rgb[0], rgb[1], rgb[2])
            row += 1
    return verts[:row].reshape(-1)


def build_ego_mesh(
    length_m: float = 4.4,
    width_m: float = 1.9,
    height_m: float = 1.5,
    assets_root: Optional[Path] = None,
    origin_forward_offset_m: float = 0.0,
    wheelbase_m: Optional[float] = None,
) -> np.ndarray:
    """Build ego body facing +Z (GL forward).

    Prefers ``assets/vehicle/J01.obj`` (libavm extract). Falls back to boxes.
    ``origin_forward_offset_m`` is geometric-centre offset along GL +Z.
    """
    if assets_root is not None:
        obj = Path(assets_root) / "vehicle" / "J01.obj"
        if obj.is_file():
            mesh = load_j01_obj_mesh(obj, origin_forward_offset_m=origin_forward_offset_m)
            if mesh.size:
                return mesh

    verts: list[float] = []
    cz = float(origin_forward_offset_m)

    body = (0.12, 0.45, 0.75)
    cabin = (0.25, 0.55, 0.85)
    dark = (0.08, 0.08, 0.1)
    wheel = (0.05, 0.05, 0.05)
    accent = (0.95, 0.85, 0.15)

    body_h = min(0.9, height_m * 0.6)
    cabin_h = min(0.7, height_m * 0.45)
    verts += box_mesh(0.0, body_h * 0.5, cz, width_m, body_h, length_m, body)
    verts += box_mesh(
        0.0,
        body_h + cabin_h * 0.35,
        cz - 0.15,
        width_m * 0.9,
        cabin_h,
        length_m * 0.55,
        cabin,
    )
    half_l = length_m * 0.5
    verts += box_mesh(0.0, 0.35, cz + half_l - 0.05, width_m * 0.9, 0.35, 0.12, dark)
    verts += box_mesh(0.0, 0.35, cz - half_l + 0.05, width_m * 0.9, 0.35, 0.12, dark)
    verts += box_mesh(
        0.0, body_h + 0.02, cz + half_l * 0.55, width_m * 0.35, 0.06, 0.35, accent
    )

    wx = width_m * 0.45
    if wheelbase_m is None:
        wz_front = cz + length_m * 0.32
        wz_rear = cz - length_m * 0.32
    else:
        wz_front = float(wheelbase_m)
        wz_rear = 0.0
    for sx, sz in [(-wx, wz_front), (wx, wz_front), (-wx, wz_rear), (wx, wz_rear)]:
        verts += box_mesh(sx, 0.28, sz, 0.28, 0.55, 0.55, wheel)

    return np.asarray(verts, dtype="f4")
