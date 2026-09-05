"""Low-level triangle mesh primitives in ModernGL coords (Y-up, +Z forward).

Vertices are interleaved: position(3) + normal(3) + color(3), float32.
These helpers stay GL-native so lot / ego builders remain readable; FLU↔GL
conversion happens at the pose / camera layer (see sensors.cameras).
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

Color = Tuple[float, float, float]
Vec3 = Tuple[float, float, float]


def box_mesh(
    cx: float,
    cy: float,
    cz: float,
    sx: float,
    sy: float,
    sz: float,
    color: Color,
) -> List[float]:
    """Axis-aligned box centered at (cx,cy,cz) with size (sx,sy,sz)."""
    hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
    faces = [
        ((-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz), (0, 0, 1)),
        ((hx, -hy, -hz), (-hx, -hy, -hz), (-hx, hy, -hz), (hx, hy, -hz), (0, 0, -1)),
        ((-hx, hy, hz), (hx, hy, hz), (hx, hy, -hz), (-hx, hy, -hz), (0, 1, 0)),
        ((-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz), (0, -1, 0)),
        ((hx, -hy, hz), (hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz), (1, 0, 0)),
        ((-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz), (-hx, hy, -hz), (-1, 0, 0)),
    ]
    verts: List[float] = []
    r, g, b = color
    for v0, v1, v2, v3, n in faces:
        for va, vb, vc in ((v0, v1, v2), (v0, v2, v3)):
            for v in (va, vb, vc):
                verts.extend(
                    [cx + v[0], cy + v[1], cz + v[2], n[0], n[1], n[2], r, g, b]
                )
    return verts


def ground_plane(
    size: float = 40.0,
    y: float = 0.0,
    color: Color = (0.22, 0.24, 0.26),
) -> List[float]:
    s = size * 0.5
    n = (0.0, 1.0, 0.0)
    r, g, b = color
    verts: List[float] = []
    for tri in (
        ((-s, y, -s), (s, y, -s), (s, y, s)),
        ((-s, y, -s), (s, y, s), (-s, y, s)),
    ):
        for v in tri:
            verts.extend([v[0], v[1], v[2], n[0], n[1], n[2], r, g, b])
    return verts


def line_quad(
    x0: float,
    z0: float,
    x1: float,
    z1: float,
    y: float = 0.03,
    width: float = 0.22,
    color: Color = (0.98, 0.96, 0.7),
) -> List[float]:
    """Flat strip on XZ plane approximating a painted line."""
    dx, dz = x1 - x0, z1 - z0
    length = math.hypot(dx, dz) or 1e-6
    nx, nz = -dz / length, dx / length
    hw = width * 0.5
    c0 = (x0 + nx * hw, y, z0 + nz * hw)
    c1 = (x0 - nx * hw, y, z0 - nz * hw)
    c2 = (x1 - nx * hw, y, z1 - nz * hw)
    c3 = (x1 + nx * hw, y, z1 + nz * hw)
    n = (0.0, 1.0, 0.0)
    r, g, b = color
    verts: List[float] = []
    for tri in ((c0, c1, c2), (c0, c2, c3)):
        for v in tri:
            verts.extend([v[0], v[1], v[2], n[0], n[1], n[2], r, g, b])
    return verts


def parking_slot_lines(
    ox: float,
    oz: float,
    slot_w: float = 2.5,
    slot_d: float = 5.0,
    n_slots: int = 4,
    color=(0.98, 0.95, 0.55),
) -> list:
    verts: list = []
    kw = dict(y=0.05, width=0.32, color=color)
    for i in range(n_slots + 1):
        x = ox + i * slot_w
        verts += line_quad(x, oz, x, oz + slot_d, **kw)
    verts += line_quad(ox, oz, ox + n_slots * slot_w, oz, **kw)
    verts += line_quad(ox, oz + slot_d, ox + n_slots * slot_w, oz + slot_d, **kw)
    return verts


def append_mesh(dst: List[float], src: Sequence[float]) -> None:
    dst.extend(src)
