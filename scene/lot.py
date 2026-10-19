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
    traffic_cone_mesh,
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


def _color_for_lot_material(mat_name: str) -> np.ndarray:
    n = mat_name.lower()
    if "cone" in n:
        return np.array([0.98, 0.40, 0.05], dtype=np.float32)  # 交通锥亮橙色
    if "yellow" in n:
        return np.array([0.95, 0.85, 0.15], dtype=np.float32)
    if "marking" in n or "line" in n or "paint" in n:
        return np.array([0.95, 0.95, 0.97], dtype=np.float32)
    if "building" in n or "wall" in n:
        return np.array([0.72, 0.70, 0.68], dtype=np.float32)  # 建筑/墙体浅灰
    if "stone" in n or "brick" in n or "siding" in n:
        return np.array([0.62, 0.58, 0.54], dtype=np.float32)  # 砖石墙面
    if "bridge" in n or "pillar" in n or "prop" in n:
        return np.array([0.52, 0.53, 0.56], dtype=np.float32)  # 结构梁/柱
    if "fence" in n or "post" in n:
        return np.array([0.80, 0.82, 0.85], dtype=np.float32)  # 金属围栏柱
    if "curb" in n:
        return np.array([0.62, 0.62, 0.65], dtype=np.float32)
    if "gutter" in n:
        return np.array([0.42, 0.43, 0.45], dtype=np.float32)
    if "sidewalk" in n:
        return np.array([0.68, 0.68, 0.70], dtype=np.float32)
    if "grass" in n or "ground" in n or "terrain" in n:
        return np.array([0.28, 0.45, 0.25], dtype=np.float32)
    if "road" in n or "asphalt" in n:
        return np.array([0.25, 0.26, 0.28], dtype=np.float32)
    return np.array([0.40, 0.40, 0.42], dtype=np.float32)


def load_target_vehicle_template(obj_path: Path):
    positions = []
    normals = []
    tris = []
    usemtl = "body"

    with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            if raw.startswith("v "):
                x, y, z = map(float, raw.split()[1:4])
                positions.append((x, y, z))
            elif raw.startswith("vn "):
                nx, ny, nz = map(float, raw.split()[1:4])
                normals.append((nx, ny, nz))
            elif raw.startswith("usemtl "):
                usemtl = raw.split(None, 1)[1].strip()
            elif raw.startswith("f "):
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
                    tris.append((idxs[0], idxs[k], idxs[k + 1], usemtl))

    pos = np.asarray(positions, dtype=np.float32)
    nrm = np.asarray(normals, dtype=np.float32) if normals else np.zeros_like(pos)
    return pos, nrm, tris


def surface_material_ids(vertices: np.ndarray) -> np.ndarray:
    """Recover surface IDs from the existing Nx9 compiled-lot palette.

    0: unchanged (cars, props), 1: asphalt, 2: wall, 3: concrete,
    4: paint, 5: curb, 6: cone orange, 7: cone reflective tape.
    """
    # ponytail: cached RGB identifies materials; add explicit IDs to the cache
    # format if future assets share colors between unrelated surface types.
    ids = np.zeros((len(vertices), 1), dtype=np.float32)
    for material_id, names in (
        (1, ("asphalt",)),
        (2, ("wall", "brick")),
        (3, ("gutter", "sidewalk", "pillar")),
        (4, ("marking", "yellow")),
        (5, ("curb",)),
        (6, ("cone",)),
    ):
        for name in names:
            match = np.all(np.isclose(vertices[:, 6:9], _color_for_lot_material(name),
                                      rtol=0, atol=1e-5), axis=1)
            ids[match, 0] = material_id
    tape = np.all(np.isclose(vertices[:, 6:9], (0.88, 0.89, 0.86),
                             rtol=0, atol=1e-5), axis=1)
    ids[tape, 0] = 7
    return ids


def add_slot_cones(vertices: np.ndarray) -> np.ndarray:
    """Occupy two empty map bays; leave the left central bay available."""
    import os
    if os.environ.get("LOT_SLOT_CONES", "1") == "0":
        return vertices
    triangles = vertices.reshape(-1, 3, 9)
    road = triangles[np.all(np.isclose(triangles[:, 0, 6:9],
                         _color_for_lot_material("asphalt"), atol=1e-5), axis=1), :, :3]
    cones = []
    for x, z in ((2.8, 0.55), (2.8, 1.55), (-3.7, 7.85), (-3.7, 8.85)):
        # Interpolate the actual road surface so bases neither float nor sink
        # into the elevated road sections in this asset.
        a, b, c = road[:, 0], road[:, 1], road[:, 2]
        den = (b[:, 2]-c[:, 2])*(a[:, 0]-c[:, 0]) + (c[:, 0]-b[:, 0])*(a[:, 2]-c[:, 2])
        valid = np.abs(den) > 1e-9
        a, b, c, den = a[valid], b[valid], c[valid], den[valid]
        u = ((b[:, 2]-c[:, 2])*(x-c[:, 0]) + (c[:, 0]-b[:, 0])*(z-c[:, 2]))/den
        v = ((c[:, 2]-a[:, 2])*(x-c[:, 0]) + (a[:, 0]-c[:, 0])*(z-c[:, 2]))/den
        inside = (u >= -1e-6) & (v >= -1e-6) & (u+v <= 1+1e-6)
        heights = (u*a[:, 1] + v*b[:, 1] + (1-u-v)*c[:, 1])[inside]
        if not len(heights):
            raise ValueError("No road surface beneath slot cone at %s" % ((x, z),))
        cones.extend(traffic_cone_mesh(x, z, float(heights.max()) + 0.001))
    return np.vstack((vertices, np.asarray(cones, dtype=np.float32).reshape(-1, 9)))


def spawn_real_target_cars(template_pos, template_nrm, template_tris, car_instances):
    """car_instances: list of (cx, cz, heading_deg, body_rgb) -> float32 vertex buffer."""
    import math
    all_verts = []
    for cx, cz, heading_deg, rgb in car_instances:
        rad = math.radians(heading_deg)
        cos_h = math.cos(rad)
        sin_h = math.sin(rad)
        n_tris = len(template_tris)
        verts = np.empty((n_tris * 3, 9), dtype=np.float32)
        row = 0
        for i0, i1, i2, mtl in template_tris:
            n = mtl.lower()
            if "glass" in n or "window" in n:
                c = np.array([0.15, 0.17, 0.20], dtype=np.float32)
            elif "wheel" in n or "tire" in n or "interior" in n or "black" in n:
                c = np.array([0.10, 0.10, 0.12], dtype=np.float32)
            elif "light" in n:
                c = np.array([0.88, 0.88, 0.82], dtype=np.float32)
            else:
                c = np.array(rgb, dtype=np.float32)

            for vi, ni in (i0, i1, i2):
                x, y, z = template_pos[vi]
                nx, ny, nz = template_nrm[ni]

                px = x * cos_h + z * sin_h + cx
                py = y
                pz = -x * sin_h + z * cos_h + cz

                gx = nx * cos_h + nz * sin_h
                gy = ny
                gz = -nx * sin_h + nz * cos_h

                ln = math.sqrt(gx * gx + gy * gy + gz * gz)
                if ln > 1e-12:
                    gx /= ln; gy /= ln; gz /= ln

                verts[row] = (px, py, pz, gx, gy, gz, c[0], c[1], c[2])
                row += 1
        all_verts.append(verts)

    return np.vstack(all_verts)


def load_lot_obj_mesh(obj_path: Path, preset: str = "perpendicular") -> np.ndarray:
    """Load static lot OBJ mesh → interleaved pos/normal/color float32 in GL frame.

    Supports two parking scenarios:
      - 'perpendicular' (Road 9): 26 perpendicular/slanted parking slots, centered at (92.74, 19.37)
      - 'parallel' (Road 2): 10 parallel/side parking slots (6.5m x 2.5m), centered at (4.30, -0.88)
    """
    obj_path = Path(obj_path)
    positions = []
    normals = []
    tris = []
    group = ""
    mtl = "default"

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
            elif raw.startswith("usemtl "):
                mtl = raw.split(None, 1)[1].strip()
            elif raw.startswith("f "):
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
                    # Original winding matches CW top-down view in ModernGL
                    tris.append((idxs[0], idxs[k], idxs[k + 1], group, mtl))

    pos = np.asarray(positions, dtype=np.float64)
    nrm = np.asarray(normals, dtype=np.float64) if normals else np.zeros_like(pos)
    if len(nrm) != len(pos):
        nrm = np.zeros_like(pos)

    # Parking scene presets
    preset_lower = str(preset).lower()
    if "map_package" in preset_lower:
        cx, cz = 21.23, -12.18
    elif "parking_lot_3" in preset_lower or "lot3" in preset_lower or "map" in preset_lower:
        cx, cz = 0.0, 6.5
    elif "parallel" in preset_lower or "road2" in preset_lower or "side" in preset_lower:
        cx, cz = 4.30, -0.88
    elif "none" in preset_lower or "raw" in preset_lower:
        cx, cz = 0.0, 0.0
    else:  # default 'perpendicular' (Road 9)
        cx, cz = 92.74, 19.37

    verts = np.empty((len(tris) * 3, 9), dtype=np.float32)
    row = 0
    for a, b, c, grp, mtl_name in tris:
        rgb = _color_for_lot_material(mtl_name)
        # Marking slightly above ground (+3mm) to prevent Z-fighting on flat planes
        y_lift = 0.003 if "marking" in mtl_name.lower() else 0.0
        for vi, ni in (a, b, c):
            x, y, z = pos[vi]
            nx, ny, nz = nrm[ni]
            px, py, pz = x - cx, y + y_lift, z - cz
            gx, gy, gz = nx, ny, nz
            ln = (gx * gx + gy * gy + gz * gz) ** 0.5
            if ln > 1e-12:
                gx, gy, gz = gx / ln, gy / ln, gz / ln
            verts[row] = (px, py, pz, gx, gy, gz, rgb[0], rgb[1], rgb[2])
            row += 1

    # Add real parked target/obstacle cars precisely inside parking slots
    parked_specs = []
    blue = (0.15, 0.35, 0.75)
    red = (0.75, 0.15, 0.15)
    white = (0.88, 0.88, 0.90)
    grey = (0.28, 0.28, 0.30)
    silver = (0.72, 0.74, 0.78)

    if "parking_lot_3" in preset_lower or "lot3" in preset_lower or "map" in preset_lower:
        # Perpendicular slots: slot depth extends along X (center X=3.70 on right, X=-3.70 on left)
        # Slot width along Z with 2.4m spacing. Slot centers: Z = ... -3.85, -1.40, 1.05 (empty for ego), 3.50, 5.90 ...
        # Right slots: heading=-90 (facing aisle)
        # Left slots: heading=90 (facing aisle)
        parked_specs = [
            ( 3.70, -3.85, -90.0, grey),
            ( 3.70, -1.40, -90.0, blue),
            ( 3.70,  3.50, -90.0, red),
            ( 3.70,  5.90, -90.0, white),
            (-3.70, -3.85,  90.0, silver),
            (-3.70, -1.40,  90.0, white),
            (-3.70,  3.50,  90.0, blue),
            (-3.70,  5.90,  90.0, red),
        ]
    elif "parallel" in preset_lower or "road2" in preset_lower or "side" in preset_lower:
        # Parallel slots: length extends along Z axis (6.5m), width along X (2.5m).
        # Center X=0.00, heading=0.0 (facing road forward)
        parked_specs = [
            (0.00,  6.50, 0.0, blue),
            (0.00, -6.50, 0.0, grey),
            (0.00, 13.00, 0.0, white),
        ]
    elif "perpendicular" in preset_lower:
        # round_parking_lot perpendicular slots (Road 9)
        # Depth along X, width along Z (interval 2.55m), centered at Z=0.0
        parked_specs = [
            ( 0.00,  2.55, -90.0, red),
            ( 0.00, -2.55, -90.0, blue),
            ( 0.00, -5.10, -90.0, silver),
        ]

    if parked_specs:
        target_obj = obj_path.parent / "target_tesla.obj"
        if target_obj.is_file():
            t_pos, t_nrm, t_tris = load_target_vehicle_template(target_obj)
            car_arr = spawn_real_target_cars(t_pos, t_nrm, t_tris, parked_specs)
            verts = np.vstack([verts, car_arr])
        else:
            # Fallback
            pass

    return verts


def build_procedural_lot() -> np.ndarray:
    """Build interleaved float32 vertex buffer for procedural lot."""
    verts: list[float] = []

    verts += ground_plane(size=50.0, y=0.0, color=(0.32, 0.34, 0.36))
    slot_paint = (0.93, 0.93, 0.95)
    verts += parking_slot_lines(
        SLOT_ORIGIN_X, SLOT_ORIGIN_Z_SOUTH, slot_w=SLOT_W, slot_d=SLOT_D, n_slots=N_SLOTS,
        color=slot_paint,
    )
    verts += parking_slot_lines(
        SLOT_ORIGIN_X, SLOT_ORIGIN_Z_NORTH, slot_w=SLOT_W, slot_d=SLOT_D, n_slots=N_SLOTS,
        color=slot_paint,
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
        (-3.75, -9.5, car_colors[2]),
        (1.25, -9.5, car_colors[0]),
        (-6.25, 9.5, car_colors[1]),
        (-1.25, 9.5, car_colors[2]),
    ]
    for cx, cz, col in parked:
        verts += box_mesh(cx, 0.55, cz, 1.8, 1.1, 4.2, col)
        cabin = (col[0] * 0.7, col[1] * 0.7, col[2] * 0.7)
        verts += box_mesh(cx, 1.25, cz - 0.2, 1.6, 0.6, 2.2, cabin)

    return np.asarray(verts, dtype="f4")


def compiled_lot_path(obj_path: Path) -> Path:
    return Path(obj_path).with_suffix(".bin")


def save_compiled_lot(path: Path, verts: np.ndarray) -> None:
    data = np.ascontiguousarray(verts, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 9 or len(data) == 0:
        raise ValueError("compiled lot must be Nx9 float vertices")
    path = Path(path)
    path.write_bytes(np.uint32(len(data)).tobytes() + data.tobytes())


def load_compiled_lot(path: Path) -> np.ndarray:
    raw = Path(path).read_bytes()
    count = int(np.frombuffer(raw[:4], dtype=np.uint32)[0])
    data = np.frombuffer(raw[4:], dtype=np.float32)
    if count < 1 or data.size != count * 9:
        raise ValueError("compiled lot %s is truncated" % path)
    return data.reshape(count, 9).copy()


def build_static_lot(assets_root: Optional[Path] = None, preset: Optional[str] = None) -> np.ndarray:
    """Build interleaved float32 vertex buffer for the static lot.

    Supports presets via LOT_PRESET env or preset parameter:
      - 'map' / 'parking_lot_3': 153-slot large parking lot from map.zip (Parking_lot_3)
      - 'parallel': Road 2 parallel/side parking from round_parking_lot.zip
      - 'perpendicular': Road 9 perpendicular parking from round_parking_lot.zip
      - 'map_package': early parking lot map from map_package.zip
    """
    import os
    if preset is None:
        preset = os.environ.get("LOT_PRESET", "map")

    if assets_root is not None:
        p = Path(assets_root)
        preset_lower = str(preset).lower()
        if "parking_lot_3" in preset_lower or "lot3" in preset_lower or "map" in preset_lower and "map_package" not in preset_lower:
            lot_obj = p / "lot" / "lot_parking_lot_3.obj"
            compiled = compiled_lot_path(lot_obj)
            if compiled.is_file() and (not lot_obj.is_file() or compiled.stat().st_mtime >= lot_obj.stat().st_mtime):
                return add_slot_cones(load_compiled_lot(compiled))
            if lot_obj.is_file():
                return add_slot_cones(load_lot_obj_mesh(lot_obj, preset="parking_lot_3"))

        if "map_package" in preset_lower:
            lot_obj = p / "lot" / "lot_map_package.obj"
            if lot_obj.is_file():
                return load_lot_obj_mesh(lot_obj, preset="map_package")

        lot_obj = p / "lot" / "lot.obj"
        if lot_obj.is_file():
            return load_lot_obj_mesh(lot_obj, preset=preset)

    return build_procedural_lot()
