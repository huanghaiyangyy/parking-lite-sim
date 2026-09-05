# Assets

Procedural geometry is the Phase 1 default. Drop replacements here later:

| Path | Purpose |
|------|---------|
| `lot/lot.obj` or `lot/lot.gltf` | Static parking lot (ground, lines, pillars, walls) |
| `vehicle/ego.obj` or `vehicle/ego.gltf` | Ego body mesh (local origin, +Z forward in GL / +X FLU) |

Loaders are not wired yet; `scene/lot.py` and `scene/ego_vehicle.py` accept an
`assets_root` argument so a future glTF/OBJ path can swap in without changing
the World API.
