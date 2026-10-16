# Assets

Procedural geometry is the Phase 1 default. Drop replacements here later:

| Path | Purpose |
|------|---------|
| `lot/lot.obj` or `lot/lot.gltf` | Static parking lot (ground, lines, pillars, walls) |
| `vehicle/ego.obj` or `vehicle/ego.gltf` | Ego body mesh (local origin, +Z forward in GL / +X FLU) |

`scene/ego_vehicle.py` loads `vehicle/J01.obj` when that file exists (J01
sedan extracted from `libavm.so`). Lot geometry is still procedural.

Interactive orbit: `python scripts/view_scene.py --config configs/j01.yaml`
(writes a turntable collage/GIF; a live window needs a working display).
