# Ego vehicle mesh

The renderer loads `J01.obj` from this directory when present (Wavefront OBJ
extracted from `libavm.so`). Local checkout uses a symlink to:

`fawvw_apa_env/src/android/android_avm_service/artifacts/j01_model/`

Needed files:

- `J01.obj` / `J01.mtl`
- `tex/EXT.png`, `tex/INT.png`, `tex/lamp_off.png`

Coordinate conversion (done in `scene/ego_vehicle.py`):

- Source: millimetres, +X right, +Y forward, +Z up, geometric centre
- Sim GL: metres, +X right, +Y up, +Z forward, rear-axle origin

If the OBJ is missing, a procedural box body is used instead.
