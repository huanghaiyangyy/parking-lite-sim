"""AVM package: surround-view stitcher (Phase 3) + validation (Phase 4)."""

from .stitcher import (
    bev_grid_shape,
    draw_slot_overlay_bev,
    stitch_avm,
)
from .validate import (
    THRESHOLDS,
    run_full_validation,
    validate_bev_smoke,
)

__all__ = [
    "stitch_avm",
    "bev_grid_shape",
    "draw_slot_overlay_bev",
    "validate_bev_smoke",
    "run_full_validation",
    "THRESHOLDS",
]
