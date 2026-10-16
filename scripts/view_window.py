#!/usr/bin/env python3
"""Open a draggable orbit window on the local X11 display."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from view_scene import main  # noqa: E402


if __name__ == "__main__":
    extra = sys.argv[1:]
    raise SystemExit(main(["--window", *extra]))
