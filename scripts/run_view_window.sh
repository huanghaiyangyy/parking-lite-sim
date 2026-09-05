#!/usr/bin/env bash
# GNOME on this machine is DISPLAY=:1 (not :0).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export DISPLAY="${DISPLAY:-:1}"
if [[ -z "${XAUTHORITY:-}" ]]; then
  if [[ -f "/run/user/$(id -u)/gdm/Xauthority" ]]; then
    export XAUTHORITY="/run/user/$(id -u)/gdm/Xauthority"
  elif [[ -f "$HOME/.Xauthority" ]]; then
    export XAUTHORITY="$HOME/.Xauthority"
  fi
fi
cd "$ROOT"
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/view_window.py" "$@"
