#!/usr/bin/env bash
# One VehicleController drives the overview, four video cameras, and CAN.
set -euo pipefail
sim_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sim_python=/home/huang/tx2-sim-venv/bin/python
export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}"
export SDL_AUDIODRIVER=dummy
export PYGAME_HIDE_SUPPORT_PROMPT=1
cd "$sim_root"
mkdir -p output
exec 9>output/tx2-window.lock
flock -n 9 || { echo 'The TX2 simulation window is already running.' >&2; exit 1; }
serial="$("$sim_python" -c 'import subprocess; listed=subprocess.check_output(["adb","devices"],text=True,timeout=5); found=[line.split()[0] for line in listed.splitlines()[1:] if line.strip().endswith("\tdevice")]; print(found[0] if found else "")')"
if [ -z "$serial" ]; then
  echo 'No ADB device. Window not started.' >&2
  exit 1
fi
echo "ADB device: $serial"
exec "$sim_python" -u scripts/view_window.py \
  --can can0 --stream-adb --odom-adb --serial "$serial" \
  --encoder jetson --fps 10 --bitrate 12M --capture-scale 4 "$@"
