#!/usr/bin/env bash
# Connect to the 8155 board and start its AIS video pipeline.
set -euo pipefail

device=192.168.168.103:5555
adb connect "$device"
adb -s "$device" get-state >/dev/null
exec adb -s "$device" shell su 0 sh /data/local/tmp/ais_video_ctl.sh start
