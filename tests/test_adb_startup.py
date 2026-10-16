"""Run with the simulator Python: python tests/test_adb_startup.py."""
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from stream_h264_adb import connect_adb, prepare_adb


with patch("stream_h264_adb.subprocess.run") as run:
    assert prepare_adb("192.168.168.100:5555") == "192.168.168.100:5555"
    assert [call.args[0] for call in run.call_args_list] == [
        ["adb", "connect", "192.168.168.100:5555"],
        ["adb", "-s", "192.168.168.100:5555", "wait-for-device"],
        ["adb", "-s", "192.168.168.100:5555", "forward", "tcp:27164", "tcp:27164"],
    ]
    # Workers reserve stdout for binary previews, including during reconnect.
    assert all(call.kwargs["stdout"] == subprocess.DEVNULL for call in run.call_args_list)
    assert all(call.kwargs["timeout"] == 10 for call in run.call_args_list)

with patch("stream_h264_adb.subprocess.run") as run:
    prepare_adb("5e9350b6")
    assert run.call_count == 2
    assert all("connect" not in call.args[0] for call in run.call_args_list)

def unavailable(cmd, **kwargs):
    if "wait-for-device" in cmd:
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
    return subprocess.CompletedProcess(cmd, 0)

with patch("stream_h264_adb.subprocess.run", side_effect=unavailable) as run, \
        patch("stream_h264_adb.socket.create_connection") as socket:
    try:
        connect_adb("192.168.168.100:5555")
    except RuntimeError as exc:
        assert "not ready" in str(exc)
    else:
        raise AssertionError("unavailable ADB must fail before streaming")
    assert run.call_count == 2
    socket.assert_not_called()

print("ADB startup checks passed")
