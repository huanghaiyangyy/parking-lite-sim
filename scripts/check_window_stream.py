#!/usr/bin/env python3
"""Check that window previews and complete H264 AUs share the commanded poses."""
import socket
import argparse
import os
import subprocess
import sys
import threading
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scene.world import World
from stream_h264_adb import HEADER, SEND_STATUS, _read_exact, _nal_types
from view_scene import _GodViewProxy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-scale", type=int, choices=(1, 2, 4), default=4)
    args = parser.parse_args()
    config = ROOT / "configs/j01.yaml"
    chunks, previews, captures = [], [], []
    poses = [(0.5+i*0.2, 1.5, 0.0, 0.488692) for i in range(8)]
    with tempfile.TemporaryDirectory() as tmp, socket.socket() as server:
        # Exercise the actual adb connection path without a board. A noisy adb
        # forward must not replace the worker's "ready" protocol greeting.
        adb = Path(tmp) / "adb"
        adb.write_text(f"#!{sys.executable}\nimport sys\nprint(sys.argv[-1].split(':')[1])\n")
        adb.chmod(0o700)
        env = dict(os.environ, PATH=tmp+os.pathsep+os.environ.get("PATH", ""))
        server.bind(("127.0.0.1", 0)); server.listen(1); server.settimeout(20)
        proc = subprocess.Popen([sys.executable, str(ROOT/"scripts/view_render_worker.py"),
            "--stream-adb", "--serial", "self-test", "--port", str(server.getsockname()[1]),
            "--capture-scale", str(args.capture_scale)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
        try:
            proc.stdin.write(f"cams\n{config}\n".encode()); proc.stdin.flush()
            peer, _ = server.accept()
            def receive():
                with peer:
                    while block := peer.recv(65536):
                        chunks.append(block)
            reader = threading.Thread(target=receive)
            reader.start()
            ready = proc.stdout.readline().decode().split()
            assert ready[0] == "ready", ready
            names = ready[1:]
            proc.stdin.write((" ".join(map(str, poses[0]))+"\n").encode()); proc.stdin.flush()
            for _ in names:
                _GodViewProxy._read_frame(proc)
            _read_exact(proc.stdout, SEND_STATUS.size)
            time.sleep(2)  # Exclude hardware encoder cold start from steady fps.
            for pose in poses:
                proc.stdin.write((" ".join(map(str, pose))+"\n").encode()); proc.stdin.flush()
                previews.append({name: _GodViewProxy._read_frame(proc) for name in names})
                sent, fps, mbps, source_seq, *capture_pose = SEND_STATUS.unpack(_read_exact(proc.stdout, SEND_STATUS.size))
                captures.append((source_seq, capture_pose))
                time.sleep(0.2)  # A slow preview consumer must not slow board video.
            assert sent > len(poses) and 9 <= fps <= 11 and mbps > 0, (sent, fps, mbps)
            proc.stdin.write(b"quit\n"); proc.stdin.flush()
            assert proc.wait(timeout=20) == 0
            reader.join(timeout=5); assert not reader.is_alive()
        finally:
            if proc.poll() is None:
                proc.kill(); proc.wait()
    data, offset, aus, pts = b"".join(chunks), 0, [], []
    while offset < len(data):
        h = HEADER.unpack_from(data, offset); offset += HEADER.size
        assert h[:3] == (b"PLIT", 2, 1) and h[6] == len(aus) and h[9:] == (2560, 1920)
        au = data[offset:offset+h[8]]; offset += h[8]
        assert len(au) == h[8] and _nal_types(au) & {1, 5}
        aus.append(au); pts.append(h[7])
    assert len(aus) > len(poses) and all(a < b for a, b in zip(pts, pts[1:]))
    steady_pts = [p for p in pts if p >= 2000000]
    actual_fps = (len(steady_pts)-1)*1e6/(steady_pts[-1]-steady_pts[0])
    assert 9.8 <= actual_fps <= 10.2, actual_fps
    decoded = subprocess.check_output(["ffmpeg", "-v", "error", "-threads", "2", "-f", "h264",
        "-i", "pipe:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], input=b"".join(aus))
    frames = np.frombuffer(decoded, np.uint8).reshape(len(aus), 1920, 2560, 3)
    world = World.from_config(config)
    world.rig.render_cfg["pinhole_capture_scale"] = args.capture_scale
    try:
        for i, (source_seq, pose) in enumerate(captures):
            world.set_ego_pose(*pose); grid = world.render_camera_grid()
            for name, (row, col) in dict(front=(0, 0), left=(0, 1), right=(1, 0), rear=(1, 1)).items():
                expected = cv2.resize(grid[row*960:(row+1)*960, col*1280:(col+1)*1280],
                                      (640, 480), interpolation=cv2.INTER_AREA)
                assert np.array_equal(previews[i][name], expected), (i, name)
            mae = np.abs(frames[source_seq].astype(np.int16)-grid.astype(np.int16)).mean()
            assert mae < 6, (i, mae)
        assert not np.array_equal(previews[0]["front"], previews[-1]["front"])
    finally:
        world.release()
    print(f"PASS: matched preview source sequences/poses; slow preview consumer; video {actual_fps:.2f} fps")


if __name__ == "__main__":
    main()
