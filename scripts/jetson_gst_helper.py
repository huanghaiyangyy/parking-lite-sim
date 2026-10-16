#!/usr/bin/env python3
"""Length-prefixed RGBA in, Annex-B AU out. Python 3.6 GI on Jetson TX2.

stdin: raw bottom-up RGBA frames of width*height*4, then EOF.
stdout: uint32 length + AU bytes for each frame, then uint32 0 after EOS.
VIC flips rows so the renderer can hand GL readback over without a CPU copy.
NVENC runs on a side thread with at most two frames queued, so the next GL
readback overlaps encoding. A full queue blocks; compressed frames are not dropped.
"""
import struct
import sys
import threading

try:
    import queue
except ImportError:
    import Queue as queue

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst


def _read_exact(stream, size):
    result = bytearray()
    while len(result) < size:
        block = stream.read(size - len(result))
        if not block:
            raise EOFError
        result.extend(block)
    return bytes(result)


def main():
    width, height, fps, bitrate = (int(arg) for arg in sys.argv[1:5])
    Gst.init(None)
    pipeline = Gst.parse_launch(
        "appsrc name=src is-live=false format=time do-timestamp=false block=true "
        "caps=video/x-raw,format=RGBA,width=%d,height=%d,framerate=%d/1 ! "
        "nvvidconv flip-method=6 ! video/x-raw(memory:NVMM),format=NV12 ! "
        "nvv4l2h264enc bitrate=%d control-rate=1 iframeinterval=%d idrinterval=%d "
        "num-B-Frames=0 insert-sps-pps=true insert-aud=true insert-vui=true "
        "preset-level=1 maxperf-enable=true ! h264parse config-interval=-1 ! "
        "video/x-h264,stream-format=byte-stream,alignment=au ! "
        "appsink name=sink emit-signals=false sync=false max-buffers=8 drop=false"
        % (width, height, fps, bitrate, fps, fps))
    source, sink = pipeline.get_by_name("src"), pipeline.get_by_name("sink")
    if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        raise RuntimeError("Jetson encoder pipeline failed to start")
    frame_bytes = width * height * 4
    frames = queue.Queue(maxsize=2)
    state = {"error": None}

    def encode():
        index = 0
        try:
            while True:
                raw = frames.get()
                if raw is None:
                    break
                buffer = Gst.Buffer.new_allocate(None, frame_bytes, None)
                buffer.fill(0, raw)
                buffer.pts = buffer.dts = index * Gst.SECOND // fps
                buffer.duration = Gst.SECOND // fps
                if source.emit("push-buffer", buffer) != Gst.FlowReturn.OK:
                    raise RuntimeError("Jetson encoder rejected a frame")
                sample = sink.emit("try-pull-sample", 5 * Gst.SECOND)
                if sample is None:
                    raise RuntimeError("Jetson encoder produced no AU for a submitted frame")
                out = sample.get_buffer()
                payload = out.extract_dup(0, out.get_size())
                sys.stdout.buffer.write(struct.pack("<I", len(payload)))
                sys.stdout.buffer.write(payload)
                sys.stdout.buffer.flush()
                index += 1
            source.emit("end-of-stream")
            message = pipeline.get_bus().timed_pop_filtered(
                5 * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
            if message is None or message.type != Gst.MessageType.EOS:
                detail = message.parse_error() if message is not None else "timeout"
                raise RuntimeError("Jetson encoder did not drain: %s" % (detail,))
            if sink.emit("try-pull-sample", 0) is not None:
                raise RuntimeError("Jetson encoder emitted an AU without a submitted frame")
            sys.stdout.buffer.write(struct.pack("<I", 0))
            sys.stdout.buffer.flush()
        except BaseException as exc:
            state["error"] = exc

    worker = threading.Thread(target=encode)
    worker.start()
    try:
        while state["error"] is None:
            frames.put(_read_exact(sys.stdin.buffer, frame_bytes), timeout=10)
    except EOFError:
        frames.put(None, timeout=10)
    worker.join(timeout=15)
    pipeline.set_state(Gst.State.NULL)
    if state["error"] is not None:
        raise state["error"]
    if worker.is_alive():
        raise RuntimeError("Jetson encoder thread did not finish")


if __name__ == "__main__":
    main()
