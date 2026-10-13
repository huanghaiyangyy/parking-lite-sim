#!/usr/bin/env python3
"""Sample stock AVM screen colors and bound latency using Android uptime.

Use the sender's --sequence-pattern --frame-log. These are screen freshness
bounds, including the age within a 10 Hz frame, not first-visible latency.
No vendor application or library instrumentation is needed.
"""
import argparse
import json
import struct
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from stream_h264_adb import sequence_color

PALETTE = np.asarray([sequence_color(i) for i in range(512)], dtype=np.float32)
# Full-size 2560x1440 screen, away from the car / UI / seam overlays.
BEV = dict(front=(380, 390), left=(120, 630), right=(720, 630), rear=(380, 1180))
MAIN = [(x, y) for y in range(360, 1161, 160) for x in range(1100, 2401, 260)]


def split_capture(data):
    first, data = data.split(b"\n", 1)
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("screencap did not return PNG")
    pos = 8
    while pos + 12 <= len(data):
        length = struct.unpack_from(">I", data, pos)[0]
        kind = data[pos+4:pos+8]
        pos += length + 12
        if kind == b"IEND":
            return float(first.split()[0]), data[:pos], float(data[pos:].split()[0])
    raise ValueError("truncated screencap PNG")


def decode_patch(rgb, xy):
    x, y = xy
    patch = np.median(rgb[y-5:y+5, x-5:x+5].reshape(-1, 3), axis=0)
    distances = np.max(np.abs(PALETTE - patch), axis=1)
    index = int(np.argmin(distances))
    return dict(code=index if distances[index] <= 6 else None,
                error=float(distances[index]), rgb=patch.tolist())


def decode_screen(png):
    bgr = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None or bgr.shape != (1440, 2560, 3):
        raise ValueError(f"expected 2560x1440 screen, got {None if bgr is None else bgr.shape}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return decode_rgb(rgb)


def decode_rgb(rgb):
    main = [decode_patch(rgb, xy) for xy in MAIN]
    bev = {name: decode_patch(rgb, xy) for name, xy in BEV.items()}
    codes = [p["code"] for p in main if p["code"] is not None]
    code = max(set(codes), key=codes.count) if codes else None
    return dict(code=code, main=main, bev=bev,
                mixed_main=len(set(codes)) > 1,
                bev_matches=code is not None and all(p["code"] == code for p in bev.values()),
                valid_main=len(codes))


def winscope_times(data):
    magic = b"#VV1NSC0PET1ME!#"
    if not data.startswith(magic):
        raise ValueError("screenrecord Winscope metadata missing")
    count = struct.unpack_from("<I", data, len(magic))[0]
    if len(data) != len(magic)+4+count*8:
        raise ValueError("screenrecord timestamp count/length mismatch")
    stamps = np.frombuffer(data, "<u8", offset=len(magic)+4).astype(np.float64)/1e6
    if count < 2 or np.any(np.diff(stamps) <= 0):
        raise ValueError("screenrecord timestamps must increase")
    return stamps


def analyze_video(args):
    profiles = [json.loads(path.read_text()) for path in args.clock_calibration]
    # For a longer recording, pass RPC profiles from before and after it.
    # The envelope retains uncertainty and covers drift between those bounds.
    offset = [min(p["summary"]["clock_offset_bounds_s"][0] for p in profiles),
              max(p["summary"]["clock_offset_bounds_s"][1] for p in profiles)]
    source = [json.loads(line) for line in args.source_log.read_text().splitlines()]
    by_code = {}
    for s in source:
        if s.get("pattern") == "sequence512":
            by_code.setdefault(s["seq"] % 512, []).append(s)
    metadata = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(args.video),
                                       "-map", "0:d:0", "-f", "data", "pipe:1"])
    stamps = winscope_times(metadata)
    cap = cv2.VideoCapture(str(args.video))
    records, transitions = [], []
    previous = None
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        index = len(records)
        if index >= len(stamps) or bgr.shape != (1440, 2560, 3):
            raise ValueError("video dimensions/frame count do not match metadata")
        decoded = decode_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        bounds = [stamps[index]+o for o in offset]
        record = dict(index=index, capture_bounds_s=bounds, **decoded)
        candidates = [s for s in by_code.get(decoded["code"], [])
                      if s["source_ns"]/1e9 <= bounds[1] and bounds[0]-s["source_ns"]/1e9 < 5]
        if candidates:
            frame = candidates[-1]
            record["source"] = frame
            if previous is not None and previous["source"]["seq"] != frame["seq"]:
                transition = dict(index=index, seq=frame["seq"],
                    first_observed_bounds_ms=[(t-frame["source_ns"]/1e9)*1000 for t in bounds],
                    display_transition_bounds_ms=[
                        max(0, (previous["capture_bounds_s"][0]-frame["source_ns"]/1e9)*1000),
                        (bounds[1]-frame["source_ns"]/1e9)*1000])
                transitions.append(transition)
            previous = record
        elif previous is not None:
            previous = None
        records.append(record)
        if index % 200 == 0:
            print(f"video_frames={index+1}/{len(stamps)} matched={bool(candidates)}", flush=True)
    cap.release()
    if len(records) != len(stamps):
        raise ValueError("decoded video frames do not match metadata count")
    valid = [r for r in records if "source" in r]
    # Bugreport mode begins with a short device-info page. No other unmatched
    # frame is silently accepted.
    first = valid[0]["index"] if valid else len(records)
    unmatched = [r["index"] for r in records[first:] if "source" not in r]
    deltas = np.diff([r["capture_bounds_s"][0] for r in records])
    summary = dict(frames=len(records), matched=len(valid), skipped_info_page_frames=first,
        startup_info_duration_s=float(stamps[first]-stamps[0]) if valid else None,
        duration_s=float(stamps[-1]-stamps[0]), transitions=len(transitions),
        clock_uncertainty_ms=(offset[1]-offset[0])*1000,
        median_recording_interval_ms=float(np.median(deltas)*1000),
        p95_recording_interval_ms=float(np.percentile(deltas, 95)*1000),
        p50_first_observed_bounds_ms=np.percentile([r["first_observed_bounds_ms"] for r in transitions], 50, axis=0).tolist() if transitions else None,
        p95_first_observed_bounds_ms=np.percentile([r["first_observed_bounds_ms"] for r in transitions], 95, axis=0).tolist() if transitions else None,
        p95_display_transition_bounds_ms=np.percentile([r["display_transition_bounds_ms"] for r in transitions], 95, axis=0).tolist() if transitions else None,
        mixed_main_frames=[r["index"] for r in valid if r["mixed_main"]],
        bev_mismatch_frames=[r["index"] for r in valid if not r["bev_matches"]],
        insufficient_main_frames=[r["index"] for r in valid if r["valid_main"] < 20],
        unmatched_frames=unmatched,
        sequence_gaps=sum(b["seq"] > a["seq"]+1 for a, b in zip(transitions, transitions[1:])),
        order_regressions=sum(b["seq"] < a["seq"] for a, b in zip(transitions, transitions[1:])),
        scope="source generation to SurfaceFlinger virtual display; Winscope elapsed timestamps; 36 main + 4 BEV patches; excludes physical panel scanout")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"video_results.json").write_text(json.dumps(dict(summary=summary,
        transitions=transitions, frames=records), indent=2))
    print(json.dumps(summary, indent=2))
    if not transitions or not valid or summary["startup_info_duration_s"] > 1 or unmatched or any(summary[k] for k in
        ("mixed_main_frames", "bev_mismatch_frames", "insufficient_main_frames", "sequence_gaps", "order_regressions")):
        raise SystemExit("FAIL: inspect video_results.json")


def self_test():
    assert len({tuple(c) for c in PALETTE}) == 512
    for seq in range(512):
        rgb = np.empty((1440, 2560, 3), dtype=np.uint8)
        rgb[:] = sequence_color(seq)
        assert decode_patch(rgb, (1200, 600))["code"] == seq
    rgb[:] = sequence_color(111)
    rgb[:720] = sequence_color(110)
    png = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))[1].tobytes()
    start, parsed, end = split_capture(b"100.10 90.00\n" + png + b"100.14 90.02\n")
    assert parsed == png and (start, end) == (100.10, 100.14)
    result = decode_screen(png)
    assert result["mixed_main"] and not result["bev_matches"]
    data = b"#VV1NSC0PET1ME!#" + struct.pack("<IQQ", 2, 12345000, 12445000)
    assert np.allclose(winscope_times(data), [12.345, 12.445])
    try:
        winscope_times(data[:-1])
    except ValueError:
        pass
    else:
        raise AssertionError("truncated timestamp metadata accepted")
    print("PASS: 512 frame codes; PNG/time metadata bounds; mixed frame detected")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--serial", default="5e9350b6")
    p.add_argument("--source-log", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--samples", type=int, default=80)
    p.add_argument("--interval", type=float, default=.3)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--video", type=Path, help="analyze continuous screenrecord with Winscope metadata")
    p.add_argument("--clock-calibration", type=Path, nargs="+",
                   help="results.json near recording; pass before/after profiles to cover clock drift")
    args = p.parse_args()
    cv2.setNumThreads(2)
    if args.self_test:
        self_test(); return
    if not args.source_log or not args.out or args.samples < 1 or args.interval < 0:
        p.error("source-log/out required; samples positive; interval nonnegative")
    if args.video:
        if not args.clock_calibration:
            p.error("video analysis requires clock-calibration")
        analyze_video(args); return
    args.out.mkdir(parents=True, exist_ok=True)

    def adb(command):
        return subprocess.check_output(["adb", "-s", args.serial, "exec-out",
                                       "su 0 sh -c '" + command + "'"], timeout=15)

    # /proc/uptime is truncated to 10 ms. Intersect RPC time bounds, never
    # assume Android's wall clock equals the host clock.
    calibration = []
    for _ in range(24):
        before = time.monotonic_ns()/1e9
        uptime = float(adb("cat /proc/uptime").split()[0])
        after = time.monotonic_ns()/1e9
        calibration.append(dict(before=before, uptime=uptime, after=after))
    offset = [max(c["before"]-c["uptime"]-.01 for c in calibration),
              min(c["after"]-c["uptime"] for c in calibration)]
    if offset[0] > offset[1]:
        raise RuntimeError("Android uptime / host clock bounds inconsistent")
    records = []
    capture_log = (args.out/"samples.jsonl").open("w", buffering=1)
    try:
        for index in range(args.samples):
            tick = time.monotonic()
            before = time.monotonic_ns()/1e9
            a, png, b = split_capture(adb("cat /proc/uptime; screencap -p; cat /proc/uptime"))
            after = time.monotonic_ns()/1e9
            bounds = [max(before, a+offset[0]), min(after, b+.01+offset[1])]
            if bounds[0] > bounds[1]:
                raise RuntimeError("screenshot time bounds inconsistent")
            decoded = decode_screen(png)
            record = dict(index=index, uptime=[a, b], host=[before, after],
                          capture_bounds_s=bounds, **decoded)
            source = [json.loads(line) for line in args.source_log.read_text().splitlines()]
            candidates = [s for s in source if s.get("pattern") == "sequence512"
                          and s["seq"] % 512 == decoded["code"]
                          and s["source_ns"]/1e9 <= bounds[1]
                          and bounds[0]-s["source_ns"]/1e9 < 5]
            if candidates:
                frame = candidates[-1]
                record["source"] = frame
                record["freshness_ms"] = [(t-frame["source_ns"]/1e9)*1000 for t in bounds]
            records.append(record)
            capture_log.write(json.dumps(record) + "\n")
            if index % 20 == 0 or decoded["mixed_main"] or not decoded["bev_matches"]:
                (args.out/f"sample_{index:04d}.png").write_bytes(png)
            if index % 20 == 0:
                print(f"samples={index+1} code={decoded['code']} main={decoded['valid_main']} "
                      f"freshness_ms={record.get('freshness_ms')}", flush=True)
            time.sleep(max(0, args.interval-(time.monotonic()-tick)))
    finally:
        capture_log.close()
    valid = [r for r in records if "freshness_ms" in r]
    distinct = []
    for r in valid:
        if not distinct or r["source"]["seq"] != distinct[-1]:
            distinct.append(r["source"]["seq"])
    summary = dict(samples=len(records), matched=len(valid),
        clock_offset_bounds_s=offset, clock_uncertainty_ms=(offset[1]-offset[0])*1000,
        median_capture_width_ms=float(np.median([(r["capture_bounds_s"][1]-r["capture_bounds_s"][0])*1000 for r in records])),
        p50_freshness_bounds_ms=np.percentile([r["freshness_ms"] for r in valid], 50, axis=0).tolist() if valid else None,
        p95_freshness_bounds_ms=np.percentile([r["freshness_ms"] for r in valid], 95, axis=0).tolist() if valid else None,
        mixed_main_samples=[r["index"] for r in records if r["mixed_main"]],
        bev_mismatch_samples=[r["index"] for r in records if not r["bev_matches"]],
        insufficient_main_samples=[r["index"] for r in records if r["valid_main"] < 20],
        order_regressions=sum(b < a for a, b in zip(distinct, distinct[1:])),
        scope="screen freshness, not first-visible latency; 36 main + 4 BEV patches; 512-frame periodic code")
    (args.out/"results.json").write_text(json.dumps(dict(summary=summary, calibration=calibration), indent=2))
    print(json.dumps(summary, indent=2))
    if len(valid) != len(records) or summary["mixed_main_samples"] or summary["bev_mismatch_samples"] or summary["insufficient_main_samples"] or summary["order_regressions"]:
        raise SystemExit("FAIL: inspect samples.jsonl and saved screenshots")


if __name__ == "__main__":
    main()
