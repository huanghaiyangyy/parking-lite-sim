#!/usr/bin/env python3
"""Forward a local Pygame control window to an existing TX2 X11 window."""
import argparse
import fcntl
import json
import os
import re
import select
import shlex
import signal
import subprocess
import sys
import time

KEYS = ("Up", "Down", "Left", "Right", "space", "1", "2", "3", "4",
        "r", "c", "f", "b")
TIMEOUT = 0.6
BRAKE_RELEASE_DELAY = 2.0


def receive(window):
    os.environ["DISPLAY"] = ":1"
    os.environ.setdefault("XAUTHORITY", "/run/user/%d/gdm/Xauthority" % os.getuid())

    def xdo(*args):
        return subprocess.check_output(["xdotool"] + list(args), timeout=3).decode().strip()

    if not window:
        tree = subprocess.check_output(["xwininfo", "-root", "-tree"], timeout=3).decode()
        windows = re.findall(r'^\s*(0x[0-9a-fA-F]+) "parking-lite [^"]*": '
                             r'\("view_window\.py"', tree, re.M)
        if len(windows) != 1:
            raise RuntimeError("Expected one running TX2 simulator window; found %d" % len(windows))
        window = windows[0]
    window_id = int(window, 0)
    lock = open("/tmp/tx2-remote-keys-%d-%d.lock" % (os.getuid(), window_id), "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    held = set()

    def apply(desired):
        if desired and int(xdo("getwindowfocus")) != window_id:
            raise RuntimeError("TX2 simulator lost keyboard focus; reconnect the controller")
        released = held - desired
        if released:
            xdo("keyup", *sorted(released))
            held.difference_update(released)
        pressed = desired - held
        if pressed:
            # Record before the call so cleanup also handles a partial failure.
            held.update(pressed)
            xdo("keydown", *sorted(pressed))

    def interrupted(signum, frame):
        raise SystemExit(0)

    for sig in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    try:
        xdo("windowactivate", "--sync", str(window_id))
        print("READY", flush=True)
        buffer = b""
        last = time.monotonic()
        while True:
            ready, _, _ = select.select([sys.stdin.fileno()], [], [], 0.1)
            if ready:
                data = os.read(sys.stdin.fileno(), 4096)
                if not data:
                    break
                buffer += data
                if len(buffer) > 8192:
                    raise ValueError("Keyboard packet too large")
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    keys = json.loads(line.decode())
                    if not isinstance(keys, list) or any(k not in KEYS for k in keys):
                        raise ValueError("Invalid keyboard packet")
                    apply(set(keys))
                    last = time.monotonic()
            if held and time.monotonic() - last > TIMEOUT:
                apply(set())
    finally:
        apply(set())
        lock.close()


def control(args):
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import pygame

    remote = ["/usr/bin/python3", args.remote_script, "--receive"]
    if args.window_id:
        remote += ["--window-id", args.window_id]
    ssh = subprocess.Popen(
        ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
         "-o", "ServerAliveInterval=2", "-o", "ServerAliveCountMax=2",
         args.host, " ".join(shlex.quote(v) for v in remote)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True,
        bufsize=1)
    try:
        pygame.init()
        screen = pygame.display.set_mode((650, 260))
        pygame.display.set_caption("TX2 keyboard control - Esc disconnects")
        font = pygame.font.Font(None, 27)
        mapping = {pygame.K_UP: "Up", pygame.K_DOWN: "Down",
                   pygame.K_LEFT: "Left", pygame.K_RIGHT: "Right",
                   pygame.K_SPACE: "space"}
        mapping.update({pygame.key.key_code(k): k for k in KEYS if len(k) == 1})
        held = set()
        brake_until = 0.0
        connected = False
        deadline = time.monotonic() + 15
        last_send = 0.0
        running = True
        clock = pygame.time.Clock()
        while running:
            if ssh.poll() is not None:
                raise RuntimeError("SSH keyboard receiver exited (%d)" % ssh.returncode)
            if not connected:
                readable, _, _ = select.select([ssh.stdout], [], [], 0)
                if readable:
                    if ssh.stdout.readline().strip() != "READY":
                        raise RuntimeError("TX2 keyboard receiver did not become ready")
                    connected = True
                    print("Connected. Click the control window; Esc disconnects.", flush=True)
                elif time.monotonic() > deadline:
                    raise RuntimeError("Timed out waiting for the TX2 window")
            changed = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and
                                                event.key == pygame.K_ESCAPE):
                    running = False
                elif event.type == pygame.WINDOWFOCUSLOST:
                    if "space" in held:
                        brake_until = time.monotonic() + BRAKE_RELEASE_DELAY
                    held.clear()
                    changed = True
                elif event.type in (pygame.KEYDOWN, pygame.KEYUP) and event.key in mapping:
                    if event.type == pygame.KEYDOWN:
                        held.add(mapping[event.key])
                    else:
                        if mapping[event.key] == "space" and "space" in held:
                            brake_until = time.monotonic() + BRAKE_RELEASE_DELAY
                        held.discard(mapping[event.key])
                    changed = True
            now = time.monotonic()
            outgoing = set(held)
            if now < brake_until:
                outgoing.add("space")
            if connected and (changed or now - last_send >= 0.1):
                ssh.stdin.write(json.dumps(sorted(outgoing)) + "\n")
                ssh.stdin.flush()
                last_send = now
            screen.fill((25, 32, 44))
            lines = ["Connected" if connected else "Connecting to " + args.host,
                     "Click here, then hold arrow keys to drive / steer.",
                     "Space: brake + 2s after release    1/2/3/4: P/R/N/D",
                     "R: reset    C: cameras    F: follow    B: odom zero",
                     "Esc: disconnect; TX2 simulator keeps running.",
                     "Held: " + (" + ".join(sorted(outgoing)) or "none") +
                     ("    Brake hold: %.1fs" % (brake_until - now)
                      if "space" not in held and now < brake_until else "")]
            for i, line in enumerate(lines):
                screen.blit(font.render(line, True, (225, 232, 242)), (18, 18 + i * 37))
            pygame.display.flip()
            clock.tick(60)
    finally:
        try:
            if ssh.poll() is None:
                ssh.stdin.write("[]\n")
                ssh.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        try:
            ssh.stdin.close()
            ssh.wait(timeout=3)
        except subprocess.TimeoutExpired:
            ssh.terminate()
            try:
                ssh.wait(timeout=3)
            except subprocess.TimeoutExpired:
                ssh.kill()
                ssh.wait()
        finally:
            pygame.quit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="huang@192.168.168.107")
    parser.add_argument("--remote-script", default="/home/huang/tx2-sim/scripts/remote_keys.py")
    parser.add_argument("--receive", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--window-id", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.receive:
        receive(args.window_id)
    else:
        control(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print("Remote keyboard error: %s" % exc, file=sys.stderr)
        sys.exit(1)
