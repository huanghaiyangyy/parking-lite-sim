#!/usr/bin/env python3
"""Collect J05 diagnostics without the rendering window (same relay port)."""
import argparse
from pathlib import Path
import time
from odom_compare import OdomCompare

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--port', type=int, default=46000)
parser.add_argument('--output', type=Path)
args = parser.parse_args()
output = args.output or Path(__file__).resolve().parents[1] / 'output' / 'odom_runs' / time.strftime('%Y%m%d-%H%M%S-diag')
output.mkdir(parents=True, exist_ok=False)
receiver = OdomCompare(None, args.port)
receiver.enable_trace(output / 'odom_rx.csv')
print('Listening on port %d; logs: %s' % (args.port, output), flush=True)
try:
    while True:
        time.sleep(1)
        _, error = receiver.status()
        if error:
            raise RuntimeError(error)
except KeyboardInterrupt:
    pass
finally:
    receiver.close()
