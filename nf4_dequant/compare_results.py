#!/usr/bin/env python3
"""Compare CUDA kernel output against bitsandbytes reference, compute speedup"""

import numpy as np
import re
import json
import sys
from pathlib import Path

def parse_shape_bs(filename):
    m = re.search(r'(\d+)x(\d+)_bs(\d+)', filename)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None

def compare(cuda_file, cuda_ms=None, bnb_json="bnb_results.json", bnb_dir="bnb_results"):
    parsed = parse_shape_bs(cuda_file)
    if parsed is None:
        print(f"Cannot parse shape from {cuda_file}")
        return

    rows, cols, bs = parsed
    key = f"{rows}x{cols}_bs{bs}"
    bnb_file = Path(bnb_dir) / f"bnb_{rows}x{cols}_bs{bs}.fp16"
    if not bnb_file.exists():
        print(f"Missing {bnb_file}")
        return

    cuda_out = np.fromfile(cuda_file, dtype=np.float16).astype(np.float32)
    bnb_out = np.fromfile(str(bnb_file), dtype=np.float16).astype(np.float32)

    if len(cuda_out) != len(bnb_out):
        print(f"Size mismatch: cuda={len(cuda_out)} bnb={len(bnb_out)}")
        return

    mae = np.mean(np.abs(bnb_out - cuda_out))
    max_diff = np.max(np.abs(bnb_out - cuda_out))

    # Load bnb time from JSON
    bnb_ms = None
    if Path(bnb_json).exists():
        with open(bnb_json) as f:
            bnb_data = json.load(f)
        bnb_ms = bnb_data.get(key)

    # Compute speedup
    speedup = None
    if bnb_ms and cuda_ms:
        speedup = bnb_ms / cuda_ms

    print(f"MAE: {mae:.8f}, MaxDiff: {max_diff:.8f}, BnB: {bnb_ms:.4f} ms, CUDA: {cuda_ms:.4f} ms, Speedup: {speedup:.2f}x" if speedup
          else f"MAE: {mae:.8f}, MaxDiff: {max_diff:.8f}, BnB: {bnb_ms:.4f} ms" if bnb_ms
          else f"MAE: {mae:.8f}, MaxDiff: {max_diff:.8f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <cuda_output.fp16> [cuda_ms] [bnb_json] [bnb_dir]")
        sys.exit(1)
    compare(sys.argv[1],
            float(sys.argv[2]) if len(sys.argv) > 2 else None,
            sys.argv[3] if len(sys.argv) > 3 else "bnb_results.json",
            sys.argv[4] if len(sys.argv) > 4 else "bnb_results")
