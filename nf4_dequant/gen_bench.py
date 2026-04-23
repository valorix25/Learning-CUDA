#!/usr/bin/env python3
"""NF4: Generate quantized test data and benchmark bitsandbytes"""

import torch, numpy as np, struct, os, csv, json, argparse
import bitsandbytes.functional as F
from pathlib import Path

SHAPES = [
    (256,256),(512,512),(1024,1024),(2048,2048),
    (4096,4096),(8192,8192),(16384,16384),
    (3421,3146),(6578,1236),(7000,7000),
]
BLOCKSIZES = [64, 128]

def save_weight(filename, rows, cols, blocksize, packed, absmax_q, absmax2, code2, offset):
    with open(filename, 'wb') as f:
        f.write(struct.pack('qqi', rows, cols, blocksize))
        f.write(packed.cpu().numpy().astype(np.uint8).tobytes())
        f.write(absmax_q.cpu().numpy().astype(np.uint8).tobytes())
        f.write(absmax2.cpu().numpy().astype(np.float16).tobytes())
        c2 = code2.cpu().numpy().astype(np.float16)
        if len(c2) < 256:
            c2 = np.pad(c2, (0, 256 - len(c2)))
        f.write(c2[:256].tobytes())
        f.write(struct.pack('f', offset))
    print(f"   saved: {filename} ({os.path.getsize(filename)/1024:.1f} KB)")

def save_bnb_output(filename, tensor):
    tensor.cpu().numpy().astype(np.float16).tofile(filename)
    print(f"   saved: {filename} ({os.path.getsize(filename)/1024:.1f} KB)")

def benchmark_bnb(packed, state, iters=100):
    for _ in range(10):
        F.dequantize_4bit(packed, state)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        F.dequantize_4bit(packed, state)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters

def generate_and_test(rows, cols, blocksize, save_dir="weight_data", bnb_dir="bnb_results"):
    print(f"\n {rows}x{cols} bs={blocksize}")
    Path(save_dir).mkdir(exist_ok=True)
    Path(bnb_dir).mkdir(exist_ok=True)

    weight = torch.randn(rows, cols, device="cuda", dtype=torch.float16)
    packed, state = F.quantize_4bit(weight, blocksize=blocksize, quant_type="nf4", compress_statistics=True)

    save_weight(f"{save_dir}/weight_{rows}x{cols}_bs{blocksize}.bin",
                rows, cols, blocksize, packed, state.absmax.contiguous(),
                state.state2.absmax.to(torch.float16).contiguous(),
                state.state2.code.to(torch.float16).contiguous(), float(state.offset))

    bnb_time = benchmark_bnb(packed, state)
    bnb_out = F.dequantize_4bit(packed, state)
    save_bnb_output(f"{bnb_dir}/bnb_{rows}x{cols}_bs{blocksize}.fp16", bnb_out)
    print(f"   bnb: {bnb_time:.4f} ms")

    return {'shape': f"{rows}x{cols}", 'rows': rows, 'cols': cols,
            'blocksize': blocksize, 'bnb_time_ms': bnb_time, 'total': rows*cols}

def generate_all():
    print("=" * 60 + "\n NF4 数据生成 & BnB 基准测试\n" + "=" * 60)
    results = [generate_and_test(r, c, b) for r, c in SHAPES for b in BLOCKSIZES]

    # Save bnb times to JSON for compare_results.py
    bnb_info = {f"{r['shape']}_bs{r['blocksize']}": r['bnb_time_ms'] for r in results}
    with open("bnb_results.json", "w") as f:
        json.dump(bnb_info, f, indent=2)

    print("\n" + "-" * 60)
    print(f"{'Shape':<12} {'BS':<6} {'BnB(ms)':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['shape']:<12} {r['blocksize']:<6} {r['bnb_time_ms']:<10.4f}")
    print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NF4: generate data & benchmark bnb')
    parser.add_argument('--mode', default='all', choices=['all', 'single'])
    parser.add_argument('--rows', type=int, default=1024)
    parser.add_argument('--cols', type=int, default=1024)
    parser.add_argument('--blocksize', type=int, default=64)
    args = parser.parse_args()

    if args.mode == 'all':
        generate_all()
    else:
        r = generate_and_test(args.rows, args.cols, args.blocksize)
        # Save single result to JSON too
        bnb_info = {f"{r['shape']}_bs{r['blocksize']}": r['bnb_time_ms']}
        with open("bnb_results.json", "w") as f:
            json.dump(bnb_info, f, indent=2)
        print(f"bnb time: {r['bnb_time_ms']:.4f} ms")
