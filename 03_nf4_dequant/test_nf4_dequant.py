#!/usr/bin/env python3
"""
NF4 Dequantization Test Script
Generates test data and validates results against bitsandbytes reference implementation
"""

import argparse
import os
import struct
import numpy as np
import torch
import math

import bitsandbytes as bnb
from bitsandbytes.functional import quantize_nf4, dequantize_nf4, dequantize_blockwise
from torch.utils.cpp_extension import load

_cuda_modules = {}
def get_cuda_module(platform="nvidia"):
    """Lazy load and cache the CUDA module from external files
    
    Args:
        platform: Target platform, e.g., "nvidia", "metax"
    """
    global _cuda_modules
    
    if platform not in _cuda_modules:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Platform-specific source directory
        src_dir = os.path.join(script_dir, "src", platform)
        if not os.path.exists(src_dir):
            raise ValueError(f"Unsupported platform: {platform}, source directory not found: {src_dir}")
        
        # Collect source files
        sources = []
        for fname in os.listdir(src_dir):
            if fname.endswith('.cpp') or fname.endswith('.cu'):
                sources.append(os.path.join(src_dir, fname))
        
        if not sources:
            raise ValueError(f"No source files found in {src_dir}")
        
        # Platform-specific compile flags
        extra_cflags = ["-O3"]
        extra_cuda_cflags = ["-O3"]
        
        if platform == "nvidia":
            # Let nvcc auto-detect GPU architecture or compile for common architectures
            extra_cuda_cflags.extend([
                "-gencode=arch=compute_70,code=sm_70",
                "-gencode=arch=compute_80,code=sm_80",
                "-gencode=arch=compute_90,code=sm_90"
            ])
        elif platform == "metax":
            # Metax-specific flags (to be added when needed)
            pass
        
        module_name = f"nf4_dequant_{platform}"
        _cuda_modules[platform] = load(
            name=module_name,
            sources=sources,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            verbose=False
        )
    
    return _cuda_modules[platform]

def generate_test_data(args):
    """Generate test quantized weights using bitsandbytes"""
    print(f"Generating test data: {args.rows}x{args.cols}, blocksize={args.blocksize}")
    torch.manual_seed(42)
    weights = torch.randn(args.rows, args.cols, dtype=torch.float32).cuda()
    quantized, quant_state = quantize_nf4(weights, blocksize=args.blocksize, compress_statistics=True)
    packed_weights = quantized.cpu().numpy()
    """
        absmax_q: 一级缩放因子 (per-block), uint8
        absmax2 : 二级缩放因子 (per-group), half
        code2   : 二级码表 [256], half
    """
    absmax_q = quant_state.absmax.cpu().numpy()
    offset = quant_state.offset.item() if quant_state.offset is not None else 0.0
    if args.compute_type == "bf16":
        # 转为 bfloat16 -> 提成 int16 -> 转 numpy 用于保存二进制
        absmax2 = quant_state.state2.absmax.to(torch.bfloat16).view(torch.int16).cpu().numpy()
        code2 = quant_state.state2.code.to(torch.bfloat16).view(torch.int16).cpu().numpy()
    else:
        absmax2 = quant_state.state2.absmax.to(torch.float16).cpu().numpy()
        code2 = quant_state.state2.code.to(torch.float16).cpu().numpy()

    with open(args.weights_path, "wb") as f:
        f.write(struct.pack('<qqi', args.rows, args.cols, args.blocksize))
        f.write(packed_weights.tobytes())
        f.write(absmax_q.tobytes())
        f.write(absmax2.tobytes())
        f.write(code2.tobytes())
        f.write(struct.pack('<f', float(offset)))
    
    with open(args.config_path, "w") as f:
        f.writelines([
            f"blocksize = {args.blocksize}\n",
            f'compute_type = "{args.compute_type}\n',
            f'target_gpu = "{args.target_gpu}\n'
        ])
    
    reference_tensor = dequantize_nf4(quantized, quant_state)
    if args.compute_type == "bf16":
        reference_bf16 = reference_tensor.to(torch.bfloat16).cpu().view(torch.int16).numpy()
        with open(args.ref_output_path, "wb") as f:
            f.write(reference_bf16.tobytes())
    else:
        reference_fp16 = reference_tensor.to(torch.float16).cpu().numpy()
        with open(args.ref_output_path, "wb") as f:
            f.write(reference_fp16.tobytes())
    print(f"Reference output shape: {reference_tensor.shape}")
    print(f"Reference output range:[{reference_tensor.min():.4f}, {reference_tensor.max():.4f}]")

def validate_results(args, threshold=1e-2):
    """Validate CUDA kernel output against bitsandbytes reference"""
    print(f"Validating results ({args.compute_type} mode)...")

    if args.compute_type == "bf16":
        output_raw = np.fromfile(args.kernel_output_path, dtype=np.int16)
        reference_raw = np.fromfile(args.ref_output_path, dtype=np.int16)
        # 借助 PyTorch: int16 -> bfloat16 -> float32，再转回 NumPy 用于计算 MAE 等指标
        output_f32 = torch.from_numpy(output_raw).view(torch.bfloat16).to(torch.float32).numpy()
        reference_f32 = torch.from_numpy(reference_raw).view(torch.bfloat16).to(torch.float32).numpy()
    else: # fp16
        # NumPy 原生支持 float16，直接读取并升格为 float32 计算以防溢出
        output_f32 = np.fromfile(args.kernel_output_path, dtype=np.float16).astype(np.float32)
        reference_f32 = np.fromfile(args.ref_output_path, dtype=np.float16).astype(np.float32)

    if len(reference_f32) != len(output_f32):
        print(f"Error: Size mismatch - reference: {len(reference_f32)}, output: {len(output_f32)}")
        return False

    mae = np.mean(np.abs(reference_f32 - output_f32))
    relative_error = mae / (np.abs(reference_f32).mean() + 1e-8)
    max_error = np.max(np.abs(reference_f32 - output_f32))

    print("\n" + "="*50)
    print("Validation Results")
    print("="*50)
    print(f"Mean Absolute Error (MAE):  {mae:.6f}")
    print(f"Relative Error:             {relative_error:.6f}")
    print(f"Max Absolute Error:         {max_error:.6f}")
    print(f"Reference range:[{reference_f32.min():.4f}, {reference_f32.max():.4f}]")
    print(f"Output range:[{output_f32.min():.4f}, {output_f32.max():.4f}]")
    print("="*50)

    if mae < threshold:
        print(f"\n✓ PASSED: MAE ({mae:.6f}) < threshold ({threshold})")
    else:
        print(f"\n✗ FAILED: MAE ({mae:.6f}) >= threshold ({threshold})")

def dequantize_and_save(args):
    """Dequantize using CUDA kernel and save results"""
    print(f"Dequantizing using CUDA kernel (platform={args.platform})...")
    with open(args.weights_path, "rb") as f:
        num_rows = struct.unpack('<q', f.read(8))[0]
        num_cols = struct.unpack('<q', f.read(8))[0]
        blocksize = struct.unpack('<i', f.read(4))[0]
        
        total_elements = num_rows * num_cols
        packed_size = (total_elements + 1) // 2
        num_blocks = (total_elements + blocksize - 1) // blocksize
        num_groups = (num_blocks + 255) // 256

        packed_weights = np.frombuffer(f.read(packed_size), dtype=np.uint8).copy()
        absmax_q = np.frombuffer(f.read(num_blocks), dtype=np.uint8).copy()
        if args.compute_type == "bf16":
            absmax2 = np.frombuffer(f.read(num_groups * 2), dtype=np.int16).copy()
            code2 = np.frombuffer(f.read(256 * 2), dtype=np.int16).copy()
        else:
            absmax2 = np.frombuffer(f.read(num_groups * 2), dtype=np.float16).copy()
            code2 = np.frombuffer(f.read(256 * 2), dtype=np.float16).copy()
        offset = struct.unpack('<f', f.read(4))[0]
    
    packed_weights_tensor = torch.from_numpy(packed_weights).cuda()
    absmax_q_tensor = torch.from_numpy(absmax_q).cuda()
    log2_blocksize = int(math.log2(blocksize))
    if args.compute_type == "bf16":
        absmax2_tensor = torch.from_numpy(absmax2).cuda().view(torch.bfloat16)
        code2_tensor = torch.from_numpy(code2).cuda().view(torch.bfloat16)
    else:
        absmax2_tensor = torch.from_numpy(absmax2).cuda()
        code2_tensor = torch.from_numpy(code2).cuda()
    
    cuda_module = get_cuda_module(platform=args.platform)
    if args.compute_type == "bf16":
        dequant_func = cuda_module.nf4_dequant_bf16
    else:
        dequant_func = cuda_module.nf4_dequant_fp16

    print("Warming up CUDA kernel...")
    for _ in range(5):
        output_tensor = dequant_func(
            packed_weights_tensor, absmax_q_tensor, absmax2_tensor, code2_tensor,
            total_elements, log2_blocksize, offset
        )
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.test_time):
        output_tensor = dequant_func(
            packed_weights_tensor, absmax_q_tensor, absmax2_tensor, code2_tensor,
            total_elements, log2_blocksize, offset
        )
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    avg_ms_per_iter = elapsed_ms / args.test_time
    print(f"Average execution time: {avg_ms_per_iter:.3f} ms")
    print(f"Throughput: {total_elements / (avg_ms_per_iter * 1e6):.3f} G elements/s")

    if args.compute_type == "bf16":
        output = output_tensor.cpu().to(torch.bfloat16).view(torch.int16).numpy()
    else:
        output = output_tensor.cpu().to(torch.float16).numpy()
    with open(args.kernel_output_path, "wb") as f:
        f.write(output.tobytes())
    
    print(f"Output saved to {args.kernel_output_path}")
    print(f"Output shape: ({num_rows}, {num_cols})")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")

def main():
    parser = argparse.ArgumentParser(description="NF4 Dequantization Test Script")
    parser.add_argument("--generate", action="store_true", help="Generate test data")
    parser.add_argument("--dequantize", action="store_true", help="Dequantize using bitsandbytes")
    parser.add_argument("--validate", action="store_true", help="Validate results")
    parser.add_argument("--test_time", type=int, default=1)
    parser.add_argument("--rows", type=int, default=1024, help="Number of rows (default: 1024)")
    parser.add_argument("--cols", type=int, default=1024, help="Number of columns (default: 1024)")
    parser.add_argument("--blocksize", type=int, default=64, help="Block size (default: 64)")
    parser.add_argument("--weights_path", type=str, default="output/test_weights.bin")
    parser.add_argument("--ref_output_path", type=str, default="output/reference_output.bin")
    parser.add_argument("--kernel_output_path", type=str, default="output/kernel_output.bin")
    parser.add_argument("--config_path", type=str, default="output/test_config.txt")
    parser.add_argument("--compute_type", type=str, default="fp16")
    parser.add_argument("--target_gpu", type=str, default="A100")
    parser.add_argument("--platform", type=str, default="nvidia")

    args = parser.parse_args()
    
    if args.generate:
        generate_test_data(args)
    elif args.dequantize:
        dequantize_and_save(args)
    elif args.validate:
        validate_results(args)

if __name__ == "__main__":
    main()