#!/bin/bash
set -e

# export CUDA_VISIBLE_DEVICES=0
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
# pip install -q bitsandbytes torch numpy

ROWS=43031
COLS=40279
# ROWS=43
# COLS=40
KERNEL_TYPE="vectorized" # naive vectorized optimized
COMPUTE_TYPE="fp16" # fp16 bf16
PLATFORM="nvidia" # nvidia moore metax iluvatar

python3 test_nf4_dequant.py --generate \
    --rows $ROWS --cols $COLS --blocksize 64 \
    --compute_type $COMPUTE_TYPE

python3 test_nf4_dequant.py --dequantize --compute_type $COMPUTE_TYPE --test_time 2 \
    --launcher_name launcher_$KERNEL_TYPE --platform $PLATFORM --compare_bnb

python3 test_nf4_dequant.py --validate --compute_type $COMPUTE_TYPE




# 增加 --kernel-name 过滤，只抓取你的目标 Kernel
# 增加 --launch-skip 5，跳过 warmup 的前 5 次调用
# 增加 --launch-count 1，只分析正式运行中的第 1 次，节省时间
ncu --set full --kernel-name "regex:nf4_dequant_${KERNEL_TYPE}_kernel.*" \
    --launch-skip 5 --launch-count 1 -o "output/nf4_${KERNEL_TYPE}-$((ROWS*COLS))" -f \
    python3 test_nf4_dequant.py --dequantize \
    --compute_type fp16 --test_time 2 --launcher_name "launcher_${KERNEL_TYPE}"
