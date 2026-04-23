#!/bin/bash

# 跳过全部 10 次 warmup launch，profile benchmark 阶段的第 1 次调用
# 增加 --launch-count 1，只分析 1 次，节省时间

set -e
export LC_ALL=en_US.UTF-8
export GCONV_PATH=/usr/lib/x86_64-linux-gnu/gconv

WEIGHT=${1:-weight_data/weight_8192x8192_bs128.bin}
OUT=${2:-v1_8192x8192}
VERSION=1
ITERS=10
# Auto-detect GPU arch via Python
ARCH=$(python3 -c "import torch; cap=torch.cuda.get_device_capability(); print(f'sm_{cap[0]}{cap[1]}')")

nvcc -O3 -arch=$ARCH --use_fast_math -lineinfo nf4_dequant_cuda.cu main.cu -o nf4_dequant_cuda
mkdir -p ncu_reports

sudo env LC_ALL=en_US.UTF-8 GCONV_PATH=/usr/lib/x86_64-linux-gnu/gconv /usr/local/cuda/bin/ncu --set full \
    --import-source 1 \
    --source-folders "$(pwd)" \
    --launch-skip 10 \
    --launch-count 1 \
    -f \
    -o "ncu_reports/${OUT}_ncu" \
    ./nf4_dequant_cuda "$WEIGHT" "$VERSION" "$ITERS" 