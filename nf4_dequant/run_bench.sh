#!/bin/bash
set -e
# ./run_bench.sh all 1
MODE=all    # all or single
VERSION=7    # kernel version 1-7
ITERS=100    # benchmark iterations
ROWS=1024
COLS=1024
BS=64    # Block Size（量化块大小）
# Auto-detect GPU arch via Python
ARCH=$(python3 -c "import torch; cap=torch.cuda.get_device_capability(); print(f'sm_{cap[0]}{cap[1]}')")

echo "========================================"
echo " NF4 Dequant Pipeline"
echo " Mode: $MODE | Version: v$VERSION | Iters: $ITERS"
echo "========================================"

# 1. Generate weight data & bnb benchmark (writes bnb_results.json)
echo ""
echo "[1/3] Generating weight data & bnb benchmark..."
rm -rf weight_data bnb_results cuda_results bnb_results.json
if [ "$MODE" = "all" ]; then
    python3 gen_bench.py --mode all
else
    python3 gen_bench.py --mode single --rows $ROWS --cols $COLS --blocksize $BS
fi

# 2. Compile CUDA kernel
echo ""
echo "[2/3] Compiling v$VERSION kernel..."
nvcc -O3 -arch=$ARCH --use_fast_math nf4_dequant_cuda.cu main.cu -o nf4_dequant_cuda

# 3. Run kernel & compare for each weight file
echo ""
echo "[3/3] Running v$VERSION & comparing..."
echo "----------------------------------------"
printf "%-35s %-10s %-10s %-10s %-10s\n" "Weight" "BnB(ms)" "CUDA(ms)" "Speedup" "MAE"
echo "----------------------------------------"

for weight in weight_data/*.bin; do
    [ -f "$weight" ] || continue
    base=$(basename "$weight" .bin)

    # Run CUDA kernel
    output=$(./nf4_dequant_cuda "$weight" $VERSION $ITERS 2>&1)
    cuda_ms=$(echo "$output" | grep -oP 'v\d+: \K[\d.]+')
    cuda_result="cuda_results/v${VERSION}_${base}.bin.fp16"

    # Compare & compute speedup
    result=$(python3 compare_results.py "$cuda_result" "$cuda_ms" 2>/dev/null)
    mae=$(echo "$result" | grep -oP 'MAE: \K[\d.]+')
    speedup=$(echo "$result" | grep -oP 'Speedup: \K[\d.x]+')
    bnb_ms=$(echo "$result" | grep -oP 'BnB: \K[\d.]+')

    printf "%-35s %-10s %-10s %-10s %-10s\n" "$base" "${bnb_ms:--}" "${cuda_ms:--}" "${speedup:--}" "${mae:--}"

    # Accumulate for average
    if [ -n "$bnb_ms" ]; then sum_bnb=$(python3 -c "print(${sum_bnb:-0}+$bnb_ms)"); fi
    if [ -n "$cuda_ms" ]; then sum_cuda=$(python3 -c "print(${sum_cuda:-0}+$cuda_ms)"); fi
    if [ -n "$speedup" ]; then sum_speedup=$(python3 -c "print(${sum_speedup:-0}+${speedup%x})"); fi
    count=$((${count:-0}+1))
done

echo "----------------------------------------"
if [ "$count" -gt 0 ]; then
    avg_bnb=$(python3 -c "print(f'{$sum_bnb/$count:.4f}')")
    avg_cuda=$(python3 -c "print(f'{$sum_cuda/$count:.4f}')")
    avg_speedup=$(python3 -c "print(f'{$sum_speedup/$count:.2f}x')")
    printf "%-35s %-10s %-10s %-10s\n" "AVERAGE" "$avg_bnb" "$avg_cuda" "$avg_speedup"
fi

echo "----------------------------------------"
echo "Done."
