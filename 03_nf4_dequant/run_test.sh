#!/bin/bash

set -e  # Exit on error
COMPUTE_TYPE="fp16"
PLATFORM="nvidia"
# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# pip install -q bitsandbytes torch numpy

python3 test_nf4_dequant.py --generate \
    --rows 43031 --cols 40279 --blocksize 64 \
    --compute_type $COMPUTE_TYPE
python3 test_nf4_dequant.py --dequantize --compute_type $COMPUTE_TYPE --test_time 2
python3 test_nf4_dequant.py --validate --compute_type $COMPUTE_TYPE

# COMPUTE_TYPE="bf16"
# python3 test_nf4_dequant.py --generate \
#     --rows 23031 --cols 10279 --blocksize 64 \
#     --compute_type $COMPUTE_TYPE
# python3 test_nf4_dequant.py --dequantize --compute_type $COMPUTE_TYPE --test_time 2
# python3 test_nf4_dequant.py --validate --compute_type $COMPUTE_TYPE