#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Constant memory LUTs
__constant__ float NF4_LUT[16] = {
    -1.00000000f, -0.69619280f, -0.52507305f, -0.39491710f,
    -0.28444138f, -0.18477343f, -0.09105003f, 0.00000000f,
    0.07958030f,  0.16093020f,  0.24611230f,  0.33791524f,
    0.44070983f,  0.56261700f,  0.72295684f,  1.00000000f};

__constant__ __half NF4_LUT_HALF[16];
__constant__ __half CODE2_LUT[256];

void init_nf4_lut() {
    float lut_f[16] = {-1.00000000f, -0.69619280f, -0.52507305f, -0.39491710f,
                       -0.28444138f, -0.18477343f, -0.09105003f, 0.00000000f,
                       0.07958030f,  0.16093020f,  0.24611230f,  0.33791524f,
                       0.44070983f,  0.56261700f,  0.72295684f,  1.00000000f};
    __half lut_h[16];
    for (int i = 0; i < 16; i++)
        lut_h[i] = __float2half(lut_f[i]);
    CUDA_CHECK(cudaMemcpyToSymbol(NF4_LUT_HALF, lut_h, sizeof(lut_h)));
}

// ============================================================
// v1: float intermediate, 1 byte/thread -> __half2 output
// ============================================================
__global__ void nf4_dequant_v1(
    const uint8_t *__restrict__ packed, const uint8_t *__restrict__ absmax_q,
    const __half *__restrict__ absmax2, const __half *__restrict__ code2,
    __half2 *__restrict__ output,
    int64_t packed_size, int blocksize, int group_size, float offset) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= packed_size)
        return;
    uint8_t val = packed[tid];
    int64_t idx = tid << 1;
    int64_t block_idx = idx / blocksize;
    int64_t group_idx = block_idx / group_size;
    float scale = __half2float(code2[absmax_q[block_idx]]) * __half2float(absmax2[group_idx]) + offset;
    output[tid] = __floats2half2_rn(NF4_LUT[val >> 4] * scale, NF4_LUT[val & 0x0F] * scale);
}

// ============================================================
// v7: v6 + NF4_LUT in shared memory
// ============================================================
__global__ void nf4_dequant_v7(
    const uint8_t *__restrict__ packed, const uint8_t *__restrict__ absmax_q,
    const __half *__restrict__ absmax2, __half *__restrict__ output,
    int64_t total_elements, int blocksize, int group_size, float offset) {
    __shared__ __half s_nf4[16];
    if (threadIdx.x < 16) s_nf4[threadIdx.x] = NF4_LUT_HALF[threadIdx.x];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_bytes = total_elements >> 1;
    int64_t byte_idx = (int64_t)tid * 4;
    if (byte_idx >= total_bytes) 
        return;
    uint32_t pack4 = ((const uint32_t*)packed)[tid];
    uint8_t b[4] = { (uint8_t)(pack4 & 0xFF), (uint8_t)((pack4 >> 8) & 0xFF),
                     (uint8_t)((pack4 >> 16) & 0xFF), (uint8_t)((pack4 >> 24) & 0xFF) };
    int64_t half_base = byte_idx << 1;
    int block_idx = half_base / blocksize;
    int group_idx = block_idx / group_size;
    __half scale = __hadd(__hmul(CODE2_LUT[absmax_q[block_idx]], absmax2[group_idx]), __float2half(offset));
    __half h[8];
    for (int i = 0; i < 4; i++) {
        h[i*2]   = __hmul(s_nf4[b[i] >> 4], scale);
        h[i*2+1] = __hmul(s_nf4[b[i] & 0xF], scale);
    }
    uint4 out_pack;
    for (int i = 0; i < 8; i++) 
        reinterpret_cast<__half*>(&out_pack)[i] = h[i];
    ((uint4*)(output + half_base))[0] = out_pack;
}
