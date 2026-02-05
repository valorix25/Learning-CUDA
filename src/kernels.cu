#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>
#include <cuda_runtime.h>

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
// 辅助函数：Warp 内归约
template <typename T>
__device__ inline T warp_reduce_sum(T val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// 辅助函数：Block 内归约
template <typename T>
__device__ inline T block_reduce_sum(T val) {
    static __shared__ T shared[32]; // 若 BlockSize 不超过 1024，则最多 32 个 Warps
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    // 1. 每个 Warp 执行归约
    val = warp_reduce_sum(val);

    // 2. 每个 Warp 的第一个线程将结果写入共享内存
    if (lane == 0) shared[wid] = val;
    
    __syncthreads();

    // 3. 由第一个 Warp 读取共享内存并进行最后一次归约
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : T(0);
    if (wid == 0) val = warp_reduce_sum(val);
    
    return val;
}

template <typename T>
__global__ void traceKernel(const T* input, T* result, size_t rows, size_t cols) {
    size_t min_dim = (rows < cols) ? rows : cols; 

    // 获取全局线程索引和总线程数
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = blockDim.x * gridDim.x;

    // 每个线程使用寄存器累加局部结果
    // 对角线元素位置: (i, i)，线性索引为 i * cols + i = i * (cols + 1)
    T thread_sum = T(0);
    for (int i = tid; i < min_dim; i += num_threads) {
        thread_sum += input[i * (cols + 1)]; 
    }

    // 先做 Block 内部归约
    T block_sum = block_reduce_sum(thread_sum);
    // 仅由 Block 的第一个线程将结果原子加到全局内存
    if (threadIdx.x == 0) {
        atomicAdd(result, block_sum);
    }
}

template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
    if (h_input.empty() || rows == 0 || cols == 0) {
        return T(0);
    }
    
    size_t min_dim = std::min(rows, cols);
    
    T* d_input;
    T* d_result;
    cudaMalloc(&d_input, h_input.size() * sizeof(T));
    cudaMalloc(&d_result, sizeof(T));
    
    cudaMemset(d_result, 0, sizeof(T));
    cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(T), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    // 限制 Grid 大小以避免不必要的 Block 开销
    int numBlocks = std::min((size_t)(min_dim + blockSize - 1) / blockSize, (size_t)256);
    
    traceKernel<<<numBlocks, blockSize>>>(d_input, d_result, rows, cols);
    
    T result;
    cudaMemcpy(&result, d_result, sizeof(T), cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_result);
    
    return result;
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */

// Check for fp16 definition, although standard CUDA headers provide it.
using half = __half;
constexpr int BR = 32; // Block size for Query (rows)
constexpr int BC = 32; // Block size for Key/Value (cols)

template<typename T>
struct TypeTraits;

template<>
struct TypeTraits<float> {
    using ComputeType = float;
    static __device__ __forceinline__ float zero() { return 0.0f; }
    static __device__ __forceinline__ float lowest() { return -1e20f; }
    static __device__ __forceinline__ float to_compute(float x) { return x; }
    static __device__ __forceinline__ float from_compute(float x) { return x; }
};

template<>
struct TypeTraits<half> {
    using ComputeType = float;
    static __device__ __forceinline__ half zero() { return __float2half(0.0f); }
    static __device__ __forceinline__ float lowest() { return -1e20f; }
    static __device__ __forceinline__ float to_compute(half x) { return __half2float(x); }
    static __device__ __forceinline__ half from_compute(float x) { return __float2half(x); }
};


template <typename T>
__global__ void flash_attn_kernel(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
    int batch_size,
    int tgt_seq_len,
    int src_seq_len,
    int query_heads,
    int kv_heads,
    int head_dim,
    float sm_scale,
    bool is_causal
) {
    using ComputeType = typename TypeTraits<T>::ComputeType;

    // ---------------------------
    // Shared Memory Layout
    // ---------------------------
    // Dynamic Shared Memory is used.
    // Layout:
    // s_Q: [BR][head_dim] (Size: BR * head_dim * sizeof(T))
    // s_K: [BC][head_dim] (Size: BC * head_dim * sizeof(T))
    // s_V: [BC][head_dim] (Size: BC * head_dim * sizeof(T))
    // s_O: [BR][head_dim] (Size: BR * head_dim * sizeof(ComputeType)) -> Accumulator needs float
    // s_S: [BR][BC]       (Size: BR * BC * sizeof(ComputeType))
    // s_l: [BR]           (Size: BR * sizeof(ComputeType))
    // s_m: [BR]           (Size: BR * sizeof(ComputeType))
    // s_scale: [BR]       (Size: BR * sizeof(ComputeType)) - Temp storage for scaling factor
    
    extern __shared__ char smem[];
    
    // Pointers calculation with alignment
    T* s_Q = reinterpret_cast<T*>(smem);
    T* s_K = s_Q + BR * head_dim;
    T* s_V = s_K + BC * head_dim;
    
    // Floating point buffers start after T buffers
    // Ensure proper alignment for float (4 bytes)
    char* float_start = reinterpret_cast<char*>(s_V + BC * head_dim);
    size_t offset = (size_t)(float_start - smem);
    if (offset % 4 != 0) float_start += (4 - (offset % 4));
    
    ComputeType* s_O = reinterpret_cast<ComputeType*>(float_start);
    ComputeType* s_S = s_O + BR * head_dim;
    ComputeType* s_l = s_S + BR * BC;
    ComputeType* s_m = s_l + BR;
    ComputeType* s_scale = s_m + BR;

    // ---------------------------
    // Indices and Offsets
    // ---------------------------
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Grid Mapping: 
    // x: Q Block Index
    // y: Head Index
    // z: Batch Index
    int bx = blockIdx.x; 
    int by = blockIdx.y; 
    int bz = blockIdx.z;

    int batch_idx = bz;
    int q_head_idx = by;
    int kv_head_idx = q_head_idx / (query_heads / kv_heads); // GQA mapping

    int row_start = bx * BR;
    if (row_start >= tgt_seq_len) return;

    // Global Strides
    long long q_head_stride = head_dim;
    long long q_row_stride = (long long)query_heads * head_dim;
    long long q_batch_stride = (long long)tgt_seq_len * q_row_stride;

    long long kv_head_stride = head_dim;
    long long kv_row_stride = (long long)kv_heads * head_dim;
    long long kv_batch_stride = (long long)src_seq_len * kv_row_stride;

    const T* q_base = Q + batch_idx * q_batch_stride + q_head_idx * q_head_stride;
    const T* k_base = K + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    const T* v_base = V + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    T* o_base = O + batch_idx * q_batch_stride + q_head_idx * q_head_stride;

    // ---------------------------
    // Initialization
    // ---------------------------
    // Init accumulators and stats
    for (int i = ty; i < BR; i += blockDim.y) {
        if (tx == 0) {
            s_l[i] = 0.0f;
            s_m[i] = TypeTraits<T>::lowest();
        }
        for (int d = tx; d < head_dim; d += blockDim.x) {
            s_O[i * head_dim + d] = 0.0f;
        }
    }
    __syncthreads();

    // Load Q Tile
    // s_Q[i][d] <- global Q
    for (int i = ty; i < BR; i += blockDim.y) {
        int r = row_start + i;
        for (int d = tx; d < head_dim; d += blockDim.x) {
            if (r < tgt_seq_len) {
                s_Q[i * head_dim + d] = q_base[r * q_row_stride + d];
            } else {
                s_Q[i * head_dim + d] = TypeTraits<T>::zero();
            }
        }
    }
    __syncthreads();

    // ---------------------------
    // Main Loop over K/V blocks
    // ---------------------------
    for (int j_start = 0; j_start < src_seq_len; j_start += BC) {
        
        // 1. Load K and V Tiles
        for (int i = ty; i < BC; i += blockDim.y) {
            int c = j_start + i;
            for (int d = tx; d < head_dim; d += blockDim.x) {
                if (c < src_seq_len) {
                    s_K[i * head_dim + d] = k_base[c * kv_row_stride + d];
                    s_V[i * head_dim + d] = v_base[c * kv_row_stride + d];
                } else {
                    s_K[i * head_dim + d] = TypeTraits<T>::zero();
                    s_V[i * head_dim + d] = TypeTraits<T>::zero();
                }
            }
        }
        __syncthreads();

        // 2. Compute Scores S = Q * K^T
        // Each thread (ty, tx) computes a subset of elements in S [BR][BC]
        // Parallel strategy: loop over rows i (ty), cols j (tx)
        for (int i = ty; i < BR; i += blockDim.y) {
            for (int j = tx; j < BC; j += blockDim.x) {
                ComputeType sum = 0.0f;
                // Dot product
                for (int d = 0; d < head_dim; ++d) {
                    sum += TypeTraits<T>::to_compute(s_Q[i * head_dim + d]) * 
                           TypeTraits<T>::to_compute(s_K[j * head_dim + d]);
                }
                sum *= sm_scale;
                
                // Masking
                int r_global = row_start + i;
                int c_global = j_start + j;
                if (is_causal && c_global > r_global) {
                    sum = TypeTraits<T>::lowest();
                }
                if (c_global >= src_seq_len) {
                    sum = TypeTraits<T>::lowest();
                }

                s_S[i * BC + j] = sum;
            }
        }
        __syncthreads();

        // 3. Update Softmax Statistics (Row-wise)
        // Only threads with tx=0 (or representative) need to compute stats per row
        // Or parallelize reduction. For simplicity/robustness, one thread per row.
        for (int i = ty; i < BR; i += blockDim.y) {
            if (tx == 0) { // Serial reduction per row for simplicity
                ComputeType m_curr = TypeTraits<T>::lowest();
                for (int j = 0; j < BC; ++j) {
                    m_curr = max(m_curr, s_S[i * BC + j]);
                }
                
                ComputeType m_prev = s_m[i];
                ComputeType m_new = max(m_prev, m_curr);
                
                s_m[i] = m_new;
                
                // Calculate P (exp) and row sum l
                ComputeType l_curr = 0.0f;
                for (int j = 0; j < BC; ++j) {
                    ComputeType p = expf(s_S[i * BC + j] - m_new);
                    s_S[i * BC + j] = p; // Replace score with probability
                    l_curr += p;
                }
                
                // Compute scaling factors
                ComputeType scale_prev = expf(m_prev - m_new);
                s_scale[i] = scale_prev; // Store for O update step
                
                // Update running sum
                s_l[i] = s_l[i] * scale_prev + l_curr;
            }
        }
        __syncthreads();

        // 4. Update Output Accumulator O
        // O = O * scale + P * V
        // Parallel strategy: loop over rows i (ty), dims d (tx)
        for (int i = ty; i < BR; i += blockDim.y) {
            ComputeType scale = s_scale[i];
            for (int d = tx; d < head_dim; d += blockDim.x) {
                // Compute P[i] * V[:][d]
                ComputeType pv_sum = 0.0f;
                for (int j = 0; j < BC; ++j) {
                    pv_sum += s_S[i * BC + j] * TypeTraits<T>::to_compute(s_V[j * head_dim + d]);
                }
                // Update O
                s_O[i * head_dim + d] = s_O[i * head_dim + d] * scale + pv_sum;
            }
        }
        __syncthreads();
    }

    // ---------------------------
    // Final Store
    // ---------------------------
    for (int i = ty; i < BR; i += blockDim.y) {
        int r = row_start + i;
        if (r < tgt_seq_len) {
            ComputeType l_inv = 1.0f / s_l[i];
            for (int d = tx; d < head_dim; d += blockDim.x) {
                ComputeType val = s_O[i * head_dim + d] * l_inv;
                o_base[r * q_row_stride + d] = TypeTraits<T>::from_compute(val);
            }
        }
    }
}

template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
    
    // Calculate total elements
    size_t q_size = h_q.size();
    size_t k_size = h_k.size();
    size_t v_size = h_v.size();
    size_t o_size = q_size; // Output shape matches Query shape

    // Resize output vector
    h_o.resize(o_size);

    // Device Memory Allocation
    T *d_q, *d_k, *d_v, *d_o;
    cudaMalloc(&d_q, q_size * sizeof(T));
    cudaMalloc(&d_k, k_size * sizeof(T));
    cudaMalloc(&d_v, v_size * sizeof(T));
    cudaMalloc(&d_o, o_size * sizeof(T));

    // Copy Inputs
    cudaMemcpy(d_q, h_q.data(), q_size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), k_size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), v_size * sizeof(T), cudaMemcpyHostToDevice);

    // Scaling factor
    float sm_scale = 1.0f / sqrtf((float)head_dim);

    // Launch Config
    // Grid: [Blocks of Q, Heads, Batch]
    dim3 grid((target_seq_len + BR - 1) / BR, query_heads, batch_size);
    // Block: [32, 4] = 128 threads.
    // tx covers dimension chunks or col chunks. ty covers row chunks.
    dim3 block(32, 4); 

    // Dynamic Shared Memory Size
    using ComputeType = typename TypeTraits<T>::ComputeType;
    size_t t_buf_size = (BR * head_dim + 2 * BC * head_dim) * sizeof(T);
    size_t float_buf_size = (BR * head_dim + BR * BC + 3 * BR) * sizeof(ComputeType);
    size_t smem_size = t_buf_size + float_buf_size + 128; // Padding safety

    // Adjust max dynamic shared memory if necessary (A100 supports >48KB)
    cudaFuncSetAttribute(flash_attn_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);

    flash_attn_kernel<T><<<grid, block, smem_size>>>(
        d_q, d_k, d_v, d_o,
        batch_size, target_seq_len, src_seq_len, 
        query_heads, kv_heads, head_dim, 
        sm_scale, is_causal
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }

    cudaMemcpy(h_o.data(), d_o, o_size * sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
