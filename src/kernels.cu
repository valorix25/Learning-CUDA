#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

template <typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Layout: [Batch, SeqLen, Heads, HeadDim]
__device__ __forceinline__ int get_offset(int b, int s, int h, int d, int S, int H, int D) {
    return b * (S * H * D) + s * (H * D) + h * D + d;
}

template <typename T>
__global__ void kernel_flashAttention(
    int batch_size, int q_len, int kv_len, int q_heads, int kv_heads, int head_dim,
    bool is_causal, const T* Q, const T* K, const T* V, T* O)
{
    const int tidx = threadIdx.x; // Bc 维度
    const int tidy = threadIdx.y; // Br 维度
    const int bid_h = blockIdx.x; // Q head index
    const int bid_b = blockIdx.y; // Batch index
    const int bid_q = blockIdx.z; // Q block index (Tr)

    const int Br = blockDim.y;
    const int Bc = blockDim.x;
    const int Tc = (kv_len + Bc - 1) / Bc;
    const int g_kv_h = bid_h / (q_heads / kv_heads); // 对应 GQA/MQA 的 KV head
    const float scale = 1.0f / sqrtf((float)head_dim);

    extern __shared__ char smem[];
    T* Q_sm = reinterpret_cast<T*>(smem);                     // [Br * head_dim]
    T* K_sm = reinterpret_cast<T*>(&Q_sm[Br * head_dim]);     // [Bc * head_dim]
    T* V_sm = reinterpret_cast<T*>(&K_sm[Bc * head_dim]);     // [Bc * head_dim]
    // 0–3 bytes padding reserved between V_sm and S_sm to ensure S_sm address is 4-byte aligned
    char* next_ptr = reinterpret_cast<char*>(&V_sm[Bc * head_dim]);
    uintptr_t addr = reinterpret_cast<uintptr_t>(next_ptr);
    addr = (addr + 3) & ~3;

    float* S_sm = reinterpret_cast<float*>(addr);             // [Br * Bc]
    float* O_sm = &S_sm[Br * Bc];                             // [Br * head_dim]
    float* m_prev = &O_sm[Br * head_dim];                     // [Br]
    float* m_new  = &m_prev[Br];                              // [Br]
    float* l_prev = &m_new[Br];                               // [Br]
    float* l_new  = &l_prev[Br];                              // [Br]

    for (int d = tidx; d < head_dim; d += Bc) {
        int q_idx = bid_q * Br + tidy;
        Q_sm[tidy * head_dim + d] = (q_idx < q_len) ? Q[get_offset(bid_b, q_idx, bid_h, d, q_len, q_heads, head_dim)] : (T)0.0f;
        O_sm[tidy * head_dim + d] = 0.0f;
    }
    if (tidx == 0) {
        m_prev[tidy] = -INFINITY;
        l_prev[tidy] = 0.0f;
    }
    __syncthreads();

    for (int j = 0; j < Tc; ++j) {
        // 1. 加载 K, V 到共享内存
        for (int d = tidy; d < head_dim; d += Br) {
            int kv_idx = j * Bc + tidx;
            bool mask = (kv_idx < kv_len);
            K_sm[tidx * head_dim + d] = mask ? K[get_offset(bid_b, kv_idx, g_kv_h, d, kv_len, kv_heads, head_dim)] : (T)0.0f;
            V_sm[tidx * head_dim + d] = mask ? V[get_offset(bid_b, kv_idx, g_kv_h, d, kv_len, kv_heads, head_dim)] : (T)0.0f;
        }
        __syncthreads();

        // 2. 计算 S = (Q @ K.T) * scale
        float score = 0.0f;
        int q_idx_global = bid_q * Br + tidy;
        int k_idx_global = j * Bc + tidx;

        #pragma unroll
        for (int d = 0; d < head_dim; ++d) {
            score += (float)Q_sm[tidy * head_dim + d] * (float)K_sm[tidx * head_dim + d];
        }
        score *= scale;
        
        // Causal Masking 处理
        if (is_causal && q_idx_global < k_idx_global) score = -INFINITY;
        if (q_idx_global >= q_len || k_idx_global >= kv_len) score = -INFINITY;

        S_sm[tidy * Bc + tidx] = score;

        // 3. 计算 RowMax
        float row_max = warp_reduce_max(score);
        if (tidx == 0) {
            m_new[tidy] = fmaxf(m_prev[tidy], row_max);
        }
        __syncthreads();

        // 4. 计算 P = exp(S - m_new) 并求和 (l_new)
        float p_val = expf(S_sm[tidy * Bc + tidx] - m_new[tidy]);
        S_sm[tidy * Bc + tidx] = p_val; // 复用 S_sm 存 P

        float row_sum = warp_reduce_sum(p_val);
        float alpha = expf(m_prev[tidy] - m_new[tidy]);
        if (tidx == 0) {
            l_new[tidy] = alpha * l_prev[tidy] + row_sum;
        }
        __syncthreads();

        // 5. 更新 O_sm: O = O * exp(m_prev - m_new) + P @ V
        for (int d = tidx; d < head_dim; d += Bc) {
            float pv = 0.0f;
            #pragma unroll
            for (int k = 0; k < Bc; ++k) {
                pv += S_sm[tidy * Bc + k] * (float)V_sm[k * head_dim + d];
            }
            O_sm[tidy * head_dim + d] = O_sm[tidy * head_dim + d] * alpha + pv;
        }

        if (tidx == 0) {
            m_prev[tidy] = m_new[tidy];
            l_prev[tidy] = l_new[tidy];
        }
        __syncthreads();
    }

    for (int d = tidx; d < head_dim; d += Bc) {
        int q_idx = bid_q * Br + tidy;
        if (q_idx < q_len) {
            float final_val = O_sm[tidy * head_dim + d] / l_prev[tidy];
            O[get_offset(bid_b, q_idx, bid_h, d, q_len, q_heads, head_dim)] = final_val;
        }
    }
}

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
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  // TODO: Implement the trace function
  return T(-1);
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
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
    // Uses local variables to support parallel multi-call invocation
    cudaStream_t stream;
    RUNTIME_CHECK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, -1));
    
    size_t q_size = h_q.size() * sizeof(T);
    size_t k_size = h_k.size() * sizeof(T);
    size_t v_size = h_v.size() * sizeof(T);
    size_t o_size = h_o.size() * sizeof(T);
    size_t total_bytes = q_size + k_size + v_size + o_size;
    
    T* d_all = nullptr;
    RUNTIME_CHECK(cudaMallocAsync(&d_all, total_bytes, stream));

    T *d_q = d_all;
    T *d_k = d_q + h_q.size();
    T *d_v = d_k + h_k.size();
    T *d_o = d_v + h_v.size();
    
    RUNTIME_CHECK(cudaMemcpyAsync(d_q, h_q.data(), q_size, cudaMemcpyHostToDevice, stream));
    RUNTIME_CHECK(cudaMemcpyAsync(d_k, h_k.data(), k_size, cudaMemcpyHostToDevice, stream));
    RUNTIME_CHECK(cudaMemcpyAsync(d_v, h_v.data(), v_size, cudaMemcpyHostToDevice, stream));
    
    const int Br = 32, Bc = 32;
    dim3 block_dim(Bc, Br);
    dim3 grid_dim(query_heads, batch_size, (target_seq_len + Br - 1) / Br);
    
    size_t smem_size = (Br * head_dim + Bc * head_dim * 2) * sizeof(T) + 
                       (Br * Bc + Br * head_dim + Br * 4 + 2) * sizeof(float); // +2 是为了对齐缓冲

    kernel_flashAttention<T><<<grid_dim, block_dim, smem_size, stream>>>(
        batch_size, target_seq_len, src_seq_len, 
        query_heads, kv_heads, head_dim, is_causal, 
        d_q, d_k, d_v, d_o
    );
    
    RUNTIME_CHECK(cudaMemcpyAsync(h_o.data(), d_o, o_size, cudaMemcpyDeviceToHost, stream));
    RUNTIME_CHECK(cudaFreeAsync(d_all, stream));
    RUNTIME_CHECK(cudaStreamSynchronize(stream));
    RUNTIME_CHECK(cudaStreamDestroy(stream));
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
