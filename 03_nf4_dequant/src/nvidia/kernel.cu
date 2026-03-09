#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <torch/extension.h>

// NF4 lookup table
__device__ __constant__ float NF4_LUT[16] = {
    -1.0f,                 // 0b0000
    -0.6961928009986877f,  // 0b0001
    -0.5250730514526367f,  // 0b0010
    -0.39491748809814453f, // 0b0011
    -0.28444138169288635f, // 0b0100
    -0.18477343022823334f, // 0b0101
    -0.09105003625154495f, // 0b0110
    0.0f,                  // 0b0111
    0.07958029955625534f,  // 0b1000
    0.16093020141124725f,  // 0b1001
    0.24611230194568634f,  // 0b1010
    0.33791524171829224f,  // 0b1011
    0.44070982933044434f,  // 0b1100
    0.5626170039176941f,   // 0b1101
    0.7229568362236023f,   // 0b1110
    1.0f                   // 0b1111
};

template<typename T> struct TypeTraits {};
template<> struct TypeTraits<__half> {
    using type2 = __half2;
    static __device__ __forceinline__ __half2 pack(float a, float b) { return __floats2half2_rn(a, b); }
    static __device__ __forceinline__ __half cast_from_float(float a) { return __float2half(a); }
    static __device__ __forceinline__ float cast_to_float(__half a) { return __half2float(a); }
};
template<> struct TypeTraits<__nv_bfloat16> {
    using type2 = __nv_bfloat162;
    static __device__ __forceinline__ __nv_bfloat162 pack(float a, float b) { return __floats2bfloat162_rn(a, b); }
    static __device__ __forceinline__ __nv_bfloat16 cast_from_float(float a) { return __float2bfloat16(a); }
    static __device__ __forceinline__ float cast_to_float(__nv_bfloat16 a) { return __bfloat162float(a); }
};

// Formula: output = nf4_lut[idx] * (code2[absmax_q] * absmax2[group] + offset)
template<typename T>
__global__ void nf4_dequant_naive_kernel(
    const uint8_t* __restrict__ packed_weights, const uint8_t* __restrict__ absmax_q,
    const T* __restrict__ absmax2, const T* __restrict__ code2, T* __restrict__ output,
    int64_t total_elements, int32_t log2_blocksize, float offset
) {

    // 每个线程处理 2 个元素（对应 1 个 packed uint8）
    int64_t global_idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 2;
    if (global_idx >= total_elements) 
        return;
    
    // 1. 解包 NF4 索引 (4-bit x 2)
    uint8_t packed = packed_weights[global_idx / 2];
    uint8_t idx_even = (packed >> 4) & 0x0F;
    uint8_t idx_odd = packed & 0x0F;
    
    // 2. 计算量化块索引和二级量化组索引
    int64_t block_idx = global_idx >> log2_blocksize;
    int64_t group_idx = block_idx >> 8;  // 假设 group size 是 256
    
    // 3. 计算双重解量化的 absmax 值
    uint8_t absmax_idx = absmax_q[block_idx];
    
    // 使用 Trait 进行类型转换
    float c2 = TypeTraits<T>::cast_to_float(code2[absmax_idx]);
    float a2 = TypeTraits<T>::cast_to_float(absmax2[group_idx]);
    float absmax = c2 * a2 + offset;
    
    // 4. 应用 NF4 查找表并写回结果
    output[global_idx] = TypeTraits<T>::cast_from_float(NF4_LUT[idx_even] * absmax);
    
    if (global_idx + 1 < total_elements) {
        output[global_idx + 1] = TypeTraits<T>::cast_from_float(NF4_LUT[idx_odd] * absmax);
    }
}

template <typename T>
__global__ void nf4_dequant_vectorized_kernel(
    const uint8_t* __restrict__ packed_weights, const uint8_t* __restrict__ absmax_q,
    const T* __restrict__ absmax2, const T* __restrict__ code2, T* __restrict__ output,
    int64_t total_elements, int32_t log2_blocksize, float offset
) {

    //     Constant Memory (常量内存)
    // 物理位置： 存储在显存中，但有专门的常量缓存（Constant Cache）。

    // 特点： 64KB 大小限制，只读。

    // 核心优势： 广播（Broadcast）机制。如果 Warp 内所有线程访问同一个地址，速度极快；如果访问不同地址，会发生串行化。

    // 虽然 __constant__ 带有缓存，但它有一个物理限制：单次读取只能广播一个地址。
    // 广播机制： 如果一个 Warp 内的所有线程访问同一个常量地址，速度极快。
    // 串行化： 在 NF4 反量化中，每个线程处理的权重索引（0-15）几乎肯定是不同的。当 Warp 内线程访问不同地址时，__constant__ 的访问会被串行化（Serialized），导致严重的延迟。

    // Shared Memory 允许 Warp 内的线程同时访问不同的Bank
    // 1. 将 LUT 加载到 Shared Memory 避免 Constant Memory Bank 冲突
    __shared__ float shared_lut[16];
    // 如果你定义一个局部数组 float local_lut[16] = {...}，编译器通常有两种处理方式：
    // 放入寄存器： 每个线程都会占用 16 个寄存器来存这张表。在 CUDA 中，寄存器是极其珍贵的资源。寄存器占用过多会导致 Occupancy（活跃线程束比例）下降，反而拖慢整体算子速度。
    // 溢出到 Local Memory： 如果寄存器不够，编译器会将数组放到 Local Memory（本质上是显存），这会导致性能大幅暴跌。
    if (threadIdx.x < 16) {
        shared_lut[threadIdx.x] = NF4_LUT[threadIdx.x];
    }
    __syncthreads();

    // 每个线程处理 8 个元素 (即 4 个 Byte，读取一次 uint32_t)
    int64_t vec_global_idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
    
    // Fast path: 能够完整凑齐 8 个元素的线程
    if (vec_global_idx + 7 < total_elements) {
        // 向量化读取：1 次 32-bit 加载 (包含 8 个元素)
        const uint32_t* packed_weights_u32 = reinterpret_cast<const uint32_t*>(packed_weights);
        uint32_t packed = packed_weights_u32[vec_global_idx / 8];
        
        uint32_t out_packed[4]; // 存放 4 个 half2 / bf162
        
        int64_t last_block_idx = -1;
        float absmax = 0.0f;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            // 提取 1 个 byte
            uint8_t byte_val = (packed >> (i * 8)) & 0xFF;
            uint8_t idx_even = (byte_val >> 4) & 0x0F;
            uint8_t idx_odd  = byte_val & 0x0F;

            // 计算当前所在的量化块
            int64_t current_idx = vec_global_idx + i * 2;
            int64_t block_idx = current_idx >> log2_blocksize;
            
            // 消除冗余计算：只有跨越 block 边界时才重新计算 absmax
            if (block_idx != last_block_idx) {
                int64_t group_idx = block_idx >> 8;
                uint8_t absmax_idx = absmax_q[block_idx];
                
                // 修复点：使用泛型将其自动通过对应指令转回 Float 以便运算
                float code_val = TypeTraits<T>::cast_to_float(code2[absmax_idx]);
                float absmax_val = TypeTraits<T>::cast_to_float(absmax2[group_idx]);
                
                absmax = code_val * absmax_val + offset;
                last_block_idx = block_idx;
            }

            // LUT 查找与乘法
            float val_even = shared_lut[idx_even] * absmax;
            float val_odd  = shared_lut[idx_odd]  * absmax;

            // 打包成 half2 / bfloat162，存储在 32-bit 寄存器中
            typename TypeTraits<T>::type2 h2 = TypeTraits<T>::pack(val_even, val_odd);
            out_packed[i] = *reinterpret_cast<uint32_t*>(&h2);
        }

        // 向量化写入：1 次 128-bit 写入 (写入 8 个 FP16/BF16 元素)
        uint4* output_u4 = reinterpret_cast<uint4*>(output);
        output_u4[vec_global_idx / 8] = make_uint4(out_packed[0], out_packed[1], out_packed[2], out_packed[3]);
    } 
    // Slow path: 尾部处理未对齐 8 的部分
    else if (vec_global_idx < total_elements) {
        for (int i = 0; i < 8 && (vec_global_idx + i) <= total_elements-1; i++) {
            int64_t current_idx = vec_global_idx + i;
            int byte_idx = current_idx / 2;
            uint8_t packed = packed_weights[byte_idx];
            uint8_t nf4_idx = (current_idx % 2 == 0) ? ((packed >> 4) & 0x0F) : (packed & 0x0F);
            
            int64_t block_idx = current_idx >> log2_blocksize;
            int64_t group_idx = block_idx >> 8;
            uint8_t absmax_idx = absmax_q[block_idx];
            
            float code_val = TypeTraits<T>::cast_to_float(code2[absmax_idx]);
            float absmax_val = TypeTraits<T>::cast_to_float(absmax2[group_idx]);
            float absmax = code_val * absmax_val + offset;
            
            output[current_idx] = TypeTraits<T>::cast_from_float(shared_lut[nf4_idx] * absmax);
        }
    }
}

template <typename T>
__global__ void nf4_dequant_optimized_kernel(
    const uint8_t* __restrict__ packed_weights, const uint8_t* __restrict__ absmax_q,
    const T* __restrict__ absmax2, const T* __restrict__ code2, T* __restrict__ output,
    int64_t total_elements, int32_t log2_blocksize, float offset
) {
    // 使用 Shared Memory 缓存 LUT
    __shared__ float shared_lut[16];
    if (threadIdx.x < 16) {
        shared_lut[threadIdx.x] = NF4_LUT[threadIdx.x];
    }
    __syncthreads();

    // 优化 1：每个线程处理 32 个元素 (1 个 uint4 = 16 Bytes)
    // 能够最大化利用 128-bit 显存带宽
    int64_t elem_global_idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) << 5; // << 5 等于 * 32
    
    // Fast path: 能够完整凑齐 32 个元素的线程 (PyTorch Tensor 内存默认 512 字节对齐，绝大多数走此分支)
    if (elem_global_idx + 31 < total_elements) {
        
        // 使用 128-bit 一次性读取 16 字节 (32个 NF4 权重)
        const uint4* packed_weights_u4 = reinterpret_cast<const uint4*>(packed_weights);
        uint4 packed_16b = __ldg(&packed_weights_u4[elem_global_idx >> 5]); // 使用 __ldg 绕过L1直接通过纹理缓存读取
        
        uint32_t packed_chunks[4] = {packed_16b.x, packed_16b.y, packed_16b.z, packed_16b.w};
        uint4 out_u4[4]; // 存放 4 个 128-bit 输出
        
        int64_t last_block_idx = -1;
        float absmax = 0.0f;

        // 优化 4：编译器展开此循环，内部寄存器使用达到软件流水线的效果
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            uint32_t packed = packed_chunks[c];
            uint32_t out_packed[4];

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                uint8_t byte_val = (packed >> (i << 3)) & 0xFF; // i<<3 等同于 i*8
                uint8_t idx_even = (byte_val >> 4) & 0x0F;
                uint8_t idx_odd  = byte_val & 0x0F;

                int64_t current_idx = elem_global_idx + (c << 3) + (i << 1);
                int64_t block_idx = current_idx >> log2_blocksize;
                
                // Block 边界检查
                if (block_idx != last_block_idx) {
                    int64_t group_idx = block_idx >> 8;
                    // 优化 2：使用 __ldg 只读缓存读取 scales
                    uint8_t absmax_idx = __ldg(&absmax_q[block_idx]);
                    
                    float code_val = TypeTraits<T>::cast_to_float(__ldg(&code2[absmax_idx]));
                    float absmax_val = TypeTraits<T>::cast_to_float(__ldg(&absmax2[group_idx]));
                    
                    absmax = code_val * absmax_val + offset;
                    last_block_idx = block_idx;
                }

                float val_even = shared_lut[idx_even] * absmax;
                float val_odd  = shared_lut[idx_odd]  * absmax;

                typename TypeTraits<T>::type2 h2 = TypeTraits<T>::pack(val_even, val_odd);
                out_packed[i] = *reinterpret_cast<uint32_t*>(&h2);
            }
            // 组装成 128-bit 准备写入
            out_u4[c] = make_uint4(out_packed[0], out_packed[1], out_packed[2], out_packed[3]);
        }

        // 一次性写入 4 个 uint4 (即写入了 64 字节，32个 FP16/BF16)
        uint4* output_u4 = reinterpret_cast<uint4*>(output);
        int64_t out_base_idx = elem_global_idx >> 3; // >> 3 因为 1 个 uint4 = 8 个元素
        
        output_u4[out_base_idx]     = out_u4[0];
        output_u4[out_base_idx + 1] = out_u4[1];
        output_u4[out_base_idx + 2] = out_u4[2];
        output_u4[out_base_idx + 3] = out_u4[3];
    } 
    // Slow path: 处理尾部不对齐的数据 (保持基本逻辑，但增加了位运算优化)
    else if (elem_global_idx < total_elements) {
        for (int i = 0; i < 32 && (elem_global_idx + i) < total_elements; i++) {
            int64_t current_idx = elem_global_idx + i;
            int64_t byte_idx = current_idx >> 1;
            uint8_t packed = packed_weights[byte_idx];
            uint8_t nf4_idx = ((current_idx & 1) == 0) ? ((packed >> 4) & 0x0F) : (packed & 0x0F);
            
            int64_t block_idx = current_idx >> log2_blocksize;
            int64_t group_idx = block_idx >> 8;
            uint8_t absmax_idx = absmax_q[block_idx];
            
            float code_val = TypeTraits<T>::cast_to_float(code2[absmax_idx]);
            float absmax_val = TypeTraits<T>::cast_to_float(absmax2[group_idx]);
            float absmax = code_val * absmax_val + offset;
            
            output[current_idx] = TypeTraits<T>::cast_from_float(shared_lut[nf4_idx] * absmax);
        }
    }
}


// 泛型 Launcher 供 Python 绑定调用
template <typename T>
torch::Tensor nf4_dequant_launcher(
    torch::Tensor packed_weights, torch::Tensor absmax_q, torch::Tensor absmax2, torch::Tensor code2,
    int64_t total_elements, int32_t log2_blocksize, float offset
) {
    auto dtype = std::is_same<T, __half>::value ? torch::kFloat16 : torch::kBFloat16;
    auto output = torch::empty({total_elements}, torch::dtype(dtype).device(packed_weights.device()));
    const int threads = 256;
    // const int blocks = (total_elements + threads * 2 - 1) / (threads * 2);
    // nf4_dequant_naive_kernel<T><<<blocks, threads>>>(
    //     static_cast<const uint8_t*>(packed_weights.data_ptr()),
    //     static_cast<const uint8_t*>(absmax_q.data_ptr()),
    //     static_cast<const T*>(absmax2.data_ptr()),
    //     static_cast<const T*>(code2.data_ptr()),
    //     static_cast<T*>(output.data_ptr()),
    //     total_elements,
    //     log2_blocksize,
    //     offset
    // );
    
    const int blocks = (total_elements + threads * 8 - 1) / (threads * 8);
    nf4_dequant_vectorized_kernel<T><<<blocks, threads>>>(
        static_cast<const uint8_t*>(packed_weights.data_ptr()),
        static_cast<const uint8_t*>(absmax_q.data_ptr()),
        static_cast<const T*>(absmax2.data_ptr()),
        static_cast<const T*>(code2.data_ptr()),
        static_cast<T*>(output.data_ptr()),
        total_elements,
        log2_blocksize,
        offset
    );

    // // 线程块计算公式调整：现在每个线程处理 32 个元素
    // const int elements_per_block = threads * 32;
    // const int blocks = (total_elements + elements_per_block - 1) / elements_per_block;
    
    // nf4_dequant_optimized_kernel<T><<<blocks, threads>>>(
    //     static_cast<const uint8_t*>(packed_weights.data_ptr()),
    //     static_cast<const uint8_t*>(absmax_q.data_ptr()),
    //     static_cast<const T*>(absmax2.data_ptr()),
    //     static_cast<const T*>(code2.data_ptr()),
    //     static_cast<T*>(output.data_ptr()),
    //     total_elements,
    //     log2_blocksize,
    //     offset
    // );


    return output;
}

torch::Tensor nf4_dequant_fp16_launcher(
    torch::Tensor packed_weights, torch::Tensor absmax_q, torch::Tensor absmax2, torch::Tensor code2,
    int64_t total_elements, int32_t log2_blocksize, float offset) {
    return nf4_dequant_launcher<__half>(packed_weights, absmax_q, absmax2, code2, total_elements, log2_blocksize, offset);
}
torch::Tensor nf4_dequant_bf16_launcher(
    torch::Tensor packed_weights, torch::Tensor absmax_q, torch::Tensor absmax2, torch::Tensor code2,
    int64_t total_elements, int32_t log2_blocksize, float offset) {
    return nf4_dequant_launcher<__nv_bfloat16>(packed_weights, absmax_q, absmax2, code2, total_elements, log2_blocksize, offset);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nf4_dequant_fp16", &nf4_dequant_fp16_launcher, "NF4 Dequantization CUDA Kernel (FP16)");
    m.def("nf4_dequant_bf16", &nf4_dequant_bf16_launcher, "NF4 Dequantization CUDA Kernel (BF16)");
}