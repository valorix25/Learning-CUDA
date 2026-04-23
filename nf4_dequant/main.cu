/**
 * NF4 Dequantization - Main driver
 * Usage: ./nf4_dequant_cuda <weight.bin> [version=7] [iters=100]
 * version: 1 or 7, selects which kernel to benchmark
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#include <string>

// Kernel declarations (defined in nf4_dequant_cuda.cu)
extern __constant__ __half NF4_LUT_HALF[16];
extern __constant__ __half CODE2_LUT[256];
extern void init_nf4_lut();

// v1: output __half2 (1 byte/thread)
extern __global__ void nf4_dequant_v1(const uint8_t*, const uint8_t*, const __half*, const __half*, __half2*, int64_t, int, int, float);
// v7: output __half (4 bytes/thread), code2 in constant memory
extern __global__ void nf4_dequant_v7(const uint8_t*, const uint8_t*, const __half*, __half*, int64_t, int, int, float);

// ============================================================
// Weight data
// ============================================================
struct WeightData {
    int64_t rows, cols;
    int blocksize, group_size;
    uint8_t *packed, *absmax_q;
    __half *absmax2, *code2;
    float offset;
    int64_t total_elements, packed_size;
    int num_blocks, num_groups;
};

WeightData read_weight_file(const char* filename) {
    WeightData w = {};
    FILE* f = fopen(filename, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", filename); exit(1); }

    fread(&w.rows, sizeof(int64_t), 1, f);
    fread(&w.cols, sizeof(int64_t), 1, f);
    fread(&w.blocksize, sizeof(int), 1, f);

    w.total_elements = w.rows * w.cols;
    w.packed_size = (w.total_elements + 1) / 2;
    w.num_blocks = (w.total_elements + w.blocksize - 1) / w.blocksize;
    w.num_groups = (w.num_blocks + 255) / 256;
    w.group_size = 256;

    w.packed = (uint8_t*)malloc(w.packed_size);
    w.absmax_q = (uint8_t*)malloc(w.num_blocks);
    w.absmax2 = (__half*)malloc(w.num_groups * sizeof(__half));
    w.code2 = (__half*)malloc(256 * sizeof(__half));

    fread(w.packed, 1, w.packed_size, f);
    fread(w.absmax_q, 1, w.num_blocks, f);
    fread(w.absmax2, sizeof(__half), w.num_groups, f);
    fread(w.code2, sizeof(__half), 256, f);
    fread(&w.offset, sizeof(float), 1, f);
    fclose(f);

    printf("Loaded %s: %ldx%ld bs=%d blocks=%d groups=%d\n",
           filename, w.rows, w.cols, w.blocksize, w.num_blocks, w.num_groups);
    return w;
}

void save_output(const char* filename, __half* data, int64_t count) {
    FILE* f = fopen(filename, "wb");
    if (!f) { fprintf(stderr, "Cannot write %s\n", filename); return; }
    fwrite(data, sizeof(__half), count, f);
    fclose(f);
    printf("Saved: %s (%ld elements, %.2f KB)\n", filename, count, count * sizeof(__half) / 1024.0);
}

// ============================================================
// Kernel launch wrapper
// ============================================================
void launch_kernel(int version, int blocks, int threads, int smem,
                   const uint8_t* d_packed, const uint8_t* d_absmax_q,
                   const __half* d_absmax2, const __half* d_code2,
                   __half* d_output, __half2* d_output_h2,
                   const WeightData& w) {
    switch (version) {
    case 1:
        nf4_dequant_v1<<<blocks, threads>>>(d_packed, d_absmax_q, d_absmax2, d_code2, d_output_h2, w.packed_size, w.blocksize, w.group_size, w.offset);
        break;
    case 7:
        nf4_dequant_v7<<<blocks, threads>>>(d_packed, d_absmax_q, d_absmax2, d_output, w.total_elements, w.blocksize, w.group_size, w.offset);
        break;
    default:
        fprintf(stderr, "Invalid version: %d (must be 1 or 7)\n", version);
        exit(1);
    }
}

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <weight.bin> [version=7] [iters=100]\n", argv[0]);
        return 1;
    }
    const char* weight_file = argv[1];
    int version = argc > 2 ? atoi(argv[2]) : 7;
    int iters = argc > 3 ? atoi(argv[3]) : 100;

    if (version != 1 && version != 7) {
        fprintf(stderr, "version must be 1 or 7, got %d\n", version);
        return 1;
    }

    init_nf4_lut();
    WeightData w = read_weight_file(weight_file);

    // Upload to GPU
    uint8_t *d_packed, *d_absmax_q;
    __half *d_absmax2, *d_code2, *d_output;
    __half2 *d_output_h2;
    cudaMalloc(&d_packed, w.packed_size);
    cudaMalloc(&d_absmax_q, w.num_blocks);
    cudaMalloc(&d_absmax2, w.num_groups * sizeof(__half));
    cudaMalloc(&d_code2, 256 * sizeof(__half));
    cudaMalloc(&d_output, w.total_elements * sizeof(__half));
    cudaMalloc(&d_output_h2, w.packed_size * sizeof(__half2));

    cudaMemcpy(d_packed, w.packed, w.packed_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_absmax_q, w.absmax_q, w.num_blocks, cudaMemcpyHostToDevice);
    cudaMemcpy(d_absmax2, w.absmax2, w.num_groups * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_code2, w.code2, 256 * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(CODE2_LUT, w.code2, 256 * sizeof(__half));

    // Launch config
    int threads = 256;
    int blocks, smem = 0;
    if (version == 1) {
        // v1: 1 byte/thread
        blocks = (w.packed_size + threads - 1) / threads;
    } else {
        // v7: 4 bytes/thread
        int64_t total_bytes = w.total_elements >> 1;
        blocks = (total_bytes / 4 + threads - 1) / threads;
    }

    // Warmup
    for (int i = 0; i < 10; i++)
        launch_kernel(version, blocks, threads, smem, d_packed, d_absmax_q, d_absmax2, d_code2, d_output, d_output_h2, w);
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++)
        launch_kernel(version, blocks, threads, smem, d_packed, d_absmax_q, d_absmax2, d_code2, d_output, d_output_h2, w);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / iters;

    int64_t in_bytes = w.packed_size + w.num_blocks + w.num_groups * sizeof(__half) + 256 * sizeof(__half) + sizeof(float);
    int64_t out_bytes = w.total_elements * sizeof(__half);
    float bw = (in_bytes + out_bytes) / (avg_ms * 1e6);

    printf("\nv%d: %.4f ms/iter, %.2f GB/s\n", version, avg_ms, bw);

    // Save (convert __half2 output to __half for v1)
    __half* h_out = (__half*)malloc(w.total_elements * sizeof(__half));
    if (version == 1) {
        // v1 output __half2, reinterpret as __half array
        cudaMemcpy(h_out, d_output_h2, w.packed_size * sizeof(__half2), cudaMemcpyDeviceToHost);
    } else {
        cudaMemcpy(h_out, d_output, w.total_elements * sizeof(__half), cudaMemcpyDeviceToHost);
    }

    std::string path(weight_file);
    size_t slash = path.find_last_of('/');
    std::string base = (slash != std::string::npos) ? path.substr(slash + 1) : path;
    mkdir("cuda_results", 0755);
    save_output(("cuda_results/v" + std::to_string(version) + "_" + base + ".fp16").c_str(), h_out, w.total_elements);

    // Cleanup
    free(h_out); free(w.packed); free(w.absmax_q); free(w.absmax2); free(w.code2);
    cudaFree(d_packed); cudaFree(d_absmax_q); cudaFree(d_absmax2); cudaFree(d_code2); cudaFree(d_output); cudaFree(d_output_h2);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
