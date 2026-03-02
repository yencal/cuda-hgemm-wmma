// autotune.cuh
// Autotuning framework for HGEMM kernels (FP16)
// Expanded configuration space for A100 (164KB dynamic shared memory)

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <functional>
#include <cfloat>
#include <cstdio>
#include <string>

#include "utils.cuh"
#include "01_wmma_block_tiling.cuh"
#include "02_wmma_vectorized.cuh"
#include "03_wmma_async_copy.cuh"
#include "04_wmma_padded.cuh"
#include "05_wmma_multistage.cuh"
#include "06_wmma_double_buffer.cuh"
#include "07_wmma_dynsmem.cuh"
#include "08_wmma_final.cuh"

struct TuneConfig {
    const char* name;
    std::function<void(int, int, int, __half, const __half*, const __half*, __half, __half*)> run;
};

struct WMMABlockTilingTag {};
struct WMMAVectorizedTag {};
struct WMMAAsyncTag {};
struct WMMAPaddedTag {};
struct WMMAMultistageTag {};
struct WMMADoubleBufferTag {};
struct WMMADynSmemTag {};
struct WMMAFinalTag {};

template<typename Tag>
struct Autotuned {
    static inline TuneConfig config;

    static void Run(int M, int N, int K, __half alpha,
                    const __half* A, const __half* B,
                    __half beta, __half* C) {
        config.run(M, N, K, alpha, A, B, beta, C);
    }
};

#define TUNE_CONFIG(Kernel, BM, BN, BK, WM, WN) \
    TuneConfig{#BM "x" #BN "x" #BK "_" #WM "x" #WN, Kernel<BM, BN, BK, WM, WN>::Run}

#define TUNE_CONFIG_MULTISTAGE(Kernel, BM, BN, BK, WM, WN, STAGES) \
    TuneConfig{#BM "x" #BN "x" #BK "_" #WM "x" #WN "_S" #STAGES, \
               Kernel<BM, BN, BK, WM, WN, STAGES>::Run}

// For WMMAFinal with swizzle options
#define TUNE_CONFIG_FINAL(Kernel, BM, BN, BK, WM, WN, STAGES, USE_SWIZZLE, GROUP_SIZE_M) \
    TuneConfig{#BM "x" #BN "x" #BK "_" #WM "x" #WN "_S" #STAGES \
               "_sw" #USE_SWIZZLE "_g" #GROUP_SIZE_M, \
               Kernel<BM, BN, BK, WM, WN, STAGES, USE_SWIZZLE, GROUP_SIZE_M>::Run}

template<template<int, int, int, int, int> class Kernel>
inline std::vector<TuneConfig> GetWMMAVariants() {
    return {
        // BK must be multiple of 16 (MMA_K)
        TUNE_CONFIG(Kernel, 64,  64,  16, 16, 16),
        TUNE_CONFIG(Kernel, 64,  64,  32, 16, 16),
        TUNE_CONFIG(Kernel, 64,  128, 16, 16, 32),
        TUNE_CONFIG(Kernel, 128, 64,  16, 32, 16),
        TUNE_CONFIG(Kernel, 128, 128, 16, 32, 32),
        TUNE_CONFIG(Kernel, 128, 128, 32, 32, 32),
        TUNE_CONFIG(Kernel, 128, 128, 64, 32, 32),
        TUNE_CONFIG(Kernel, 64,  128, 32, 32, 32),
    };
}

// For vectorized kernel: need (BM*BK)/8 >= NUM_THREADS and divisible
template<template<int, int, int, int, int> class Kernel>
inline std::vector<TuneConfig> GetWMMAVectorizedVariants() {
    return {
        // These all satisfy: (BM*BK)/8 % NUM_THREADS == 0
        TUNE_CONFIG(Kernel, 128, 128, 32, 32, 32),  // 512 threads, 512 vecs
        TUNE_CONFIG(Kernel, 128, 128, 64, 32, 32),  // 512 threads, 1024 vecs
        TUNE_CONFIG(Kernel, 128, 256, 32, 32, 64),  // 512 threads, 512 vecs
        TUNE_CONFIG(Kernel, 256, 128, 32, 64, 32),  // 512 threads, 1024 vecs
        TUNE_CONFIG(Kernel, 128, 128, 32, 64, 64),  // 128 threads, 512 vecs
        TUNE_CONFIG(Kernel, 64,  128, 64, 32, 32),  // 256 threads, 512 vecs
        TUNE_CONFIG(Kernel, 128, 64,  64, 32, 32),  // 256 threads, 1024 vecs
    };
}

// For multistage
template<template<int, int, int, int, int, int> class Kernel>
inline std::vector<TuneConfig> GetWMMAMultistageVariants() {
    return {
        // sm_80 (A100): 48KB shared mem, STAGES * BK <= 96
        // 2-stage
        TUNE_CONFIG_MULTISTAGE(Kernel, 128, 128, 32, 64, 64, 2), 
        TUNE_CONFIG_MULTISTAGE(Kernel, 128, 128, 32, 32, 32, 2), 
        // 3-stage
        TUNE_CONFIG_MULTISTAGE(Kernel, 128, 64, 32, 64, 32, 3),
        TUNE_CONFIG_MULTISTAGE(Kernel, 64, 128, 32, 32, 64, 3),
        // 4-stage
        TUNE_CONFIG_MULTISTAGE(Kernel, 128, 128, 16, 64, 64, 4),
        TUNE_CONFIG_MULTISTAGE(Kernel, 64, 64, 32, 32, 32, 4),
    };  
}

template<template<int, int, int, int, int, int> class Kernel>
inline std::vector<TuneConfig> GetWMMADynSmemVariants() {
    return {
        TUNE_CONFIG_MULTISTAGE(Kernel, 128, 128, 32, 64, 64, 2),
        TUNE_CONFIG_MULTISTAGE(Kernel, 128, 128, 32, 64, 64, 3),
        TUNE_CONFIG_MULTISTAGE(Kernel, 128, 128, 32, 64, 64, 4),
        TUNE_CONFIG_MULTISTAGE(Kernel, 256, 128, 32, 64, 64, 3),
    };
}

// ============================================================================
// Expanded configuration space for WMMAFinal (optimized for A100)
// A100 supports up to 164KB dynamic shared memory per block
//
// CONSTRAINT: TOTAL_AB_SIZE >= C_SMEM_SIZE (for epilogue reuse)
//   TOTAL_AB_SIZE = STAGES * (BM * (BK+8) + BK * (BN+8))
//   C_SMEM_SIZE = BM * (BN + 8)
// ============================================================================
template<template<int, int, int, int, int, int, bool, int> class Kernel>
inline std::vector<TuneConfig> GetWMMAFinalVariants() {
    return {
        // ====================================================================
        // 128x128 block tiles, BK=32 (~18.5 KB/stage)
        // C_SMEM = 17408, 2-stage AB = 18944 ✓
        // ====================================================================
        TUNE_CONFIG_FINAL(Kernel, 128, 128, 32, 64, 64, 2, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 128, 128, 32, 64, 64, 3, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 128, 128, 32, 64, 64, 4, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 128, 128, 32, 64, 64, 5, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 128, 128, 32, 64, 64, 2, true, 4),
        TUNE_CONFIG_FINAL(Kernel, 128, 128, 32, 64, 64, 2, true, 8),
        TUNE_CONFIG_FINAL(Kernel, 128, 128, 32, 64, 64, 2, true, 16),
        TUNE_CONFIG_FINAL(Kernel, 128, 128, 32, 64, 64, 3, true, 4),
        TUNE_CONFIG_FINAL(Kernel, 128, 128, 32, 64, 64, 3, true, 8),
        TUNE_CONFIG_FINAL(Kernel, 128, 128, 32, 64, 64, 3, true, 16),
        TUNE_CONFIG_FINAL(Kernel, 128, 128, 32, 64, 64, 4, true, 8),
        TUNE_CONFIG_FINAL(Kernel, 128, 128, 32, 64, 64, 5, true, 8),

        // ====================================================================
        // 128x128 block tiles, BK=64 (~35 KB/stage)
        // C_SMEM = 17408, 2-stage AB = 35840 ✓
        // ====================================================================
        TUNE_CONFIG_FINAL(Kernel, 128, 128, 64, 64, 64, 2, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 128, 128, 64, 64, 64, 3, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 128, 128, 64, 64, 64, 4, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 128, 128, 64, 64, 64, 2, true, 4),
        TUNE_CONFIG_FINAL(Kernel, 128, 128, 64, 64, 64, 2, true, 8),
        TUNE_CONFIG_FINAL(Kernel, 128, 128, 64, 64, 64, 2, true, 16),
        TUNE_CONFIG_FINAL(Kernel, 128, 128, 64, 64, 64, 3, true, 8),
        TUNE_CONFIG_FINAL(Kernel, 128, 128, 64, 64, 64, 4, true, 8),

        // ====================================================================
        // 128x128 with 32x32 warp tiles (16 warps vs 4)
        // C_SMEM = 17408, 2-stage AB = 18944 ✓
        // ====================================================================
        TUNE_CONFIG_FINAL(Kernel, 128, 128, 32, 32, 32, 2, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 128, 128, 32, 32, 32, 3, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 128, 128, 32, 32, 32, 4, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 128, 128, 32, 32, 32, 2, true, 8),
        TUNE_CONFIG_FINAL(Kernel, 128, 128, 32, 32, 32, 3, true, 8),
        TUNE_CONFIG_FINAL(Kernel, 128, 128, 32, 32, 32, 4, true, 8),
        TUNE_CONFIG_FINAL(Kernel, 128, 128, 64, 32, 32, 2, true, 8),
        TUNE_CONFIG_FINAL(Kernel, 128, 128, 64, 32, 32, 3, true, 8),

        // ====================================================================
        // 256x128 block tiles, BK=32 (~28.5 KB/stage)
        // C_SMEM = 34816, need 3+ stages (3-stage AB = 43776 ✓)
        // ====================================================================
        TUNE_CONFIG_FINAL(Kernel, 256, 128, 32, 64, 64, 3, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 256, 128, 32, 64, 64, 4, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 256, 128, 32, 64, 64, 5, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 256, 128, 32, 64, 64, 3, true, 4),
        TUNE_CONFIG_FINAL(Kernel, 256, 128, 32, 64, 64, 3, true, 8),
        TUNE_CONFIG_FINAL(Kernel, 256, 128, 32, 64, 64, 3, true, 16),
        TUNE_CONFIG_FINAL(Kernel, 256, 128, 32, 64, 64, 4, true, 8),
        TUNE_CONFIG_FINAL(Kernel, 256, 128, 32, 64, 64, 5, true, 8),

        // ====================================================================
        // 256x128 block tiles, BK=64 (~54 KB/stage)
        // C_SMEM = 34816, 2-stage AB = 54272 ✓
        // ====================================================================
        TUNE_CONFIG_FINAL(Kernel, 256, 128, 64, 64, 64, 2, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 256, 128, 64, 64, 64, 2, true, 8),
        TUNE_CONFIG_FINAL(Kernel, 256, 128, 64, 64, 64, 3, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 256, 128, 64, 64, 64, 3, true, 8),

        // ====================================================================
        // 128x256 block tiles, BK=32 (~26.5 KB/stage)
        // C_SMEM = 33792, need 3+ stages (3-stage AB = 40704 ✓)
        // ====================================================================
        TUNE_CONFIG_FINAL(Kernel, 128, 256, 32, 64, 64, 3, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 128, 256, 32, 64, 64, 4, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 128, 256, 32, 64, 64, 5, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 128, 256, 32, 64, 64, 3, true, 4),
        TUNE_CONFIG_FINAL(Kernel, 128, 256, 32, 64, 64, 3, true, 8),
        TUNE_CONFIG_FINAL(Kernel, 128, 256, 32, 64, 64, 3, true, 16),
        TUNE_CONFIG_FINAL(Kernel, 128, 256, 32, 64, 64, 4, true, 8),
        TUNE_CONFIG_FINAL(Kernel, 128, 256, 32, 64, 64, 5, true, 8),

        // ====================================================================
        // 128x256 block tiles, BK=64 (~52 KB/stage)
        // C_SMEM = 33792, 2-stage AB = 52224 ✓
        // ====================================================================
        TUNE_CONFIG_FINAL(Kernel, 128, 256, 64, 64, 64, 2, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 128, 256, 64, 64, 64, 2, true, 8),
        TUNE_CONFIG_FINAL(Kernel, 128, 256, 64, 64, 64, 3, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 128, 256, 64, 64, 64, 3, true, 8),

        // ====================================================================
        // 256x256 block tiles, BK=32 (~36.5 KB/stage)
        // C_SMEM = 67584, need 4+ stages (4-stage AB = 74752 ✓)
        // ====================================================================
        TUNE_CONFIG_FINAL(Kernel, 256, 256, 32, 64, 64, 4, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 256, 256, 32, 64, 64, 5, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 256, 256, 32, 64, 64, 4, true, 8),
        TUNE_CONFIG_FINAL(Kernel, 256, 256, 32, 64, 64, 5, true, 8),
        TUNE_CONFIG_FINAL(Kernel, 256, 256, 32, 64, 64, 4, true, 4),
        TUNE_CONFIG_FINAL(Kernel, 256, 256, 32, 64, 64, 4, true, 16),

        // ====================================================================
        // 256x256 block tiles, BK=64 (~70 KB/stage)
        // C_SMEM = 67584, 2-stage AB = 70656 ✓ (barely!)
        // ====================================================================
        TUNE_CONFIG_FINAL(Kernel, 256, 256, 64, 64, 64, 2, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 256, 256, 64, 64, 64, 2, true, 8),

        // ====================================================================
        // 64x64 block tiles (~9.5 KB/stage with BK=32)
        // C_SMEM = 4608, 2-stage AB = 9728 ✓
        // ====================================================================
        TUNE_CONFIG_FINAL(Kernel, 64, 64, 32, 32, 32, 2, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 64, 64, 32, 32, 32, 3, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 64, 64, 32, 32, 32, 4, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 64, 64, 32, 32, 32, 5, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 64, 64, 32, 32, 32, 2, true, 8),
        TUNE_CONFIG_FINAL(Kernel, 64, 64, 32, 32, 32, 3, true, 8),
        TUNE_CONFIG_FINAL(Kernel, 64, 64, 32, 32, 32, 4, true, 8),
        TUNE_CONFIG_FINAL(Kernel, 64, 64, 32, 32, 32, 5, true, 8),
        // With BK=64 (~18 KB/stage)
        TUNE_CONFIG_FINAL(Kernel, 64, 64, 64, 32, 32, 2, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 64, 64, 64, 32, 32, 3, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 64, 64, 64, 32, 32, 4, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 64, 64, 64, 32, 32, 2, true, 8),
        TUNE_CONFIG_FINAL(Kernel, 64, 64, 64, 32, 32, 3, true, 8),
        TUNE_CONFIG_FINAL(Kernel, 64, 64, 64, 32, 32, 4, true, 8),

        // ====================================================================
        // 256x64 block tiles (~24.5 KB/stage)
        // C_SMEM = 18432, 2-stage AB = 25088 ✓
        // ====================================================================
        TUNE_CONFIG_FINAL(Kernel, 256, 64, 32, 64, 32, 2, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 256, 64, 32, 64, 32, 3, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 256, 64, 32, 64, 32, 4, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 256, 64, 32, 64, 32, 2, true, 8),
        TUNE_CONFIG_FINAL(Kernel, 256, 64, 32, 64, 32, 3, true, 8),
        TUNE_CONFIG_FINAL(Kernel, 256, 64, 32, 64, 32, 4, true, 8),
        TUNE_CONFIG_FINAL(Kernel, 256, 64, 64, 64, 32, 2, true, 8),
        TUNE_CONFIG_FINAL(Kernel, 256, 64, 64, 64, 32, 3, true, 8),

        // ====================================================================
        // 64x256 block tiles (~21.5 KB/stage)
        // C_SMEM = 16896, 2-stage AB = 22016 ✓
        // ====================================================================
        TUNE_CONFIG_FINAL(Kernel, 64, 256, 32, 32, 64, 2, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 64, 256, 32, 32, 64, 3, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 64, 256, 32, 32, 64, 4, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 64, 256, 32, 32, 64, 2, true, 8),
        TUNE_CONFIG_FINAL(Kernel, 64, 256, 32, 32, 64, 3, true, 8),
        TUNE_CONFIG_FINAL(Kernel, 64, 256, 32, 32, 64, 4, true, 8),
        TUNE_CONFIG_FINAL(Kernel, 64, 256, 64, 32, 64, 2, true, 8),
        TUNE_CONFIG_FINAL(Kernel, 64, 256, 64, 32, 64, 3, true, 8),

        // ====================================================================
        // 64x128 block tiles (~13.5 KB/stage)
        // C_SMEM = 8704, 2-stage AB = 13824 ✓
        // ====================================================================
        TUNE_CONFIG_FINAL(Kernel, 64, 128, 32, 32, 64, 2, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 64, 128, 32, 32, 64, 3, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 64, 128, 32, 32, 64, 4, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 64, 128, 32, 32, 64, 2, true, 8),
        TUNE_CONFIG_FINAL(Kernel, 64, 128, 32, 32, 64, 3, true, 8),
        TUNE_CONFIG_FINAL(Kernel, 64, 128, 64, 32, 64, 2, true, 8),
        TUNE_CONFIG_FINAL(Kernel, 64, 128, 64, 32, 64, 3, true, 8),

        // ====================================================================
        // 128x64 block tiles (~14.5 KB/stage)
        // C_SMEM = 9216, 2-stage AB = 14848 ✓
        // ====================================================================
        TUNE_CONFIG_FINAL(Kernel, 128, 64, 32, 64, 32, 2, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 128, 64, 32, 64, 32, 3, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 128, 64, 32, 64, 32, 4, false, 8),
        TUNE_CONFIG_FINAL(Kernel, 128, 64, 32, 64, 32, 2, true, 8),
        TUNE_CONFIG_FINAL(Kernel, 128, 64, 32, 64, 32, 3, true, 8),
        TUNE_CONFIG_FINAL(Kernel, 128, 64, 64, 64, 32, 2, true, 8),
        TUNE_CONFIG_FINAL(Kernel, 128, 64, 64, 64, 32, 3, true, 8),
    };
}

inline TuneConfig Autotune(
    const std::vector<TuneConfig>& variants,
    int M, int N, int K, __half alpha,
    const __half* A, const __half* B,
    __half beta, __half* C,
    int warmup = 2, int iters = 10)
{
    float best_time = FLT_MAX;
    TuneConfig best = variants[0];

    printf("\n[Autotune] Testing %zu configurations on %dx%dx%d...\n",
           variants.size(), M, N, K);

    for (const auto& config : variants) {
        for (int i = 0; i < warmup; i++) {
            config.run(M, N, K, alpha, A, B, beta, C);
        }
        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("  %-40s SKIP (%s)\n", config.name, cudaGetErrorString(err));
            continue;
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        for (int i = 0; i < iters; i++) {
            config.run(M, N, K, alpha, A, B, beta, C);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        ms /= iters;

        double tflops = (2.0 * M * N * K) / (ms * 1e9);
        printf("  %-40s %7.3f ms  %6.2f TFLOPS\n", config.name, ms, tflops);

        if (ms < best_time) {
            best_time = ms;
            best = config;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    double best_tflops = (2.0 * M * N * K) / (best_time * 1e9);
    printf("[Autotune] Best: %s (%.3f ms, %.2f TFLOPS)\n\n",
           best.name, best_time, best_tflops);

    return best;
}

template<typename Tag>
inline void RunAutotune(
    const std::vector<TuneConfig>& variants,
    int tuneN = 4096,
    __half alpha = __float2half(1.0f),
    __half beta = __float2half(0.0f))
{
    __half *tune_A, *tune_B, *tune_C;
    CHECK_CUDA(cudaMalloc(&tune_A, (size_t)tuneN * tuneN * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&tune_B, (size_t)tuneN * tuneN * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&tune_C, (size_t)tuneN * tuneN * sizeof(__half)));

    FillRandomDevice(tune_A, (size_t)tuneN * tuneN);
    FillRandomDevice(tune_B, (size_t)tuneN * tuneN);

    Autotuned<Tag>::config = Autotune(
        variants, tuneN, tuneN, tuneN, alpha, tune_A, tune_B, beta, tune_C);

    CHECK_CUDA(cudaFree(tune_A));
    CHECK_CUDA(cudaFree(tune_B));
    CHECK_CUDA(cudaFree(tune_C));
}

// Helper function: benchmark with autotuned config, label includes winning config name
template<typename Tag>
void RunAndRecordAutotuned(
    std::vector<BenchmarkResult>& results,
    const char* kernel_name,
    int M, int N, int K,
    __half alpha, const __half* A, const __half* B,
    __half beta, __half* C, const __half* C_ref)
{
    CHECK_CUDA(cudaMemset(C, 0, M * N * sizeof(__half)));
    // std::string label = std::string(kernel_name) + " [" + Autotuned<Tag>::config.name + "]";
    std::string label = std::string(kernel_name);
    results.push_back(RunBenchmark<Autotuned<Tag>>(
        label.c_str(), M, N, K, alpha, A, B, beta, C, C_ref));
}
