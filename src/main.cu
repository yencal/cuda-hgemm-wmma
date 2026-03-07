// main.cu
// Benchmark runner for HGEMM implementations (FP16)
//
// NOTE: All kernels expect B to be pre-transposed as B_T[N,K] row-major.
//       The transpose is done once per problem size before benchmarking.

#include <iostream>
#include <vector>

#include "utils.cuh"
#include "00_cublas.cuh"
#include "01_wmma_block_tiling.cuh"
#include "02_wmma_vectorized.cuh"
#include "03_wmma_async_copy.cuh"
#include "04_wmma_padded.cuh"
#include "05_wmma_multistage.cuh"
#include "06_wmma_double_buffer.cuh"
#include "07_wmma_dynsmem.cuh"
#include "08_wmma_final.cuh"
#include "autotune.cuh"

int main(int argc, char** argv)
{
    std::vector<int> sizes = {1024, 2048, 4096, 8192, 16384};
    std::vector<BenchmarkResult> results;

    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // ========================================
    // Autotune all kernels upfront
    // ========================================
    printf("============================================================\n");
    printf("Autotuning all kernels ...\n");
    printf("============================================================\n\n");

    printf("Autotuning 01_WMMABlockTiling\n");
    RunAutotune<WMMABlockTilingTag>(GetWMMAVariants<WMMABlockTiling>());

    printf("Autotuning 02_WMMAVectorized\n");
    RunAutotune<WMMAVectorizedTag>(GetWMMAVectorizedVariants<WMMAVectorized>());

    printf("Autotuning 03_WMMAAsync\n");
    RunAutotune<WMMAAsyncTag>(GetWMMAVectorizedVariants<WMMAAsync>());

    printf("Autotuning 04_WMMAPadded\n");
    RunAutotune<WMMAPaddedTag>(GetWMMAVectorizedVariants<WMMAPadded>());

    printf("Autotuning 05_WMMAMultistage\n");
    RunAutotune<WMMAMultistageTag>(GetWMMAMultistageVariants<WMMAMultistage>());

    printf("Autotuning 06_WMMADoubleBuffer\n");
    RunAutotune<WMMADoubleBufferTag>(GetWMMAMultistageVariants<WMMADoubleBuffer>());

    printf("Autotuning 07_WMMADynSmem\n");
    RunAutotune<WMMADynSmemTag>(GetWMMADynSmemVariants<WMMADynSmem>());

    printf("Autotuning 08_WMMAFinal\n");
    RunAutotune<WMMAFinalTag>(GetWMMAFinalVariants<WMMAFinal>());

    printf("============================================================\n");
    printf("Autotuning complete. Running benchmarks...\n");
    printf("============================================================\n");

    // ========================================
    // Benchmark loop
    // ========================================
    for (int N : sizes) {
        int M = N, K = N;

        std::cout << "\n========================================" << std::endl;
        std::cout << "N = " << N << " (" << (2.0 * M * N * K / 1e9) << " GFLOPs)" << std::endl;
        std::cout << "========================================" << std::endl;

        __half *d_A, *d_B, *d_B_T, *d_C, *d_C_ref;
        CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_B_T, N * K * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_C_ref, M * N * sizeof(__half)));

        FillRandomDevice(d_A, M * K);
        FillRandomDevice(d_B, K * N);
        
        // Transpose B[K,N] -> B_T[N,K]
        TransposeMatrix(d_B_T, d_B, K, N);

        // Generate reference (using transposed B)
        HGEMMCuBLAS::Run(handle, M, N, K, alpha, d_A, d_B_T, beta, d_C_ref);
        CHECK_CUDA(cudaDeviceSynchronize());

        // 00: cuBLAS reference
        results.push_back(RunCuBLASBenchmark<HGEMMCuBLAS>(
            "00_cuBLAS", handle, M, N, K, alpha, d_A, d_B_T, beta, d_C));

        // 01-07: Autotuned kernels (all use transposed B)
        RunAndRecordAutotuned<WMMABlockTilingTag>(
            results, "01_WMMABlockTiling", M, N, K, alpha, d_A, d_B_T, beta, d_C, d_C_ref);

        RunAndRecordAutotuned<WMMAVectorizedTag>(
            results, "02_WMMAVectorized", M, N, K, alpha, d_A, d_B_T, beta, d_C, d_C_ref);

        RunAndRecordAutotuned<WMMAAsyncTag>(
            results, "03_WMMAAsync", M, N, K, alpha, d_A, d_B_T, beta, d_C, d_C_ref);

        RunAndRecordAutotuned<WMMAPaddedTag>(
            results, "04_WMMAPadded", M, N, K, alpha, d_A, d_B_T, beta, d_C, d_C_ref);

        RunAndRecordAutotuned<WMMAMultistageTag>(
            results, "05_WMMAMultistage", M, N, K, alpha, d_A, d_B_T, beta, d_C, d_C_ref);

        RunAndRecordAutotuned<WMMADoubleBufferTag>(
            results, "06_WMMADoubleBuffer", M, N, K, alpha, d_A, d_B_T, beta, d_C, d_C_ref);

        RunAndRecordAutotuned<WMMADynSmemTag>(
            results, "07_WMMADynSmem", M, N, K, alpha, d_A, d_B_T, beta, d_C, d_C_ref);

        // 08: WMMAFinal - autotuned per size
        RunAutotune<WMMAFinalTag>(GetWMMAFinalVariants<WMMAFinal>(), N);
        CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(__half)));
        results.push_back(RunBenchmark<Autotuned<WMMAFinalTag>>(
            "08_WMMAFinal", M, N, K, alpha, d_A, d_B_T, beta, d_C, d_C_ref));

        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_B_T));
        CHECK_CUDA(cudaFree(d_C));
        CHECK_CUDA(cudaFree(d_C_ref));
    }
    CHECK_CUBLAS(cublasDestroy(handle));

    WriteCSV(results, "hgemm_results.csv");
    std::cout << "\nResults saved to hgemm_results.csv" << std::endl;

    return 0;
}
