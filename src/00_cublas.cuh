// 00_cublas.cuh
// cuBLAS wrapper for HGEMM benchmarking (FP16)
//
// NOTE: Expects B to be pre-transposed as B_T[N,K] row-major.
//       Uses CUBLAS_OP_T to handle the transposed layout.

#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include "utils.cuh"

struct HGEMMCuBLAS {
    static void Run(cublasHandle_t handle, int M, int N, int K, __half alpha,
                    const __half* A, const __half* B,
                    __half beta, __half* C) {
        // A is [M,K] row-major, B is [N,K] row-major (pre-transposed)
        // Row-major C = A * B^T becomes column-major C^T = B * A^T
        cublasGemmEx(handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    B, CUDA_R_16F, K,
                    A, CUDA_R_16F, K,
                    &beta,
                    C, CUDA_R_16F, N,
                    CUBLAS_COMPUTE_16F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
};
