// kernel_helpers.cuh
// Reusable functions for WMMA HGEMM kernels (FP16)
//
// NOTE: All loadTileB functions expect B to be pre-transposed as B_T[N,K] row-major.
//       This allows B to use the same small stride as A (BK + pad) for zero bank conflicts.

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline.h>

using namespace nvcuda;

// =========================================================================
// Tile Loading A: Scalar
// =========================================================================

template <int BM, int BK, int NUM_THREADS>
__device__ void loadTileA_scalar(
    const __half *A,
    __half *As,
    int K,
    uint tid)
{
    constexpr int ELEMS_PER_THREAD = (BM * BK) / NUM_THREADS;
    
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        uint idx = tid + i * NUM_THREADS;
        uint row = idx / BK;
        uint col = idx % BK;
        As[row * BK + col] = A[row * K + col];
    }
}

// =========================================================================
// Tile Loading B: Scalar (B is [N,K] row-major)
// =========================================================================

template <int BN, int BK, int NUM_THREADS>
__device__ void loadTileB_scalar(
    const __half *B,   // B[N,K] row-major, pointing to tile start
    __half *Bs,        // Bs[BN][BK]
    int K,
    uint tid)
{
    constexpr int ELEMS_PER_THREAD = (BN * BK) / NUM_THREADS;
    
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        uint idx = tid + i * NUM_THREADS;
        uint row = idx / BK;
        uint col = idx % BK;
        Bs[row * BK + col] = B[row * K + col];
    }
}

// =========================================================================
// Tile Loading A: Vectorized float4 = 8 halves
// =========================================================================

template <int BM, int BK, int NUM_THREADS>
__device__ void loadTileA_vec4(
    const __half *A,
    __half *As,
    int K,
    uint tid)
{
    constexpr int TOTAL_VEC = (BM * BK) / 8;
    constexpr int VEC_PER_THREAD = TOTAL_VEC / NUM_THREADS;
    
    static_assert((BM * BK) % 8 == 0, "Tile size must be divisible by 8");
    static_assert(TOTAL_VEC % NUM_THREADS == 0, "vec count must be divisible by NUM_THREADS");
    static_assert(BK % 8 == 0, "BK must be divisible by 8 for vectorized loads");
    
    #pragma unroll
    for (int i = 0; i < VEC_PER_THREAD; ++i) {
        uint idx = tid + i * NUM_THREADS;
        uint vec_per_row = BK / 8;
        uint row = idx / vec_per_row;
        uint col8 = idx % vec_per_row;
        
        float4 val = reinterpret_cast<const float4*>(&A[row * K + col8 * 8])[0];
        reinterpret_cast<float4*>(&As[row * BK + col8 * 8])[0] = val;
    }
}

// =========================================================================
// Tile Loading B: Vectorized float4 = 8 halves (B is [N,K] row-major)
// =========================================================================

template <int BN, int BK, int NUM_THREADS>
__device__ void loadTileB_vec4(
    const __half *B,   // B[N,K] row-major, pointing to tile start
    __half *Bs,        // Bs[BN][BK]
    int K,
    uint tid)
{
    constexpr int TOTAL_VEC = (BN * BK) / 8;
    constexpr int VEC_PER_THREAD = TOTAL_VEC / NUM_THREADS;
    
    static_assert((BN * BK) % 8 == 0, "Tile size must be divisible by 8");
    static_assert(TOTAL_VEC % NUM_THREADS == 0, "vec count must be divisible by NUM_THREADS");
    static_assert(BK % 8 == 0, "BK must be divisible by 8 for vectorized loads");
    
    #pragma unroll
    for (int i = 0; i < VEC_PER_THREAD; ++i) {
        uint idx = tid + i * NUM_THREADS;
        uint vec_per_row = BK / 8;
        uint row = idx / vec_per_row;
        uint col8 = idx % vec_per_row;
        
        float4 val = reinterpret_cast<const float4*>(&B[row * K + col8 * 8])[0];
        reinterpret_cast<float4*>(&Bs[row * BK + col8 * 8])[0] = val;
    }
}

// =========================================================================
// Tile Loading A: Async (cp.async, 16 bytes = 8 halves per copy)
// =========================================================================

template <int BM, int BK, int NUM_THREADS>
__device__ void loadTileA_async(
    const __half *A,
    __half *As,
    int K,
    uint tid)
{
    constexpr int TOTAL_VEC = (BM * BK) / 8;
    constexpr int VEC_PER_THREAD = TOTAL_VEC / NUM_THREADS;

    #pragma unroll
    for (int i = 0; i < VEC_PER_THREAD; ++i) {
        uint idx = tid + i * NUM_THREADS;
        uint row = idx / (BK / 8);
        uint col8 = idx % (BK / 8);
        __pipeline_memcpy_async(
            &As[row * BK + col8 * 8],
            &A[row * K + col8 * 8],
            sizeof(float4)
        );
    }
}

// =========================================================================
// Tile Loading B: Async (B is [N,K] row-major)
// =========================================================================

template <int BN, int BK, int NUM_THREADS>
__device__ void loadTileB_async(
    const __half *B,   // B[N,K] row-major, pointing to tile start
    __half *Bs,        // Bs[BN][BK]
    int K,
    uint tid)
{
    constexpr int TOTAL_VEC = (BN * BK) / 8;
    constexpr int VEC_PER_THREAD = TOTAL_VEC / NUM_THREADS;

    #pragma unroll
    for (int i = 0; i < VEC_PER_THREAD; ++i) {
        uint idx = tid + i * NUM_THREADS;
        uint row = idx / (BK / 8);
        uint col8 = idx % (BK / 8);
        __pipeline_memcpy_async(
            &Bs[row * BK + col8 * 8],
            &B[row * K + col8 * 8],
            sizeof(float4)
        );
    }
}

// =========================================================================
// Tile Loading A: Async with padding
// =========================================================================

template <int BM, int BK, int A_STRIDE, int NUM_THREADS>
__device__ void loadTileA_async_padded(
    const __half *A,
    __half *As,
    int K,
    uint tid)
{
    constexpr int TOTAL_VEC = (BM * BK) / 8;
    constexpr int VEC_PER_THREAD = TOTAL_VEC / NUM_THREADS;

    #pragma unroll
    for (int i = 0; i < VEC_PER_THREAD; ++i) {
        uint idx = tid + i * NUM_THREADS;
        uint row = idx / (BK / 8);
        uint col8 = idx % (BK / 8);
        
        __pipeline_memcpy_async(
            &As[row * A_STRIDE + col8 * 8],
            &A[row * K + col8 * 8],
            sizeof(float4)
        );
    }
}

// =========================================================================
// Tile Loading B: Async with padding (B is [N,K] row-major)
// =========================================================================

template <int BN, int BK, int B_STRIDE, int NUM_THREADS>
__device__ void loadTileB_async_padded(
    const __half *B,   // B[N,K] row-major, pointing to tile start
    __half *Bs,        // Bs[BN][B_STRIDE]
    int K,
    uint tid)
{
    constexpr int TOTAL_VEC = (BN * BK) / 8;
    constexpr int VEC_PER_THREAD = TOTAL_VEC / NUM_THREADS;

    #pragma unroll
    for (int i = 0; i < VEC_PER_THREAD; ++i) {
        uint idx = tid + i * NUM_THREADS;
        uint row = idx / (BK / 8);
        uint col8 = idx % (BK / 8);
        
        __pipeline_memcpy_async(
            &Bs[row * B_STRIDE + col8 * 8],
            &B[row * K + col8 * 8],
            sizeof(float4)
        );
    }
}

// =========================================================================
// Epilogue: C = alpha * acc + beta * C (FP16 accumulator)
// =========================================================================

template <int MMA_M, int MMA_N, int MMA_K, int MMA_M_TILES, int MMA_N_TILES, int WM, int WN>
__device__ void epilogueAndStore(
    wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, __half> acc[MMA_M_TILES][MMA_N_TILES],
    __half *C,
    int N,
    __half alpha,
    __half beta,
    uint warpM,
    uint warpN)
{
    #pragma unroll
    for (int m = 0; m < MMA_M_TILES; ++m) {
        #pragma unroll
        for (int n = 0; n < MMA_N_TILES; ++n) {
            __half *C_ptr = C + (warpM * WM + m * MMA_M) * N 
                              + (warpN * WN + n * MMA_N);

            #pragma unroll
            for (int i = 0; i < acc[m][n].num_elements; ++i) {
                acc[m][n].x[i] = __hmul(acc[m][n].x[i], alpha);
            }

            if (__heq(beta, __float2half(0.0f)) == false) {
                wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, __half> c_frag;
                wmma::load_matrix_sync(c_frag, C_ptr, N, wmma::mem_row_major);
                
                #pragma unroll
                for (int i = 0; i < acc[m][n].num_elements; ++i) {
                    acc[m][n].x[i] = __hadd(acc[m][n].x[i], __hmul(beta, c_frag.x[i]));
                }
            }

            wmma::store_matrix_sync(C_ptr, acc[m][n], N, wmma::mem_row_major);
        }
    }
}

// =========================================================================
// Vectorized Epilogue (reuses existing shared memory)
// =========================================================================

template <int BM, int BN, int WM, int WN, 
          int MMA_M, int MMA_N, int MMA_K,
          int MMA_M_TILES, int MMA_N_TILES, 
          int WARPS_M, int WARPS_N,
          int C_SMEM_STRIDE>
__device__ void epilogueAndStore_vec4(
    wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, __half> acc[MMA_M_TILES][MMA_N_TILES],
    __half *smem,
    __half *C,
    int N,
    __half alpha,
    __half beta,
    uint tid,
    uint warpM,
    uint warpN)
{
    __half *C_smem = smem;
    
    // Scale by alpha and handle beta
    #pragma unroll
    for (int m = 0; m < MMA_M_TILES; ++m) {
        #pragma unroll
        for (int n = 0; n < MMA_N_TILES; ++n) {
            #pragma unroll
            for (int i = 0; i < acc[m][n].num_elements; ++i) {
                acc[m][n].x[i] = __hmul(acc[m][n].x[i], alpha);
            }
            
            if (__heq(beta, __float2half(0.0f)) == false) {
                __half *C_ptr = C + (warpM * WM + m * MMA_M) * N 
                                  + (warpN * WN + n * MMA_N);
                wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, __half> c_frag;
                wmma::load_matrix_sync(c_frag, C_ptr, N, wmma::mem_row_major);
                
                #pragma unroll
                for (int i = 0; i < acc[m][n].num_elements; ++i) {
                    acc[m][n].x[i] = __hadd(acc[m][n].x[i], __hmul(beta, c_frag.x[i]));
                }
            }
        }
    }
    
    // Store fragments to shared memory
    #pragma unroll
    for (int m = 0; m < MMA_M_TILES; ++m) {
        #pragma unroll
        for (int n = 0; n < MMA_N_TILES; ++n) {
            __half *C_smem_ptr = &C_smem[(warpM * WM + m * MMA_M) * C_SMEM_STRIDE 
                                        + (warpN * WN + n * MMA_N)];
            wmma::store_matrix_sync(C_smem_ptr, acc[m][n], C_SMEM_STRIDE, wmma::mem_row_major);
        }
    }
    
    __syncthreads();
    
    // Vectorized copy from shared to global
    constexpr int NUM_THREADS = WARPS_M * WARPS_N * 32;
    constexpr int TOTAL_ELEMENTS = BM * BN;
    constexpr int ELEMENTS_PER_VEC = 8;
    constexpr int TOTAL_VECS = TOTAL_ELEMENTS / ELEMENTS_PER_VEC;
    constexpr int VECS_PER_THREAD = TOTAL_VECS / NUM_THREADS;
    
    static_assert(BN % 8 == 0, "BN must be divisible by 8 for vectorized stores");
    static_assert(TOTAL_VECS % NUM_THREADS == 0, "Vectors must divide evenly among threads");
    
    #pragma unroll
    for (int i = 0; i < VECS_PER_THREAD; ++i) {
        int vec_idx = tid + i * NUM_THREADS;
        
        int vecs_per_row = BN / ELEMENTS_PER_VEC;
        int row = vec_idx / vecs_per_row;
        int col8 = vec_idx % vecs_per_row;
        
        float4 val = *reinterpret_cast<float4*>(&C_smem[row * C_SMEM_STRIDE + col8 * 8]);
        *reinterpret_cast<float4*>(&C[row * N + col8 * 8]) = val;
    }
}
