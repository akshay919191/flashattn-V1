#pragma once

#ifndef MMA_HELPERS_CUH
#define MMA_HELPERS_CUH

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#define WARP_FULL_MASK 0xffffffff


__device__ __forceinline__ uint32_t smem_u32_ptr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ uint32_t get_smem_ptr(
    const void* ptr, int row, int col, int stride
) {
    int lane = threadIdx.x % 32;
    int r = row + (lane % 8) + (lane / 16) * 8;
    int c = col + ((lane / 8) % 2) * 8;
    return smem_u32_ptr(reinterpret_cast<const __half*>(ptr) + r * stride + c);
}

__device__ __forceinline__
void ldmatrix_x2(uint32_t* frag, uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
        "{%0, %1}, [%2];\n"
        : "=r"(frag[0]), "=r"(frag[1])
        : "r"(addr)
    );
}

__device__ __forceinline__
void ldmatrix_x4(uint32_t* frag, uint32_t smem_int_ptr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3])
        : "r"(smem_int_ptr)
    );
}

__device__ __forceinline__
void ldmatrix_x2_trans(uint32_t* frag, uint32_t smem_int_ptr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(frag[0]), "=r"(frag[1])
        : "r"(smem_int_ptr)
    );
}

__device__ __forceinline__
void ldmatrix_x4_trans(uint32_t (&frag)[4], uint32_t smem_int_ptr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3])
        : "r"(smem_int_ptr)
    );
}


__device__ __forceinline__
uint32_t pack_float2_to_half2_u32(float x, float y) {
    __half2 h2 = __floats2half2_rn(x, y);
    return *reinterpret_cast<uint32_t*>(&h2);
}

__device__ __forceinline__
uint32_t pack_half2_u32(__half x, __half y) {
    __half2 h2 = __halves2half2(x, y);
    return *reinterpret_cast<uint32_t*>(&h2);
} 

template<int Rows, int Cols, int blockdim>
__device__ __forceinline__ void asyncLOAD_2D_TILE(
    const __half* matrix,
    uint32_t      smemptr,
    int           tid,
    int           smem_stride,
    int           total_rows,
    int           total_cols,
    int           global_stride,
    int           row_tile,
    int           col_start
) {
    static_assert(Cols % 8 == 0,
                  "Cols must be divisible by 8 for 16-byte cp.async loads");
    constexpr int VEC          = 8;
    constexpr int chunks_x     = Cols / VEC;
    constexpr int total_chunks = (Rows * Cols) / VEC;

    const int full_cols = total_cols & ~7;
    const int tail      = total_cols - full_cols;
    const bool vec_ok   = (global_stride % VEC) == 0;

    if (vec_ok) {
        for (int i = tid; i < total_chunks; i += blockdim) {
            const int lr = i / chunks_x;
            const int c8 = (i % chunks_x) * VEC;
            const int gr = row_tile * Rows + lr;
            const int gc = col_start + c8;

            const uint32_t smemaddr =
                smemptr + (lr * smem_stride + c8) * (int)sizeof(__half);

            const bool row_ok  = (gr < total_rows);
            const bool do_load = row_ok && (c8 < full_cols);
            const bool skip    = row_ok && (c8 == full_cols) && (tail != 0);

            const __half* src = do_load
                ? matrix + (size_t)gr * global_stride + gc
                : matrix;
            const int nbytes = do_load ? 16 : 0;

            if (!skip) {
                asm volatile(
                    "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                    :
                    : "r"(smemaddr), "l"(src), "r"(nbytes)
                    : "memory"
                );
            }
        }
        if (tail != 0) {
            for (int r = tid; r < Rows; r += blockdim) {
                const int gr = row_tile * Rows + r;
                const bool row_ok = (gr < total_rows);
                const uint32_t d =
                    smemptr + (r * smem_stride + full_cols) * (int)sizeof(__half);
                for (int j = 0; j < VEC; ++j) {
                    uint16_t v = 0;
                    if (row_ok && j < tail) {
                        const __half h =
                            matrix[(size_t)gr * global_stride + full_cols + j];
                        v = *reinterpret_cast<const uint16_t*>(&h);
                    }
                    asm volatile("st.shared.u16 [%0], %1;\n"
                                 :: "r"(d + j * (int)sizeof(__half)), "h"(v)
                                 : "memory");
                }
            }
        }
    } else {
        for (int i = tid; i < Rows * Cols; i += blockdim) {
            const int lr = i / Cols;
            const int lc = i % Cols;
            const int gr = row_tile * Rows + lr;
            const int gc = col_start + lc;
            uint16_t v = 0;
            if (gr < total_rows && gc < total_cols) {
                const __half h = matrix[(size_t)gr * global_stride + gc];
                v = *reinterpret_cast<const uint16_t*>(&h);
            }
            const uint32_t smemaddr =
                smemptr + (lr * smem_stride + lc) * (int)sizeof(__half);
            asm volatile("st.shared.u16 [%0], %1;\n"
                         :: "r"(smemaddr), "h"(v) : "memory");
        }
    }
}
 

__device__ __forceinline__ void mma_score_strided(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float*        __restrict__ C,
    int M, int K, int N,
    int A_STRIDE, int B_STRIDE, int C_STRIDE
) {
    int tid  = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int warps_per_block = blockDim.x >> 5;
    int group = lane >> 2;
    int tid4  = lane & 3;

    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;

    int num_m_tiles = (M + 15) / 16;
    int num_n_tiles = (N + 7)  / 8;
    int num_k_tiles = (K + 15) / 16;
    int total_tiles = num_m_tiles * num_n_tiles;

    for (int tile_idx = warp; tile_idx < total_tiles; tile_idx += warps_per_block) {
        int mt = tile_idx / num_n_tiles;
        int nt = tile_idx % num_n_tiles;
        int row_start = mt * MMA_M;
        int col_start = nt * MMA_N;

        float acc[4] = {0.f, 0.f, 0.f, 0.f};

        for (int kt = 0; kt < num_k_tiles; kt++) {
            int k_start = kt * MMA_K;
            int k0 = k_start + tid4 * 2;

            uint32_t a_frag[4];
            uint32_t b_frag[2];

            int a_row0 = row_start + group;
            int a_row1 = row_start + group + 8;
            int b_row  = col_start + group;

            a_frag[0] = (a_row0 < M && k0 + 1 < K) ?
                *reinterpret_cast<const uint32_t*>(&A[a_row0 * A_STRIDE + k0]) : 0;
            a_frag[1] = (a_row1 < M && k0 + 1 < K) ?
                *reinterpret_cast<const uint32_t*>(&A[a_row1 * A_STRIDE + k0]) : 0;
            a_frag[2] = (a_row0 < M && k0 + 9 < K) ?
                *reinterpret_cast<const uint32_t*>(&A[a_row0 * A_STRIDE + k0 + 8]) : 0;
            a_frag[3] = (a_row1 < M && k0 + 9 < K) ?
                *reinterpret_cast<const uint32_t*>(&A[a_row1 * A_STRIDE + k0 + 8]) : 0;

            b_frag[0] = (b_row < N && k0 + 1 < K) ?
                *reinterpret_cast<const uint32_t*>(&B[b_row * B_STRIDE + k0]) : 0;
            b_frag[1] = (b_row < N && k0 + 9 < K) ?
                *reinterpret_cast<const uint32_t*>(&B[b_row * B_STRIDE + k0 + 8]) : 0;

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a_frag[0]), "r"(a_frag[1]),
                  "r"(a_frag[2]), "r"(a_frag[3]),
                  "r"(b_frag[0]), "r"(b_frag[1])
            );
        }

        int c_row0 = row_start + group;
        int c_row1 = row_start + group + 8;
        int c_col0 = col_start + tid4 * 2;
        int c_col1 = c_col0 + 1;

        if (c_row0 < M && c_col0 < N) C[c_row0 * C_STRIDE + c_col0] = acc[0];
        if (c_row0 < M && c_col1 < N) C[c_row0 * C_STRIDE + c_col1] = acc[1];
        if (c_row1 < M && c_col0 < N) C[c_row1 * C_STRIDE + c_col0] = acc[2];
        if (c_row1 < M && c_col1 < N) C[c_row1 * C_STRIDE + c_col1] = acc[3];
    }
}
 

template<int M, int K, int N, int A_STRIDE, int B_STRIDE, int C_STRIDE>
__device__ __forceinline__ void mma_score_f16_tiled(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float*        __restrict__ C
) {
    static_assert(M > 0 && M % 16 == 0, "M must be divisible by 16");
    static_assert(K > 0 && K % 16 == 0, "K must be divisible by 16");
    static_assert(N > 0 && N % 8  == 0, "N must be divisible by 8");

    constexpr int kWarps       = 4;
    constexpr int kNumMTiles   = M / 16;
    constexpr int kNumNTiles   = N / 8;
    constexpr int kNumKTiles   = K / 16;
    constexpr int kTotalTiles  = kNumMTiles * kNumNTiles;
    constexpr int kTilesPerWarp = (kTotalTiles + kWarps - 1) / kWarps;

    const int warp  = threadIdx.x >> 5;
    const int lane  = threadIdx.x & 31;
    const int group = lane >> 2;
    const int lane4 = lane & 3;

    #pragma unroll
    for (int slot = 0; slot < kTilesPerWarp; ++slot) {
        const int tile = warp + slot * kWarps;
        if (tile < kTotalTiles) {
            const int m_tile = tile / kNumNTiles;
            const int n_tile = tile % kNumNTiles;

            float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

            #pragma unroll
            for (int k_tile = 0; k_tile < kNumKTiles; ++k_tile) {
                uint32_t a_frag[4];
                const int a_row = m_tile * 16 + (lane & 15);
                const int a_col = k_tile * 16 + ((lane < 16) ? 0 : 8);
                ldmatrix_x4(a_frag, smem_u32_ptr(A + a_row * A_STRIDE + a_col));

                uint32_t b_frag[2];
                const int lane16 = lane & 15;
                const int b_row  = n_tile * 8 + (lane16 & 7);
                const int b_col  = k_tile * 16 + ((lane16 >> 3) * 8);
                ldmatrix_x2(b_frag, smem_u32_ptr(B + b_row * B_STRIDE + b_col));

                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                    : "r"(a_frag[0]), "r"(a_frag[1]),
                      "r"(a_frag[2]), "r"(a_frag[3]),
                      "r"(b_frag[0]), "r"(b_frag[1])
                );
            }

            const int row0 = m_tile * 16 + group;
            const int row1 = row0 + 8;
            const int col0 = n_tile * 8 + lane4 * 2;
            const int col1 = col0 + 1;

            C[row0 * C_STRIDE + col0] = c0;
            C[row0 * C_STRIDE + col1] = c1;
            C[row1 * C_STRIDE + col0] = c2;
            C[row1 * C_STRIDE + col1] = c3;
        }
    }
}
 

template<int M, int K, int N, int A_STRIDE, int B_STRIDE, int ACC_COUNT>
__device__ __forceinline__ void mma_accum_f16_registers(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float (&acc)[ACC_COUNT]
) {
    static_assert(M > 0 && M % 16 == 0, "M must be divisible by 16");
    static_assert(K > 0 && K % 16 == 0, "K must be divisible by 16");
    static_assert(N > 0 && N % 8  == 0, "N must be divisible by 8");

    constexpr int kWarps       = 4;
    constexpr int kNumMTiles   = M / 16;
    constexpr int kNumNTiles   = N / 8;
    constexpr int kNumKTiles   = K / 16;
    constexpr int kTotalTiles  = kNumMTiles * kNumNTiles;
    constexpr int kTilesPerWarp = (kTotalTiles + kWarps - 1) / kWarps;

    static_assert(ACC_COUNT == kTilesPerWarp * 4,
                  "Accumulator array has the wrong size");

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    #pragma unroll
    for (int slot = 0; slot < kTilesPerWarp; ++slot) {
        const int tile = warp + slot * kWarps;
        if (tile < kTotalTiles) {
            const int m_tile = tile / kNumNTiles;
            const int n_tile = tile % kNumNTiles;

            float c0 = acc[slot * 4 + 0], c1 = acc[slot * 4 + 1];
            float c2 = acc[slot * 4 + 2], c3 = acc[slot * 4 + 3];

            #pragma unroll
            for (int k_tile = 0; k_tile < kNumKTiles; ++k_tile) {
                uint32_t a_frag[4];
                const int a_row = m_tile * 16 + (lane & 15);
                const int a_col = k_tile * 16 + ((lane < 16) ? 0 : 8);
                ldmatrix_x4(a_frag, smem_u32_ptr(A + a_row * A_STRIDE + a_col));

                uint32_t b_frag[2];
                const int b_row = k_tile * 16 + (lane & 15);
                const int b_col = n_tile * 8;
                ldmatrix_x2_trans(b_frag,
                                  smem_u32_ptr(B + b_row * B_STRIDE + b_col));

                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                    : "r"(a_frag[0]), "r"(a_frag[1]),
                      "r"(a_frag[2]), "r"(a_frag[3]),
                      "r"(b_frag[0]), "r"(b_frag[1])
                );
            }

            acc[slot * 4 + 0] = c0; acc[slot * 4 + 1] = c1;
            acc[slot * 4 + 2] = c2; acc[slot * 4 + 3] = c3;
        }
    }
}
 

template<int M, int N, int ACC_COUNT>
__device__ __forceinline__ void store_f16_registers(
    const float (&acc)[ACC_COUNT],
    __half*     __restrict__ output,
    int row_offset,
    int total_rows,
    int output_stride
) {
    constexpr int kWarps       = 4;
    constexpr int kNumMTiles   = M / 16;
    constexpr int kNumNTiles   = N / 8;
    constexpr int kTotalTiles  = kNumMTiles * kNumNTiles;
    constexpr int kTilesPerWarp = (kTotalTiles + kWarps - 1) / kWarps;

    static_assert(M > 0 && M % 16 == 0, "M must be divisible by 16");
    static_assert(N > 0 && N % 8  == 0, "N must be divisible by 8");
    static_assert(ACC_COUNT == kTilesPerWarp * 4,
                  "Accumulator array has the wrong size");

    const int warp  = threadIdx.x >> 5;
    const int lane  = threadIdx.x & 31;
    const int group = lane >> 2;
    const int lane4 = lane & 3;

    #pragma unroll
    for (int slot = 0; slot < kTilesPerWarp; ++slot) {
        const int tile = warp + slot * kWarps;
        if (tile < kTotalTiles) {
            const int m_tile = tile / kNumNTiles;
            const int n_tile = tile % kNumNTiles;

            const int local_row0 = m_tile * 16 + group;
            const int local_row1 = local_row0 + 8;
            const int col0 = n_tile * 8 + lane4 * 2;
            const int col1 = col0 + 1;

            const int global_row0 = row_offset + local_row0;
            const int global_row1 = row_offset + local_row1;

            if (global_row0 < total_rows) {
                output[static_cast<size_t>(global_row0) * output_stride + col0] =
                    __float2half(acc[slot * 4 + 0]);
                output[static_cast<size_t>(global_row0) * output_stride + col1] =
                    __float2half(acc[slot * 4 + 1]);
            }
            if (global_row1 < total_rows) {
                output[static_cast<size_t>(global_row1) * output_stride + col0] =
                    __float2half(acc[slot * 4 + 2]);
                output[static_cast<size_t>(global_row1) * output_stride + col1] =
                    __float2half(acc[slot * 4 + 3]);
            }
        }
    }
}
template<int Rows, int D, int STRIDE>
__device__ __forceinline__ void load_tile_full(
    const __half* __restrict__ g,
    uint32_t smem,
    int tid,
    int tile
) {
    constexpr int CPR = D / 8;
    static_assert(D % 8 == 0, "D must be divisible by 8");
    static_assert(128 % CPR == 0, "Unsupported column count");

    constexpr int RPP = 128 / CPR;

    static_assert(Rows % RPP == 0, "Unsupported row count");

    constexpr int PASSES = Rows / RPP;

    const int r = tid / CPR;
    const int c = (tid % CPR) * 8;

    const __half* src =
        g + (static_cast<size_t>(tile) * Rows + r) * D + c;

    const uint32_t dst =
        smem + static_cast<uint32_t>(
            (r * STRIDE + c) * sizeof(__half));

    #pragma unroll
    for (int j = 0; j < PASSES; ++j) {
        const uint32_t dst_j =
            dst + static_cast<uint32_t>(
                j * RPP * STRIDE * static_cast<int>(sizeof(__half)));

        const __half* src_j =
            src + static_cast<size_t>(j) * RPP * D;

        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16;\n"
            :
            : "r"(dst_j),
            "l"(src_j)
            : "memory"
        );
    }
}
#endif // MMA_HELPERS_CUH
