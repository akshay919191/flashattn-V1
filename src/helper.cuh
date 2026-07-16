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

__device__ __forceinline__ uint32_t smem_u32_ptr(const void* ptr) {
    uint32_t addr;
    asm volatile(
        "{ .reg .u64 smem_addr;\n"
        "  cvta.to.shared.u64 smem_addr, %1;\n"
        "  cvt.u32.u64 %0, smem_addr;\n"
        "}\n"
        : "=r"(addr)
        : "l"(ptr)
    );
    return addr;
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

__device__ __forceinline__ void ldmatrix_x4(uint32_t* frag, uint32_t smem_int_ptr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3])
        : "r"(smem_int_ptr)
    );
}

__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t* frag, uint32_t smem_int_ptr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(frag[0]), "=r"(frag[1])
        : "r"(smem_int_ptr)
    );
}

__device__ __forceinline__ uint32_t pack_float2_to_half2_u32(float x, float y) {
    __half2 h2 = __floats2half2_rn(x, y);
    return *reinterpret_cast<uint32_t*>(&h2);
}




__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&frag)[4], uint32_t smem_int_ptr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3])
        : "r"(smem_int_ptr)
    );
}

// Helper to compute the exact smem ptr for ldmatrix
__device__ __forceinline__ uint32_t get_smem_ptr(const void* ptr, int row, int col, int stride) {
    // row is 0-7 within the 8x8 ldmatrix block
    // lane layout: lane % 8 determines the row.
    // lane / 8 determines which of the 4 matrices is being loaded.
    int lane = threadIdx.x % 32;
    int r = row + (lane % 8) + (lane / 16) * 8;
    int c = col + ((lane / 8) % 2) * 8;
    return smem_u32_ptr(reinterpret_cast<const __half*>(ptr) + r * stride + c);
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
    static_assert(Cols % 8 == 0, "Cols must be divisible by 8 for 16-byte cp.async loads");
    constexpr int halfs_per_async = 8;
    constexpr int vecs_per_tile = (Rows * Cols) / halfs_per_async;

    for (int i = tid; i < vecs_per_tile; i += blockdim) {
        int logical_offset = i * halfs_per_async;
        int local_row = logical_offset / Cols;
        int local_col = logical_offset % Cols;

        int global_row = row_tile * Rows + local_row;
        int global_col = col_start + local_col;

        uint32_t smemaddr = smemptr + (local_row * smem_stride + local_col) * sizeof(__half);

        bool is_valid = (global_row < total_rows) && (global_col + 7 < total_cols);

        const __half* globalsrc = is_valid 
            ? matrix + (size_t)global_row * global_stride + global_col 
            : matrix;

        int predicate = is_valid ? 1 : 0;

        asm volatile(
            "{\n"
            "  .reg .pred p;\n"
            "  .reg .u32 z;\n"
            "  mov.u32 z, 0;\n"
            "  setp.ne.b32 p, %2, 0;\n"
            "  @p  cp.async.cg.shared.global [%0], [%1], 16;\n"
            "  @!p st.shared.v4.b32 [%0], {z, z, z, z};\n"
            "}\n"
            :
            : "r"(smemaddr), "l"(globalsrc), "r"(predicate)
            : "memory"
        );
    }
}

// Computes C[M,N] = A[M,K] @ B[N,K]^T; A and B are row-major.
__device__ __forceinline__ void mma_score_strided(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float*       __restrict__ C,
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
            int b_row = col_start + group;

            a_frag[0] = (a_row0 < M && k0 + 1 < K) ? *reinterpret_cast<const uint32_t*>(&A[a_row0 * A_STRIDE + k0]) : 0;
            a_frag[1] = (a_row1 < M && k0 + 1 < K) ? *reinterpret_cast<const uint32_t*>(&A[a_row1 * A_STRIDE + k0]) : 0;
            a_frag[2] = (a_row0 < M && k0 + 9 < K) ? *reinterpret_cast<const uint32_t*>(&A[a_row0 * A_STRIDE + k0 + 8]) : 0;
            a_frag[3] = (a_row1 < M && k0 + 9 < K) ? *reinterpret_cast<const uint32_t*>(&A[a_row1 * A_STRIDE + k0 + 8]) : 0;

            b_frag[0] = (b_row < N && k0 + 1 < K) ? *reinterpret_cast<const uint32_t*>(&B[b_row * B_STRIDE + k0]) : 0;
            b_frag[1] = (b_row < N && k0 + 9 < K) ? *reinterpret_cast<const uint32_t*>(&B[b_row * B_STRIDE + k0 + 8]) : 0;

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
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

__device__ __forceinline__ uint32_t pack_half2_u32(__half x, __half y) {
    __half2 h2 = __halves2half2(x, y);
    return *reinterpret_cast<uint32_t*>(&h2);
}


template<int Br, int D, int O_STRIDE>
__device__ __forceinline__ void oaccSCALING_smem(
    float* __restrict__ Osmem,
    const float* __restrict__ Alphasmem
) {
    int tid = threadIdx.x;
    for (int i = tid; i < Br * D; i += blockDim.x) {
        int row = i / D;
        int col = i % D;
        Osmem[row * O_STRIDE + col] *= Alphasmem[row];
    }
}

__device__ __forceinline__ void mma_pv_accum_f32p_f16v(
    const float*  __restrict__ P,
    const __half* __restrict__ V,
    float*        __restrict__ Oacc,
    int M, int K, int N,
    int P_STRIDE, int V_STRIDE, int O_STRIDE
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

        int c_row0 = row_start + group;
        int c_row1 = row_start + group + 8;
        int c_col0 = col_start + tid4 * 2;
        int c_col1 = c_col0 + 1;

        float acc[4];
        acc[0] = (c_row0 < M && c_col0 < N) ? Oacc[c_row0 * O_STRIDE + c_col0] : 0.f;
        acc[1] = (c_row0 < M && c_col1 < N) ? Oacc[c_row0 * O_STRIDE + c_col1] : 0.f;
        acc[2] = (c_row1 < M && c_col0 < N) ? Oacc[c_row1 * O_STRIDE + c_col0] : 0.f;
        acc[3] = (c_row1 < M && c_col1 < N) ? Oacc[c_row1 * O_STRIDE + c_col1] : 0.f;

        for (int kt = 0; kt < num_k_tiles; kt++) {
            int k_start = kt * MMA_K;
            int k0 = k_start + tid4 * 2;

            uint32_t a_frag[4];
            uint32_t b_frag[2];

            int a_row0 = row_start + group;
            int a_row1 = row_start + group + 8;

            float p00 = (a_row0 < M && k0 < K)     ? P[a_row0 * P_STRIDE + k0]     : 0.f;
            float p01 = (a_row0 < M && k0 + 1 < K) ? P[a_row0 * P_STRIDE + k0 + 1] : 0.f;
            float p10 = (a_row1 < M && k0 < K)     ? P[a_row1 * P_STRIDE + k0]     : 0.f;
            float p11 = (a_row1 < M && k0 + 1 < K) ? P[a_row1 * P_STRIDE + k0 + 1] : 0.f;
            float p02 = (a_row0 < M && k0 + 8 < K) ? P[a_row0 * P_STRIDE + k0 + 8] : 0.f;
            float p03 = (a_row0 < M && k0 + 9 < K) ? P[a_row0 * P_STRIDE + k0 + 9] : 0.f;
            float p12 = (a_row1 < M && k0 + 8 < K) ? P[a_row1 * P_STRIDE + k0 + 8] : 0.f;
            float p13 = (a_row1 < M && k0 + 9 < K) ? P[a_row1 * P_STRIDE + k0 + 9] : 0.f;

            a_frag[0] = pack_float2_to_half2_u32(p00, p01);
            a_frag[1] = pack_float2_to_half2_u32(p10, p11);
            a_frag[2] = pack_float2_to_half2_u32(p02, p03);
            a_frag[3] = pack_float2_to_half2_u32(p12, p13);

            int out_col = col_start + group;

            __half v00 = (k0 < K && out_col < N) ? V[k0 * V_STRIDE + out_col] : __float2half(0.f);
            __half v01 = (k0 + 1 < K && out_col < N) ? V[(k0 + 1) * V_STRIDE + out_col] : __float2half(0.f);
            __half v10 = (k0 + 8 < K && out_col < N) ? V[(k0 + 8) * V_STRIDE + out_col] : __float2half(0.f);
            __half v11 = (k0 + 9 < K && out_col < N) ? V[(k0 + 9) * V_STRIDE + out_col] : __float2half(0.f);

            b_frag[0] = pack_half2_u32(v00, v01);
            b_frag[1] = pack_half2_u32(v10, v11);

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                  "r"(b_frag[0]), "r"(b_frag[1])
            );
        }

        if (c_row0 < M && c_col0 < N) Oacc[c_row0 * O_STRIDE + c_col0] = acc[0];
        if (c_row0 < M && c_col1 < N) Oacc[c_row0 * O_STRIDE + c_col1] = acc[1];
        if (c_row1 < M && c_col0 < N) Oacc[c_row1 * O_STRIDE + c_col0] = acc[2];
        if (c_row1 < M && c_col1 < N) Oacc[c_row1 * O_STRIDE + c_col1] = acc[3];
    }
}

__device__ __forceinline__ void mma_accum_f16p_f16v_smem(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float*        __restrict__ C,
    int M,
    int K,
    int N,
    int A_stride,
    int B_stride,
    int C_stride
) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int warps_per_block = blockDim.x >> 5;
    const int group = lane >> 2;
    const int lane4 = lane & 3;

    const int num_m_tiles = (M + 15) / 16;
    const int num_n_tiles = (N + 7) / 8;
    const int num_k_tiles = (K + 15) / 16;
    const int total_tiles = num_m_tiles * num_n_tiles;

    for (int tile = warp; tile < total_tiles; tile += warps_per_block) {
        const int m_tile = tile / num_n_tiles;
        const int n_tile = tile % num_n_tiles;
        const int row_start = m_tile * 16;
        const int col_start = n_tile * 8;

        const int out_row0 = row_start + group;
        const int out_row1 = out_row0 + 8;
        const int out_col0 = col_start + lane4 * 2;
        const int out_col1 = out_col0 + 1;

        float acc[4];
        acc[0] = (out_row0 < M && out_col0 < N)
            ? C[out_row0 * C_stride + out_col0]
            : 0.0f;
        acc[1] = (out_row0 < M && out_col1 < N)
            ? C[out_row0 * C_stride + out_col1]
            : 0.0f;
        acc[2] = (out_row1 < M && out_col0 < N)
            ? C[out_row1 * C_stride + out_col0]
            : 0.0f;
        acc[3] = (out_row1 < M && out_col1 < N)
            ? C[out_row1 * C_stride + out_col1]
            : 0.0f;

        for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
            const int k0 = k_tile * 16 + lane4 * 2;

            const int a_row0 = row_start + group;
            const int a_row1 = a_row0 + 8;
            const int b_col = col_start + group;

            const __half zero = __float2half(0.0f);

            const __half a00 = (a_row0 < M && k0 < K)
                ? A[a_row0 * A_stride + k0]
                : zero;
            const __half a01 = (a_row0 < M && k0 + 1 < K)
                ? A[a_row0 * A_stride + k0 + 1]
                : zero;
            const __half a10 = (a_row1 < M && k0 < K)
                ? A[a_row1 * A_stride + k0]
                : zero;
            const __half a11 = (a_row1 < M && k0 + 1 < K)
                ? A[a_row1 * A_stride + k0 + 1]
                : zero;
            const __half a20 = (a_row0 < M && k0 + 8 < K)
                ? A[a_row0 * A_stride + k0 + 8]
                : zero;
            const __half a21 = (a_row0 < M && k0 + 9 < K)
                ? A[a_row0 * A_stride + k0 + 9]
                : zero;
            const __half a30 = (a_row1 < M && k0 + 8 < K)
                ? A[a_row1 * A_stride + k0 + 8]
                : zero;
            const __half a31 = (a_row1 < M && k0 + 9 < K)
                ? A[a_row1 * A_stride + k0 + 9]
                : zero;

            uint32_t a_frag[4];
            a_frag[0] = pack_half2_u32(a00, a01);
            a_frag[1] = pack_half2_u32(a10, a11);
            a_frag[2] = pack_half2_u32(a20, a21);
            a_frag[3] = pack_half2_u32(a30, a31);

            // Correct row-major B[K,N] indexing. The old helper used
            // B[b_col * stride + k], which interpreted B as [N,K].
            const __half b00 = (k0 < K && b_col < N)
                ? B[k0 * B_stride + b_col]
                : zero;
            const __half b01 = (k0 + 1 < K && b_col < N)
                ? B[(k0 + 1) * B_stride + b_col]
                : zero;
            const __half b10 = (k0 + 8 < K && b_col < N)
                ? B[(k0 + 8) * B_stride + b_col]
                : zero;
            const __half b11 = (k0 + 9 < K && b_col < N)
                ? B[(k0 + 9) * B_stride + b_col]
                : zero;

            uint32_t b_frag[2];
            b_frag[0] = pack_half2_u32(b00, b01);
            b_frag[1] = pack_half2_u32(b10, b11);

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]),
                  "+f"(acc[2]), "+f"(acc[3])
                : "r"(a_frag[0]), "r"(a_frag[1]),
                  "r"(a_frag[2]), "r"(a_frag[3]),
                  "r"(b_frag[0]), "r"(b_frag[1])
            );
        }

        if (out_row0 < M && out_col0 < N) {
            C[out_row0 * C_stride + out_col0] = acc[0];
        }
        if (out_row0 < M && out_col1 < N) {
            C[out_row0 * C_stride + out_col1] = acc[1];
        }
        if (out_row1 < M && out_col0 < N) {
            C[out_row1 * C_stride + out_col0] = acc[2];
        }
        if (out_row1 < M && out_col1 < N) {
            C[out_row1 * C_stride + out_col1] = acc[3];
        }
    }
}

template<
    int M,
    int K,
    int N,
    int A_STRIDE,
    int B_STRIDE,
    int C_STRIDE
>
__device__ __forceinline__ void mma_score_f16_tiled(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C
) {
    static_assert(M > 0 && M % 16 == 0, "M must be divisible by 16");
    static_assert(K > 0 && K % 16 == 0, "K must be divisible by 16");
    static_assert(N > 0 && N % 8 == 0, "N must be divisible by 8");

    constexpr int kWarps = 4;
    constexpr int kNumMTiles = M / 16;
    constexpr int kNumNTiles = N / 8;
    constexpr int kNumKTiles = K / 16;
    constexpr int kTotalTiles = kNumMTiles * kNumNTiles;
    constexpr int kTilesPerWarp = (kTotalTiles + kWarps - 1) / kWarps;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int group = lane >> 2;
    const int lane4 = lane & 3;

    #pragma unroll
    for (int slot = 0; slot < kTilesPerWarp; ++slot) {
        const int tile = warp + slot * kWarps;

        if (tile < kTotalTiles) {
            const int m_tile = tile / kNumNTiles;
            const int n_tile = tile % kNumNTiles;

            float c0 = 0.0f;
            float c1 = 0.0f;
            float c2 = 0.0f;
            float c3 = 0.0f;

            #pragma unroll
            for (int k_tile = 0; k_tile < kNumKTiles; ++k_tile) {
                uint32_t a_frag[4];
                const int a_row = m_tile * 16 + (lane & 15);
                const int a_col =
                    k_tile * 16 + ((lane < 16) ? 0 : 8);

                ldmatrix_x4(
                    a_frag,
                    smem_u32_ptr(A + a_row * A_STRIDE + a_col)
                );

                uint32_t b_frag[2];
                const int lane16 = lane & 15;
                const int b_row = n_tile * 8 + (lane16 & 7);
                const int b_col =
                    k_tile * 16 + ((lane16 >> 3) * 8);

                ldmatrix_x2(
                    b_frag,
                    smem_u32_ptr(B + b_row * B_STRIDE + b_col)
                );

                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%0, %1, %2, %3};\n"
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

// ----------------------------------------------------------------------------
// Register-resident row-major MMA accumulation
// ----------------------------------------------------------------------------
// Computes C[M,N] += A[M,K] @ B[K,N], keeping each warp's C fragments in the
// caller-provided register array. A and B must reside in shared memory.
template<
    int M,
    int K,
    int N,
    int A_STRIDE,
    int B_STRIDE,
    int ACC_COUNT
>
__device__ __forceinline__ void mma_accum_f16_registers(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float (&acc)[ACC_COUNT]
) {
    static_assert(M > 0 && M % 16 == 0, "M must be divisible by 16");
    static_assert(K > 0 && K % 16 == 0, "K must be divisible by 16");
    static_assert(N > 0 && N % 8 == 0, "N must be divisible by 8");

    constexpr int kWarps = 4;
    constexpr int kNumMTiles = M / 16;
    constexpr int kNumNTiles = N / 8;
    constexpr int kNumKTiles = K / 16;
    constexpr int kTotalTiles = kNumMTiles * kNumNTiles;
    constexpr int kTilesPerWarp = (kTotalTiles + kWarps - 1) / kWarps;

    static_assert(
        ACC_COUNT == kTilesPerWarp * 4,
        "Accumulator array has the wrong size"
    );

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    #pragma unroll
    for (int slot = 0; slot < kTilesPerWarp; ++slot) {
        const int tile = warp + slot * kWarps;

        if (tile < kTotalTiles) {
            const int m_tile = tile / kNumNTiles;
            const int n_tile = tile % kNumNTiles;

            float c0 = acc[slot * 4 + 0];
            float c1 = acc[slot * 4 + 1];
            float c2 = acc[slot * 4 + 2];
            float c3 = acc[slot * 4 + 3];

            #pragma unroll
            for (int k_tile = 0; k_tile < kNumKTiles; ++k_tile) {
                uint32_t a_frag[4];
                const int a_row = m_tile * 16 + (lane & 15);
                const int a_col =
                    k_tile * 16 + ((lane < 16) ? 0 : 8);

                ldmatrix_x4(
                    a_frag,
                    smem_u32_ptr(A + a_row * A_STRIDE + a_col)
                );

                uint32_t b_frag[2];
                const int b_row = k_tile * 16 + (lane & 15);
                const int b_col = n_tile * 8;

                // B is physical row-major [K,N]. A transposed ldmatrix load
                // produces the mma.row.col B fragment directly.
                ldmatrix_x2_trans(
                    b_frag,
                    smem_u32_ptr(B + b_row * B_STRIDE + b_col)
                );

                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%0, %1, %2, %3};\n"
                    : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                    : "r"(a_frag[0]), "r"(a_frag[1]),
                      "r"(a_frag[2]), "r"(a_frag[3]),
                      "r"(b_frag[0]), "r"(b_frag[1])
                );
            }

            acc[slot * 4 + 0] = c0;
            acc[slot * 4 + 1] = c1;
            acc[slot * 4 + 2] = c2;
            acc[slot * 4 + 3] = c3;
        }
    }
}

// Stores register-resident m16n8 accumulator fragments to a row-major fp16
// global matrix. row_offset identifies this block's first global output row.
template<int M, int N, int ACC_COUNT>
__device__ __forceinline__ void store_f16_registers(
    const float (&acc)[ACC_COUNT],
    __half* __restrict__ output,
    int row_offset,
    int total_rows,
    int output_stride
) {
    constexpr int kWarps = 4;
    constexpr int kNumMTiles = M / 16;
    constexpr int kNumNTiles = N / 8;
    constexpr int kTotalTiles = kNumMTiles * kNumNTiles;
    constexpr int kTilesPerWarp = (kTotalTiles + kWarps - 1) / kWarps;

    static_assert(M > 0 && M % 16 == 0, "M must be divisible by 16");
    static_assert(N > 0 && N % 8 == 0, "N must be divisible by 8");
    static_assert(
        ACC_COUNT == kTilesPerWarp * 4,
        "Accumulator array has the wrong size"
    );

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
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
                output[
                    static_cast<size_t>(global_row0) * output_stride + col0
                ] = __float2half(acc[slot * 4 + 0]);
                output[
                    static_cast<size_t>(global_row0) * output_stride + col1
                ] = __float2half(acc[slot * 4 + 1]);
            }

            if (global_row1 < total_rows) {
                output[
                    static_cast<size_t>(global_row1) * output_stride + col0
                ] = __float2half(acc[slot * 4 + 2]);
                output[
                    static_cast<size_t>(global_row1) * output_stride + col1
                ] = __float2half(acc[slot * 4 + 3]);
            }
        }
    }
}

#endif  // MMA_HELPERS_CUH