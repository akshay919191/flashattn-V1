#ifndef MMA_HELPERS_CUH
#define MMA_HELPERS_CUH

#include <cuda_fp16.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <float.h>


#define WARP_FULL_MASK 0xffffffff
//// you can u use cvta generic to shared
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

template<int Rows, int Cols, int blockdim>
__device__ __forceinline__ void asyncLOAD_2D_TILE(
    const __half* matrix,
    uint32_t      smemptr,
    int           tid,
    int           smem_stride,   // physical shared stride
    int           total_rows,    // global matrix rows
    int           total_cols,    // global matrix cols
    int           global_stride, // global row stride
    int           row_tile,      // which row block: kv_tile or q_tile
    int           col_start      // starting column: qk_d or out_d
) {
    static_assert(Cols % 8 == 0, "Cols must be divisible by 8 for 16-byte cp.async loads");

    constexpr int halfs_per_async = 8; // 8 half = 16 bytes
    constexpr int vecs_per_tile = (Rows * Cols) / halfs_per_async;

    for (int i = tid; i < vecs_per_tile; i += blockdim) {
        int logical_offset = i * halfs_per_async;

        int local_row = logical_offset / Cols;
        int local_col = logical_offset % Cols;

        int global_row = row_tile * Rows + local_row;
        int global_col = col_start + local_col;

        uint32_t smemaddr =
            smemptr + (local_row * smem_stride + local_col) * sizeof(__half);

        bool is_valid =
            (global_row < total_rows) &&
            (global_col + 7 < total_cols);

        const __half* globalsrc =
            is_valid
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

/// for softmax reduction
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(WARP_FULL_MASK, val, offset));
    }
    return val; 
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(WARP_FULL_MASK, val, offset);
    }
    return val; 
}


__device__ __forceinline__ void mma_score_strided(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float*       __restrict__ C,
    int M,
    int K,
    int N,
    int A_STRIDE,
    int B_STRIDE,
    int C_STRIDE
)
{
    int tid  = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;

    int warps_per_block = blockDim.x >> 5;

    int group = lane >> 2;   // 0..7
    int tid4  = lane & 3;    // 0..3

    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;

    int num_m_tiles = (M + 15) / 16;
    int num_n_tiles = (N + 7)  / 8;
    int num_k_tiles = (K + 15) / 16;

    int total_tiles = num_m_tiles * num_n_tiles;

    for (int tile_idx = warp;
         tile_idx < total_tiles;
         tile_idx += warps_per_block)
    {
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

            a_frag[0] = *reinterpret_cast<const uint32_t*>(
                &A[a_row0 * A_STRIDE + k0]
            );

            a_frag[1] = *reinterpret_cast<const uint32_t*>(
                &A[a_row1 * A_STRIDE + k0]
            );

            a_frag[2] = *reinterpret_cast<const uint32_t*>(
                &A[a_row0 * A_STRIDE + k0 + 8]
            );

            a_frag[3] = *reinterpret_cast<const uint32_t*>(
                &A[a_row1 * A_STRIDE + k0 + 8]
            );

            int b_row = col_start + group;

            b_frag[0] = *reinterpret_cast<const uint32_t*>(
                &B[b_row * B_STRIDE + k0]
            );

            b_frag[1] = *reinterpret_cast<const uint32_t*>(
                &B[b_row * B_STRIDE + k0 + 8]
            );

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

__device__ __forceinline__ uint32_t pack_half2_u32(__half x, __half y)
{
    __half2 h2 = __halves2half2(x, y);
    return *reinterpret_cast<uint32_t*>(&h2);
}

__device__ __forceinline__ uint32_t pack_float2_to_half2_u32(float x, float y)
{
    __half2 h2 = __floats2half2_rn(x, y);
    return *reinterpret_cast<uint32_t*>(&h2);
}

__device__ __forceinline__ void mma_pv_accum_f32p_f16v(
    const float*  __restrict__ P,
    const __half* __restrict__ V,
    float*        __restrict__ Oacc,
    int M,
    int K,
    int N,
    int P_STRIDE,
    int V_STRIDE,
    int O_STRIDE
)
{
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

    for (int tile_idx = warp;
         tile_idx < total_tiles;
         tile_idx += warps_per_block)
    {
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

            __half v00 = (k0 < K && out_col < N)
                ? V[k0 * V_STRIDE + out_col]
                : __float2half(0.f);

            __half v01 = (k0 + 1 < K && out_col < N)
                ? V[(k0 + 1) * V_STRIDE + out_col]
                : __float2half(0.f);

            __half v10 = (k0 + 8 < K && out_col < N)
                ? V[(k0 + 8) * V_STRIDE + out_col]
                : __float2half(0.f);

            __half v11 = (k0 + 9 < K && out_col < N)
                ? V[(k0 + 9) * V_STRIDE + out_col]
                : __float2half(0.f);

            b_frag[0] = pack_half2_u32(v00, v01);
            b_frag[1] = pack_half2_u32(v10, v11);

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

        if (c_row0 < M && c_col0 < N) Oacc[c_row0 * O_STRIDE + c_col0] = acc[0];
        if (c_row0 < M && c_col1 < N) Oacc[c_row0 * O_STRIDE + c_col1] = acc[1];
        if (c_row1 < M && c_col0 < N) Oacc[c_row1 * O_STRIDE + c_col0] = acc[2];
        if (c_row1 < M && c_col1 < N) Oacc[c_row1 * O_STRIDE + c_col1] = acc[3];
    }
}

#endif 