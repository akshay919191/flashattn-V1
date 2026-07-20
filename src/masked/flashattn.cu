#ifndef FLASHATTN_MASKED_SINGLE_FILE_CUH
#define FLASHATTN_MASKED_SINGLE_FILE_CUH




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


__device__ __forceinline__ uint32_t get_smem_ptr(const void* ptr, int row, int col, int stride) {
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

#endif








template<int Br, int Bc, int D_PAD, bool masked>
__global__ void flashattn_fwd(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
          __half* __restrict__ output,
          float*  __restrict__ Logsum,
          int actual_D,
          int Skv,
          int Sq
) {

    static_assert(Br == 64, "This kernel requires Br == 64");
    static_assert(Bc > 0 && Bc % 16 == 0,
                  "Bc must be positive and divisible by 16");
    static_assert(D_PAD > 0 && D_PAD % 16 == 0,
                  "D_PAD must be positive and divisible by 16");

    if (blockDim.x != 128) return;
    if (actual_D <= 0 || actual_D > D_PAD || (actual_D & 7) != 0 || Sq <= 0 || Skv <= 0) return;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int lane4 = lane & 3;

    const int batchid = blockIdx.x;
    const int headid = blockIdx.y;
    const int tileid = blockIdx.z;
    const int num_heads = gridDim.y;

    const long long q_base =
        (static_cast<long long>(batchid) * num_heads + headid) *
        Sq * actual_D;

    const long long kv_base =
        (static_cast<long long>(batchid) * num_heads + headid) *
        Skv * actual_D;

    const long long stat_base =
        (static_cast<long long>(batchid) * num_heads + headid) * Sq;

    const __half* Qptr = Q + q_base;
    const __half* Kptr = K + kv_base;
    const __half* Vptr = V + kv_base;
          __half* Optr = output + q_base;
          float*  Lptr = Logsum + stat_base;

    const int Tr = (Sq + Br - 1) / Br;
    const int Tc = (Skv + Bc - 1) / Bc;

    if (tileid >= Tr) return;

    const int q_block_start = tileid * Br;
    const int q_block_last_unclamped = q_block_start + Br - 1;
    const int q_block_last =
        q_block_last_unclamped < Sq
            ? q_block_last_unclamped
            : Sq - 1;

    int kv_tiles_to_process = Tc;
    if constexpr (masked) {
        const int causal_tiles = q_block_last / Bc + 1;
        if (causal_tiles < kv_tiles_to_process) {
            kv_tiles_to_process = causal_tiles;
        }
    }

    constexpr int PAD = 8;
    constexpr int Q_STRIDE = D_PAD + PAD;
    constexpr int K_STRIDE = D_PAD + PAD;
    constexpr int V_STRIDE = D_PAD + PAD;

    extern __shared__ char smem_raw[];
    char* ptr = smem_raw;

    auto align_ptr = [&](size_t alignment = 16) {
        const uintptr_t value = reinterpret_cast<uintptr_t>(ptr);
        ptr = reinterpret_cast<char*>(
            (value + alignment - 1) & ~(alignment - 1)
        );
    };

    align_ptr();
    __half* Qsmem = reinterpret_cast<__half*>(ptr);
    ptr += Br * Q_STRIDE * sizeof(__half);

    align_ptr();
    __half* Ksmem0 = reinterpret_cast<__half*>(ptr);
    ptr += Bc * K_STRIDE * sizeof(__half);

    align_ptr();
    __half* Ksmem1 = reinterpret_cast<__half*>(ptr);
    ptr += Bc * K_STRIDE * sizeof(__half);
    __half* Ksmem[2] = {Ksmem0, Ksmem1};

    align_ptr();
    __half* Vsmem0 = reinterpret_cast<__half*>(ptr);
    ptr += Bc * V_STRIDE * sizeof(__half);

    align_ptr();
    __half* Vsmem1 = reinterpret_cast<__half*>(ptr);
    __half* Vsmem[2] = {Vsmem0, Vsmem1};

    constexpr int Dk = D_PAD / 16;
    constexpr int Bk = Bc / 8;
    constexpr int Dv = D_PAD / 8;

    float O_frag[Dv * 4] = {0.0f};
    float m_frag[2] = {-FLT_MAX, -FLT_MAX};
    float l_frag[2] = {0.0f, 0.0f};

    const float scale = 1.0f / sqrtf(static_cast<float>(actual_D));


    asyncLOAD_2D_TILE<Br, D_PAD, 128>(
        Qptr,
        smem_u32_ptr(Qsmem),
        tid,
        Q_STRIDE,
        Sq,
        actual_D,
        actual_D,
        tileid,
        0
    );

    asyncLOAD_2D_TILE<Bc, D_PAD, 128>(
        Kptr,
        smem_u32_ptr(Ksmem[0]),
        tid,
        K_STRIDE,
        Skv,
        actual_D,
        actual_D,
        0,
        0
    );

    asyncLOAD_2D_TILE<Bc, D_PAD, 128>(
        Vptr,
        smem_u32_ptr(Vsmem[0]),
        tid,
        V_STRIDE,
        Skv,
        actual_D,
        actual_D,
        0,
        0
    );

    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    for (int kv_tile = 0;
         kv_tile < kv_tiles_to_process;
         ++kv_tile) {
        const int current_stage = kv_tile & 1;
        const int next_stage = current_stage ^ 1;
        const int next_kv_tile = kv_tile + 1;

        if (next_kv_tile < kv_tiles_to_process) {
            asyncLOAD_2D_TILE<Bc, D_PAD, 128>(
                Kptr,
                smem_u32_ptr(Ksmem[next_stage]),
                tid,
                K_STRIDE,
                Skv,
                actual_D,
                actual_D,
                next_kv_tile,
                0
            );

            asyncLOAD_2D_TILE<Bc, D_PAD, 128>(
                Vptr,
                smem_u32_ptr(Vsmem[next_stage]),
                tid,
                V_STRIDE,
                Skv,
                actual_D,
                actual_D,
                next_kv_tile,
                0
            );

            asm volatile("cp.async.commit_group;\n");
        }

        float S_frag[Bk * 4] = {0.0f};


        const int q_row = warp * 16 + (lane & 15);
        const int lane16 = lane & 15;

        #pragma unroll
        for (int ks = 0; ks < Dk; ++ks) {
            uint32_t q_frag[4];
            const int q_col =
                ks * 16 + ((lane < 16) ? 0 : 8);

            ldmatrix_x4(
                q_frag,
                smem_u32_ptr(
                    Qsmem + q_row * Q_STRIDE + q_col
                )
            );

            #pragma unroll
            for (int kb = 0; kb < Bk; ++kb) {
                uint32_t k_frag[2];
                const int k_row =
                    kb * 8 + (lane16 & 7);
                const int k_col =
                    ks * 16 + ((lane16 >> 3) * 8);

                ldmatrix_x2(
                    k_frag,
                    smem_u32_ptr(
                        Ksmem[current_stage] +
                        k_row * K_STRIDE +
                        k_col
                    )
                );

                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%0, %1, %2, %3};\n"
                    : "+f"(S_frag[kb * 4 + 0]),
                      "+f"(S_frag[kb * 4 + 1]),
                      "+f"(S_frag[kb * 4 + 2]),
                      "+f"(S_frag[kb * 4 + 3])
                    : "r"(q_frag[0]), "r"(q_frag[1]),
                      "r"(q_frag[2]), "r"(q_frag[3]),
                      "r"(k_frag[0]), "r"(k_frag[1])
                );
            }
        }

        const int kv_start = kv_tile * Bc;
        const int kv_last_unclamped = kv_start + Bc - 1;
        const int kv_real_last =
            kv_last_unclamped < Skv
                ? kv_last_unclamped
                : Skv - 1;

        const bool q_tile_full = q_block_last_unclamped < Sq;
        const bool kv_tile_full = kv_last_unclamped < Skv;

        bool needs_causal_mask = false;
        if constexpr (masked) {
            const bool fully_valid =
                kv_real_last <= q_block_start;
            needs_causal_mask = !fully_valid;
        }

        const bool needs_any_mask =
            !q_tile_full || !kv_tile_full || needs_causal_mask;

        float tile_max[2] = {-FLT_MAX, -FLT_MAX};

        if (!needs_any_mask) {
            #pragma unroll
            for (int kb = 0; kb < Bk; ++kb) {
                const float s0 = S_frag[kb * 4 + 0] * scale;
                const float s1 = S_frag[kb * 4 + 1] * scale;
                const float s2 = S_frag[kb * 4 + 2] * scale;
                const float s3 = S_frag[kb * 4 + 3] * scale;

                S_frag[kb * 4 + 0] = s0;
                S_frag[kb * 4 + 1] = s1;
                S_frag[kb * 4 + 2] = s2;
                S_frag[kb * 4 + 3] = s3;

                float max0 = fmaxf(s0, s1);
                float max1 = fmaxf(s2, s3);

                max0 = fmaxf(
                    max0,
                    __shfl_xor_sync(0xffffffffu, max0, 1, 4)
                );
                max0 = fmaxf(
                    max0,
                    __shfl_xor_sync(0xffffffffu, max0, 2, 4)
                );

                max1 = fmaxf(
                    max1,
                    __shfl_xor_sync(0xffffffffu, max1, 1, 4)
                );
                max1 = fmaxf(
                    max1,
                    __shfl_xor_sync(0xffffffffu, max1, 2, 4)
                );

                tile_max[0] = fmaxf(tile_max[0], max0);
                tile_max[1] = fmaxf(tile_max[1], max1);
            }
        } else {
            const int query0 =
                q_block_start + warp * 16 + lane / 4;
            const int query1 = query0 + 8;
            const bool query0_in_bounds = query0 < Sq;
            const bool query1_in_bounds = query1 < Sq;

            #pragma unroll
            for (int kb = 0; kb < Bk; ++kb) {
                const int key0 =
                    kv_start + kb * 8 + lane4 * 2;
                const int key1 = key0 + 1;
                const bool key0_in_bounds = key0 < Skv;
                const bool key1_in_bounds = key1 < Skv;

                float s0 = S_frag[kb * 4 + 0] * scale;
                float s1 = S_frag[kb * 4 + 1] * scale;
                float s2 = S_frag[kb * 4 + 2] * scale;
                float s3 = S_frag[kb * 4 + 3] * scale;

                bool valid00 = query0_in_bounds && key0_in_bounds;
                bool valid01 = query0_in_bounds && key1_in_bounds;
                bool valid10 = query1_in_bounds && key0_in_bounds;
                bool valid11 = query1_in_bounds && key1_in_bounds;

                if constexpr (masked) {
                    if (needs_causal_mask) {
                        valid00 = valid00 && key0 <= query0;
                        valid01 = valid01 && key1 <= query0;
                        valid10 = valid10 && key0 <= query1;
                        valid11 = valid11 && key1 <= query1;
                    }
                }

                if (!valid00) s0 = -FLT_MAX;
                if (!valid01) s1 = -FLT_MAX;
                if (!valid10) s2 = -FLT_MAX;
                if (!valid11) s3 = -FLT_MAX;

                S_frag[kb * 4 + 0] = s0;
                S_frag[kb * 4 + 1] = s1;
                S_frag[kb * 4 + 2] = s2;
                S_frag[kb * 4 + 3] = s3;

                float max0 = fmaxf(s0, s1);
                float max1 = fmaxf(s2, s3);

                max0 = fmaxf(
                    max0,
                    __shfl_xor_sync(0xffffffffu, max0, 1, 4)
                );
                max0 = fmaxf(
                    max0,
                    __shfl_xor_sync(0xffffffffu, max0, 2, 4)
                );

                max1 = fmaxf(
                    max1,
                    __shfl_xor_sync(0xffffffffu, max1, 1, 4)
                );
                max1 = fmaxf(
                    max1,
                    __shfl_xor_sync(0xffffffffu, max1, 2, 4)
                );

                tile_max[0] = fmaxf(tile_max[0], max0);
                tile_max[1] = fmaxf(tile_max[1], max1);
            }
        }

        const float old_max0 = m_frag[0];
        const float old_max1 = m_frag[1];
        const float new_max0 = fmaxf(old_max0, tile_max[0]);
        const float new_max1 = fmaxf(old_max1, tile_max[1]);

        const float alpha0 = __expf(old_max0 - new_max0);
        const float alpha1 = __expf(old_max1 - new_max1);

        for (int vs = 0; vs < Dv; ++vs) {
            O_frag[vs * 4 + 0] *= alpha0;
            O_frag[vs * 4 + 1] *= alpha0;
            O_frag[vs * 4 + 2] *= alpha1;
            O_frag[vs * 4 + 3] *= alpha1;
        }

        m_frag[0] = new_max0;
        m_frag[1] = new_max1;

        float tile_sum0 = 0.0f;
        float tile_sum1 = 0.0f;


        for (int kb = 0; kb < Bk; kb += 2) {
            const float p0 =
                __expf(S_frag[kb * 4 + 0] - new_max0);
            const float p1 =
                __expf(S_frag[kb * 4 + 1] - new_max0);
            const float p2 =
                __expf(S_frag[kb * 4 + 2] - new_max1);
            const float p3 =
                __expf(S_frag[kb * 4 + 3] - new_max1);

            const float p4 =
                __expf(S_frag[(kb + 1) * 4 + 0] - new_max0);
            const float p5 =
                __expf(S_frag[(kb + 1) * 4 + 1] - new_max0);
            const float p6 =
                __expf(S_frag[(kb + 1) * 4 + 2] - new_max1);
            const float p7 =
                __expf(S_frag[(kb + 1) * 4 + 3] - new_max1);

            tile_sum0 += p0 + p1 + p4 + p5;
            tile_sum1 += p2 + p3 + p6 + p7;

            uint32_t p_frag[4];
            p_frag[0] = pack_float2_to_half2_u32(p0, p1);
            p_frag[1] = pack_float2_to_half2_u32(p2, p3);
            p_frag[2] = pack_float2_to_half2_u32(p4, p5);
            p_frag[3] = pack_float2_to_half2_u32(p6, p7);

            const int v_row =
                (kb / 2) * 16 + (lane & 15);

            for (int vs = 0; vs < Dv; ++vs) {
                uint32_t v_frag[2];
                const int v_col = vs * 8;

                ldmatrix_x2_trans(
                    v_frag,
                    smem_u32_ptr(
                        Vsmem[current_stage] +
                        v_row * V_STRIDE +
                        v_col
                    )
                );

                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%0, %1, %2, %3};\n"
                    : "+f"(O_frag[vs * 4 + 0]),
                      "+f"(O_frag[vs * 4 + 1]),
                      "+f"(O_frag[vs * 4 + 2]),
                      "+f"(O_frag[vs * 4 + 3])
                    : "r"(p_frag[0]), "r"(p_frag[1]),
                      "r"(p_frag[2]), "r"(p_frag[3]),
                      "r"(v_frag[0]), "r"(v_frag[1])
                );
            }
        }

        tile_sum0 += __shfl_xor_sync(
            0xffffffffu, tile_sum0, 1, 4
        );
        tile_sum0 += __shfl_xor_sync(
            0xffffffffu, tile_sum0, 2, 4
        );

        tile_sum1 += __shfl_xor_sync(
            0xffffffffu, tile_sum1, 1, 4
        );
        tile_sum1 += __shfl_xor_sync(
            0xffffffffu, tile_sum1, 2, 4
        );

        l_frag[0] = l_frag[0] * alpha0 + tile_sum0;
        l_frag[1] = l_frag[1] * alpha1 + tile_sum1;

        if (next_kv_tile < kv_tiles_to_process) {
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
            __syncthreads();
        }
    }

    const int row0 =
        q_block_start + warp * 16 + lane / 4;
    const int row1 = row0 + 8;
    const int col0 = lane4 * 2;
    const int col1 = col0 + 1;
    const float inv_l0 = 1.0f / l_frag[0];
    const float inv_l1 = 1.0f / l_frag[1];

    #pragma unroll
    for (int vs = 0; vs < Dv; ++vs) {
        const int output_col0 = vs * 8 + col0;
        const int output_col1 = vs * 8 + col1;

        if (row0 < Sq && output_col0 < actual_D) {
            Optr[
                static_cast<size_t>(row0) * actual_D + output_col0
            ] = __float2half(
                O_frag[vs * 4 + 0] * inv_l0
            );
        }
        if (row0 < Sq && output_col1 < actual_D) {
            Optr[
                static_cast<size_t>(row0) * actual_D + output_col1
            ] = __float2half(
                O_frag[vs * 4 + 1] * inv_l0
            );
        }

        if (row1 < Sq && output_col0 < actual_D) {
            Optr[
                static_cast<size_t>(row1) * actual_D + output_col0
            ] = __float2half(
                O_frag[vs * 4 + 2] * inv_l1
            );
        }
        if (row1 < Sq && output_col1 < actual_D) {
            Optr[
                static_cast<size_t>(row1) * actual_D + output_col1
            ] = __float2half(
                O_frag[vs * 4 + 3] * inv_l1
            );
        }
    }

    if (lane4 == 0) {
        if (row0 < Sq) {
            Lptr[row0] = m_frag[0] + logf(l_frag[0]);
        }
        if (row1 < Sq) {
            Lptr[row1] = m_frag[1] + logf(l_frag[1]);
        }
    }
}






namespace flashattn_masked_bwd_detail {
template<int M, int N, int ACC_COUNT>
__device__ __forceinline__ void store_f16(
    const float (&accumulator)[ACC_COUNT],
    __half* __restrict__ output,
    int row_offset,
    int total_rows,
    int actual_cols
) {
    constexpr int kWarps = 4;
    constexpr int kMTiles = M / 16;
    constexpr int kNTiles = N / 8;
    constexpr int kTotalTiles = kMTiles * kNTiles;
    constexpr int kTilesPerWarp =
        (kTotalTiles + kWarps - 1) / kWarps;

    static_assert(ACC_COUNT == kTilesPerWarp * 4,
                  "Wrong accumulator size");

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_group = lane >> 2;
    const int lane4 = lane & 3;

    #pragma unroll
    for (int slot = 0; slot < kTilesPerWarp; ++slot) {
        const int tile = warp + slot * kWarps;
        if (tile < kTotalTiles) {
            const int m_tile = tile / kNTiles;
            const int n_tile = tile % kNTiles;

            const int local_row0 = m_tile * 16 + row_group;
            const int local_row1 = local_row0 + 8;
            const int global_row0 = row_offset + local_row0;
            const int global_row1 = row_offset + local_row1;
            const int col0 = n_tile * 8 + lane4 * 2;
            const int col1 = col0 + 1;

            if (global_row0 < total_rows && col0 < actual_cols) {
                output[
                    static_cast<size_t>(global_row0) * actual_cols + col0
                ] = __float2half(accumulator[slot * 4 + 0]);
            }
            if (global_row0 < total_rows && col1 < actual_cols) {
                output[
                    static_cast<size_t>(global_row0) * actual_cols + col1
                ] = __float2half(accumulator[slot * 4 + 1]);
            }
            if (global_row1 < total_rows && col0 < actual_cols) {
                output[
                    static_cast<size_t>(global_row1) * actual_cols + col0
                ] = __float2half(accumulator[slot * 4 + 2]);
            }
            if (global_row1 < total_rows && col1 < actual_cols) {
                output[
                    static_cast<size_t>(global_row1) * actual_cols + col1
                ] = __float2half(accumulator[slot * 4 + 3]);
            }
        }
    }
}

}

template<int D_PAD>
__global__ void flashattn_bwd_delta_kernel(
    const __half* __restrict__ O,
    const __half* __restrict__ dO,
    float*        __restrict__ Delta,
    int actual_D,
    int total_q_rows
) {
    static_assert(D_PAD > 0 && D_PAD % 16 == 0,
                  "D_PAD must be divisible by 16");
    if (blockDim.x < 32 || (blockDim.x & 31) != 0) return;
    if (actual_D <= 0 || actual_D > D_PAD) return;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * warps_per_block + warp;

    if (row < total_q_rows) {
        const __half* o_row =
            O + static_cast<size_t>(row) * actual_D;
        const __half* do_row =
            dO + static_cast<size_t>(row) * actual_D;

        float sum = 0.0f;
        for (int col = lane; col < actual_D; col += 32) {
            sum +=
                __half2float(o_row[col]) *
                __half2float(do_row[col]);
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        }

        if (lane == 0) Delta[row] = sum;
    }
}


template<int Br, int Bc, int D_PAD, bool masked>
__global__ void flashattn_bwd_dkdv_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    const __half* __restrict__ dO,
    const float*  __restrict__ L,
    const float*  __restrict__ Delta,
          __half* __restrict__ dK,
          __half* __restrict__ dV,
          int actual_D,
          int Skv,
          int Sq
) {
    static_assert(Br > 0 && Br % 16 == 0,
                  "Br must be divisible by 16");
    static_assert(Bc > 0 && Bc % 16 == 0,
                  "Bc must be divisible by 16");
    static_assert(D_PAD > 0 && D_PAD % 16 == 0,
                  "D_PAD must be divisible by 16");

    if (blockDim.x != 128) return;
    if (actual_D <= 0 || actual_D > D_PAD || (actual_D & 7) != 0 ||
        Sq <= 0 || Skv <= 0) {
        return;
    }

    const int tid = threadIdx.x;
    const float scale = 1.0f / sqrtf(static_cast<float>(actual_D));

    constexpr int PAD = 8;
    constexpr int Q_STRIDE = D_PAD + PAD;
    constexpr int K_STRIDE = D_PAD + PAD;
    constexpr int V_STRIDE = D_PAD + PAD;
    constexpr int DO_STRIDE = D_PAD + PAD;
    constexpr int S_STRIDE = Bc + PAD;

    constexpr int kOutputTiles = (Bc / 16) * (D_PAD / 8);
    constexpr int kTilesPerWarp = (kOutputTiles + 3) / 4;
    constexpr int kAccumulatorCount = kTilesPerWarp * 4;

    float dK_fragment[kAccumulatorCount] = {0.0f};
    float dV_fragment[kAccumulatorCount] = {0.0f};

    const int batch = blockIdx.x;
    const int head = blockIdx.y;
    const int kv_tile = blockIdx.z;
    const int heads = gridDim.y;

    const long long q_base =
        (static_cast<long long>(batch) * heads + head) *
        Sq * actual_D;
    const long long kv_base =
        (static_cast<long long>(batch) * heads + head) *
        Skv * actual_D;
    const long long stat_base =
        (static_cast<long long>(batch) * heads + head) * Sq;

    const __half* Qptr = Q + q_base;
    const __half* Kptr = K + kv_base;
    const __half* Vptr = V + kv_base;
    const __half* dOptr = dO + q_base;
    const float* Lptr = L + stat_base;
    const float* Deltaptr = Delta + stat_base;
    __half* dKptr = dK + kv_base;
    __half* dVptr = dV + kv_base;

    const int q_tiles = (Sq + Br - 1) / Br;
    const int kv_tiles = (Skv + Bc - 1) / Bc;
    if (kv_tile >= kv_tiles) return;

    const int kv_start = kv_tile * Bc;
    const int kv_last_unclamped = kv_start + Bc - 1;
    const int kv_real_last = kv_last_unclamped < Skv
        ? kv_last_unclamped
        : Skv - 1;
    const bool kv_tile_full = kv_last_unclamped < Skv;

    int first_q_tile = 0;
    if constexpr (masked) {
        if (kv_start >= Sq) {
            flashattn_masked_bwd_detail::store_f16<
                Bc, D_PAD, kAccumulatorCount
            >(dK_fragment, dKptr, kv_start, Skv, actual_D);
            flashattn_masked_bwd_detail::store_f16<
                Bc, D_PAD, kAccumulatorCount
            >(dV_fragment, dVptr, kv_start, Skv, actual_D);
            return;
        }
        first_q_tile = kv_start / Br;
    }

    extern __shared__ char shared_raw[];
    char* shared_ptr = shared_raw;

    auto align_ptr = [&](size_t alignment = 16) {
        const uintptr_t value = reinterpret_cast<uintptr_t>(shared_ptr);
        shared_ptr = reinterpret_cast<char*>(
            (value + alignment - 1) & ~(alignment - 1)
        );
    };

    align_ptr();
    __half* smemQ0 = reinterpret_cast<__half*>(shared_ptr);
    shared_ptr += Br * Q_STRIDE * sizeof(__half);
    align_ptr();
    __half* smemQ1 = reinterpret_cast<__half*>(shared_ptr);
    shared_ptr += Br * Q_STRIDE * sizeof(__half);
    __half* smemQ[2] = {smemQ0, smemQ1};

    align_ptr();
    __half* smemdO0 = reinterpret_cast<__half*>(shared_ptr);
    shared_ptr += Br * DO_STRIDE * sizeof(__half);
    align_ptr();
    __half* smemdO1 = reinterpret_cast<__half*>(shared_ptr);
    shared_ptr += Br * DO_STRIDE * sizeof(__half);
    __half* smemdO[2] = {smemdO0, smemdO1};

    align_ptr();
    __half* smemK = reinterpret_cast<__half*>(shared_ptr);
    shared_ptr += Bc * K_STRIDE * sizeof(__half);
    align_ptr();
    __half* smemV = reinterpret_cast<__half*>(shared_ptr);
    shared_ptr += Bc * V_STRIDE * sizeof(__half);

    align_ptr();
    float* score_smem = reinterpret_cast<float*>(shared_ptr);
    shared_ptr += Br * S_STRIDE * sizeof(float);
    align_ptr();
    __half* p_ds_smem = reinterpret_cast<__half*>(shared_ptr);
    shared_ptr += Bc * Br * sizeof(__half);

    align_ptr();
    float* l_smem = reinterpret_cast<float*>(shared_ptr);
    shared_ptr += Br * sizeof(float);
    align_ptr();
    float* delta_smem = reinterpret_cast<float*>(shared_ptr);

    asyncLOAD_2D_TILE<
        Bc, D_PAD, 128
    >(
        Kptr,
        smem_u32_ptr(smemK),
        tid, K_STRIDE, Skv, actual_D, actual_D, kv_tile, 0
    );
    asyncLOAD_2D_TILE<
        Bc, D_PAD, 128
    >(
        Vptr,
        smem_u32_ptr(smemV),
        tid, V_STRIDE, Skv, actual_D, actual_D, kv_tile, 0
    );
    asm volatile("cp.async.commit_group;\n");

    asyncLOAD_2D_TILE<
        Br, D_PAD, 128
    >(
        Qptr,
        smem_u32_ptr(smemQ[0]),
        tid, Q_STRIDE, Sq, actual_D, actual_D, first_q_tile, 0
    );
    asyncLOAD_2D_TILE<
        Br, D_PAD, 128
    >(
        dOptr,
        smem_u32_ptr(smemdO[0]),
        tid, DO_STRIDE, Sq, actual_D, actual_D, first_q_tile, 0
    );
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    const int q_steps = q_tiles - first_q_tile;
    for (int step = 0; step < q_steps; ++step) {
        const int q_tile = first_q_tile + step;
        const int current_stage = step & 1;
        const int next_stage = current_stage ^ 1;
        const int next_q_tile = q_tile + 1;

        if (next_q_tile < q_tiles) {
            asyncLOAD_2D_TILE<
                Br, D_PAD, 128
            >(
                Qptr,
                smem_u32_ptr(
                    smemQ[next_stage]
                ),
                tid, Q_STRIDE, Sq, actual_D, actual_D,
                next_q_tile, 0
            );
            asyncLOAD_2D_TILE<
                Br, D_PAD, 128
            >(
                dOptr,
                smem_u32_ptr(
                    smemdO[next_stage]
                ),
                tid, DO_STRIDE, Sq, actual_D, actual_D,
                next_q_tile, 0
            );
            asm volatile("cp.async.commit_group;\n");
        }

        const int q_start = q_tile * Br;
        const int q_last_unclamped = q_start + Br - 1;
        const bool q_tile_full = q_last_unclamped < Sq;
        bool needs_causal_mask = false;
        if constexpr (masked) {
            needs_causal_mask = !(kv_real_last <= q_start);
        }
        const bool needs_any_mask =
            !q_tile_full || !kv_tile_full || needs_causal_mask;

        for (int row = tid; row < Br; row += blockDim.x) {
            const int global_q = q_start + row;
            if (global_q < Sq) {
                l_smem[row] = Lptr[global_q];
                delta_smem[row] = Deltaptr[global_q];
            } else {
                l_smem[row] = 0.0f;
                delta_smem[row] = 0.0f;
            }
        }

        mma_score_f16_tiled<
            Br, D_PAD, Bc, Q_STRIDE, K_STRIDE, S_STRIDE
        >(smemQ[current_stage], smemK, score_smem);
        __syncthreads();

        if (!needs_any_mask) {
            #pragma unroll
            for (int index = tid; index < Br * Bc; index += 128) {
                const int row = index / Bc;
                const int col = index % Bc;
                const float p = __expf(
                    score_smem[row * S_STRIDE + col] * scale -
                    l_smem[row]
                );
                p_ds_smem[col * Br + row] = __float2half(p);
            }
        } else {
            #pragma unroll
            for (int index = tid; index < Br * Bc; index += 128) {
                const int row = index / Bc;
                const int col = index % Bc;
                const int global_q = q_start + row;
                const int global_k = kv_start + col;

                bool valid = global_q < Sq && global_k < Skv;
                if constexpr (masked) {
                    if (needs_causal_mask) {
                        valid = valid && global_k <= global_q;
                    }
                }

                float p = 0.0f;
                if (valid) {
                    p = __expf(
                        score_smem[row * S_STRIDE + col] * scale -
                        l_smem[row]
                    );
                }
                p_ds_smem[col * Br + row] = __float2half(p);
            }
        }
        __syncthreads();


        mma_accum_f16_registers<
            Bc, Br, D_PAD, Br, DO_STRIDE, kAccumulatorCount
        >(p_ds_smem, smemdO[current_stage], dV_fragment);


        mma_score_f16_tiled<
            Br, D_PAD, Bc, DO_STRIDE, V_STRIDE, S_STRIDE
        >(smemdO[current_stage], smemV, score_smem);
        __syncthreads();

        if (!needs_any_mask) {
            #pragma unroll
            for (int index = tid; index < Br * Bc; index += 128) {
                const int row = index / Bc;
                const int col = index % Bc;
                const float p =
                    __half2float(p_ds_smem[col * Br + row]);
                const float dp = score_smem[row * S_STRIDE + col];
                const float ds =
                    p * (dp - delta_smem[row]) * scale;
                p_ds_smem[col * Br + row] = __float2half(ds);
            }
        } else {
            #pragma unroll
            for (int index = tid; index < Br * Bc; index += 128) {
                const int row = index / Bc;
                const int col = index % Bc;
                const int global_q = q_start + row;
                const int global_k = kv_start + col;

                bool valid = global_q < Sq && global_k < Skv;
                if constexpr (masked) {
                    if (needs_causal_mask) {
                        valid = valid && global_k <= global_q;
                    }
                }

                float ds = 0.0f;
                if (valid) {
                    const float p =
                        __half2float(p_ds_smem[col * Br + row]);
                    const float dp = score_smem[row * S_STRIDE + col];
                    ds = p * (dp - delta_smem[row]) * scale;
                }
                p_ds_smem[col * Br + row] = __float2half(ds);
            }
        }
        __syncthreads();


        mma_accum_f16_registers<
            Bc, Br, D_PAD, Br, Q_STRIDE, kAccumulatorCount
        >(p_ds_smem, smemQ[current_stage], dK_fragment);

        if (next_q_tile < q_tiles) {
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
            __syncthreads();
        }
    }

    flashattn_masked_bwd_detail::store_f16<
        Bc, D_PAD, kAccumulatorCount
    >(dK_fragment, dKptr, kv_start, Skv, actual_D);
    flashattn_masked_bwd_detail::store_f16<
        Bc, D_PAD, kAccumulatorCount
    >(dV_fragment, dVptr, kv_start, Skv, actual_D);
}


template<int Br, int Bc, int D_PAD, bool masked>
__global__ void flashattn_bwd_dq_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    const __half* __restrict__ dO,
    const float*  __restrict__ L,
    const float*  __restrict__ Delta,
          __half* __restrict__ dQ,
          int actual_D,
          int Skv,
          int Sq
) {
    static_assert(Br > 0 && Br % 16 == 0,
                  "Br must be divisible by 16");
    static_assert(Bc > 0 && Bc % 16 == 0,
                  "Bc must be divisible by 16");
    static_assert(D_PAD > 0 && D_PAD % 16 == 0,
                  "D_PAD must be divisible by 16");

    if (blockDim.x != 128) return;
    if (actual_D <= 0 || actual_D > D_PAD || (actual_D & 7) != 0 ||
        Sq <= 0 || Skv <= 0) {
        return;
    }

    const int tid = threadIdx.x;
    const float scale = 1.0f / sqrtf(static_cast<float>(actual_D));

    constexpr int PAD = 8;
    constexpr int Q_STRIDE = D_PAD + PAD;
    constexpr int K_STRIDE = D_PAD + PAD;
    constexpr int V_STRIDE = D_PAD + PAD;
    constexpr int DO_STRIDE = D_PAD + PAD;
    constexpr int S_STRIDE = Bc + PAD;

    constexpr int kOutputTiles = (Br / 16) * (D_PAD / 8);
    constexpr int kTilesPerWarp = (kOutputTiles + 3) / 4;
    constexpr int kAccumulatorCount = kTilesPerWarp * 4;
    float dQ_fragment[kAccumulatorCount] = {0.0f};

    const int batch = blockIdx.x;
    const int head = blockIdx.y;
    const int q_tile = blockIdx.z;
    const int heads = gridDim.y;

    const long long q_base =
        (static_cast<long long>(batch) * heads + head) *
        Sq * actual_D;
    const long long kv_base =
        (static_cast<long long>(batch) * heads + head) *
        Skv * actual_D;
    const long long stat_base =
        (static_cast<long long>(batch) * heads + head) * Sq;

    const __half* Qptr = Q + q_base;
    const __half* Kptr = K + kv_base;
    const __half* Vptr = V + kv_base;
    const __half* dOptr = dO + q_base;
    const float* Lptr = L + stat_base;
    const float* Deltaptr = Delta + stat_base;
    __half* dQptr = dQ + q_base;

    const int q_tiles = (Sq + Br - 1) / Br;
    const int kv_tiles = (Skv + Bc - 1) / Bc;
    if (q_tile >= q_tiles) return;

    const int q_start = q_tile * Br;
    const int q_last_unclamped = q_start + Br - 1;
    const int q_real_last = q_last_unclamped < Sq
        ? q_last_unclamped
        : Sq - 1;
    const bool q_tile_full = q_last_unclamped < Sq;

    int kv_tiles_to_process = kv_tiles;
    if constexpr (masked) {
        const int causal_tiles = q_real_last / Bc + 1;
        if (causal_tiles < kv_tiles_to_process) {
            kv_tiles_to_process = causal_tiles;
        }
    }

    extern __shared__ char shared_raw[];
    char* shared_ptr = shared_raw;

    auto align_ptr = [&](size_t alignment = 16) {
        const uintptr_t value = reinterpret_cast<uintptr_t>(shared_ptr);
        shared_ptr = reinterpret_cast<char*>(
            (value + alignment - 1) & ~(alignment - 1)
        );
    };

    align_ptr();
    __half* smemQ = reinterpret_cast<__half*>(shared_ptr);
    shared_ptr += Br * Q_STRIDE * sizeof(__half);
    align_ptr();
    __half* smemdO = reinterpret_cast<__half*>(shared_ptr);
    shared_ptr += Br * DO_STRIDE * sizeof(__half);

    align_ptr();
    __half* smemK0 = reinterpret_cast<__half*>(shared_ptr);
    shared_ptr += Bc * K_STRIDE * sizeof(__half);
    align_ptr();
    __half* smemK1 = reinterpret_cast<__half*>(shared_ptr);
    shared_ptr += Bc * K_STRIDE * sizeof(__half);
    __half* smemK[2] = {smemK0, smemK1};

    align_ptr();
    __half* smemV0 = reinterpret_cast<__half*>(shared_ptr);
    shared_ptr += Bc * V_STRIDE * sizeof(__half);
    align_ptr();
    __half* smemV1 = reinterpret_cast<__half*>(shared_ptr);
    shared_ptr += Bc * V_STRIDE * sizeof(__half);
    __half* smemV[2] = {smemV0, smemV1};

    align_ptr();
    float* score_smem = reinterpret_cast<float*>(shared_ptr);
    shared_ptr += Br * S_STRIDE * sizeof(float);
    align_ptr();
    __half* p_ds_smem = reinterpret_cast<__half*>(shared_ptr);
    shared_ptr += Br * Bc * sizeof(__half);

    align_ptr();
    float* l_smem = reinterpret_cast<float*>(shared_ptr);
    shared_ptr += Br * sizeof(float);
    align_ptr();
    float* delta_smem = reinterpret_cast<float*>(shared_ptr);

    asyncLOAD_2D_TILE<
        Br, D_PAD, 128
    >(
        Qptr,
        smem_u32_ptr(smemQ),
        tid, Q_STRIDE, Sq, actual_D, actual_D, q_tile, 0
    );
    asyncLOAD_2D_TILE<
        Br, D_PAD, 128
    >(
        dOptr,
        smem_u32_ptr(smemdO),
        tid, DO_STRIDE, Sq, actual_D, actual_D, q_tile, 0
    );
    asm volatile("cp.async.commit_group;\n");

    for (int row = tid; row < Br; row += blockDim.x) {
        const int global_q = q_start + row;
        if (global_q < Sq) {
            l_smem[row] = Lptr[global_q];
            delta_smem[row] = Deltaptr[global_q];
        } else {
            l_smem[row] = 0.0f;
            delta_smem[row] = 0.0f;
        }
    }

    asyncLOAD_2D_TILE<
        Bc, D_PAD, 128
    >(
        Kptr,
        smem_u32_ptr(smemK[0]),
        tid, K_STRIDE, Skv, actual_D, actual_D, 0, 0
    );
    asyncLOAD_2D_TILE<
        Bc, D_PAD, 128
    >(
        Vptr,
        smem_u32_ptr(smemV[0]),
        tid, V_STRIDE, Skv, actual_D, actual_D, 0, 0
    );
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    for (int kv_tile = 0; kv_tile < kv_tiles_to_process; ++kv_tile) {
        const int current_stage = kv_tile & 1;
        const int next_stage = current_stage ^ 1;
        const int next_kv_tile = kv_tile + 1;

        if (next_kv_tile < kv_tiles_to_process) {
            asyncLOAD_2D_TILE<
                Bc, D_PAD, 128
            >(
                Kptr,
                smem_u32_ptr(
                    smemK[next_stage]
                ),
                tid, K_STRIDE, Skv, actual_D, actual_D,
                next_kv_tile, 0
            );
            asyncLOAD_2D_TILE<
                Bc, D_PAD, 128
            >(
                Vptr,
                smem_u32_ptr(
                    smemV[next_stage]
                ),
                tid, V_STRIDE, Skv, actual_D, actual_D,
                next_kv_tile, 0
            );
            asm volatile("cp.async.commit_group;\n");
        }

        const int kv_start = kv_tile * Bc;
        const int kv_last_unclamped = kv_start + Bc - 1;
        const int kv_real_last = kv_last_unclamped < Skv
            ? kv_last_unclamped
            : Skv - 1;
        const bool kv_tile_full = kv_last_unclamped < Skv;

        bool needs_causal_mask = false;
        if constexpr (masked) {
            needs_causal_mask = !(kv_real_last <= q_start);
        }
        const bool needs_any_mask =
            !q_tile_full || !kv_tile_full || needs_causal_mask;

        mma_score_f16_tiled<
            Br, D_PAD, Bc, Q_STRIDE, K_STRIDE, S_STRIDE
        >(smemQ, smemK[current_stage], score_smem);
        __syncthreads();

        if (!needs_any_mask) {
            #pragma unroll
            for (int index = tid; index < Br * Bc; index += 128) {
                const int row = index / Bc;
                const int col = index % Bc;
                const float p = __expf(
                    score_smem[row * S_STRIDE + col] * scale -
                    l_smem[row]
                );
                p_ds_smem[row * Bc + col] = __float2half(p);
            }
        } else {
            #pragma unroll
            for (int index = tid; index < Br * Bc; index += 128) {
                const int row = index / Bc;
                const int col = index % Bc;
                const int global_q = q_start + row;
                const int global_k = kv_start + col;

                bool valid = global_q < Sq && global_k < Skv;
                if constexpr (masked) {
                    if (needs_causal_mask) {
                        valid = valid && global_k <= global_q;
                    }
                }

                float p = 0.0f;
                if (valid) {
                    p = __expf(
                        score_smem[row * S_STRIDE + col] * scale -
                        l_smem[row]
                    );
                }
                p_ds_smem[row * Bc + col] = __float2half(p);
            }
        }
        __syncthreads();


        mma_score_f16_tiled<
            Br, D_PAD, Bc, DO_STRIDE, V_STRIDE, S_STRIDE
        >(smemdO, smemV[current_stage], score_smem);
        __syncthreads();

        if (!needs_any_mask) {
            #pragma unroll
            for (int index = tid; index < Br * Bc; index += 128) {
                const int row = index / Bc;
                const int col = index % Bc;
                const float p =
                    __half2float(p_ds_smem[row * Bc + col]);
                const float dp = score_smem[row * S_STRIDE + col];
                const float ds =
                    p * (dp - delta_smem[row]) * scale;
                p_ds_smem[row * Bc + col] = __float2half(ds);
            }
        } else {
            #pragma unroll
            for (int index = tid; index < Br * Bc; index += 128) {
                const int row = index / Bc;
                const int col = index % Bc;
                const int global_q = q_start + row;
                const int global_k = kv_start + col;

                bool valid = global_q < Sq && global_k < Skv;
                if constexpr (masked) {
                    if (needs_causal_mask) {
                        valid = valid && global_k <= global_q;
                    }
                }

                float ds = 0.0f;
                if (valid) {
                    const float p =
                        __half2float(p_ds_smem[row * Bc + col]);
                    const float dp = score_smem[row * S_STRIDE + col];
                    ds = p * (dp - delta_smem[row]) * scale;
                }
                p_ds_smem[row * Bc + col] = __float2half(ds);
            }
        }
        __syncthreads();


        mma_accum_f16_registers<
            Br, Bc, D_PAD, Bc, K_STRIDE, kAccumulatorCount
        >(p_ds_smem, smemK[current_stage], dQ_fragment);

        if (next_kv_tile < kv_tiles_to_process) {
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
            __syncthreads();
        }
    }

    flashattn_masked_bwd_detail::store_f16<
        Br, D_PAD, kAccumulatorCount
    >(dQ_fragment, dQptr, q_start, Sq, actual_D);
}

template<int Br, int Bc, int D_PAD>
constexpr size_t flashattn_bwd_dkdv_smem_bytes() {
    constexpr int PAD = 8;
    constexpr int D_STRIDE = D_PAD + PAD;
    constexpr int S_STRIDE = Bc + PAD;

    return
        2 * Br * D_STRIDE * sizeof(__half) +
        2 * Br * D_STRIDE * sizeof(__half) +
        Bc * D_STRIDE * sizeof(__half) +
        Bc * D_STRIDE * sizeof(__half) +
        Br * S_STRIDE * sizeof(float) +
        Bc * Br * sizeof(__half) +
        2 * Br * sizeof(float) +
        256;
}

template<int Br, int Bc, int D_PAD>
constexpr size_t flashattn_bwd_dq_smem_bytes() {
    constexpr int PAD = 8;
    constexpr int D_STRIDE = D_PAD + PAD;
    constexpr int S_STRIDE = Bc + PAD;

    return
        Br * D_STRIDE * sizeof(__half) +
        Br * D_STRIDE * sizeof(__half) +
        2 * Bc * D_STRIDE * sizeof(__half) +
        2 * Bc * D_STRIDE * sizeof(__half) +
        Br * S_STRIDE * sizeof(float) +
        Br * Bc * sizeof(__half) +
        2 * Br * sizeof(float) +
        256;
}


#endif
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <limits>
#include <vector>

static int select_d_pad(int d) {
    if (d <= 32) return 32;
    if (d <= 64) return 64;
    if (d <= 80) return 80;
    if (d <= 96) return 96;
    if (d <= 128) return 128;
    if (d <= 160) return 160;
    if (d <= 192) return 192;
    if (d <= 224) return 224;
    if (d <= 256) return 256;
    TORCH_CHECK(false, "head dimension must be <= 256");
    return 0;
}

static void check_qkv(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA");
    TORCH_CHECK(K.is_cuda(), "K must be CUDA");
    TORCH_CHECK(V.is_cuda(), "V must be CUDA");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat16, "Q must be float16");
    TORCH_CHECK(K.scalar_type() == torch::kFloat16, "K must be float16");
    TORCH_CHECK(V.scalar_type() == torch::kFloat16, "V must be float16");
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");
    TORCH_CHECK(Q.dim() == 4, "Q must have shape [B,H,Sq,D]");
    TORCH_CHECK(K.dim() == 4, "K must have shape [B,H,Skv,D]");
    TORCH_CHECK(V.dim() == 4, "V must have shape [B,H,Skv,D]");
    TORCH_CHECK(Q.device() == K.device(), "Q and K must be on the same CUDA device");
    TORCH_CHECK(Q.device() == V.device(), "Q and V must be on the same CUDA device");
    TORCH_CHECK(Q.size(0) == K.size(0), "Q and K batch dimensions must match");
    TORCH_CHECK(Q.size(0) == V.size(0), "Q and V batch dimensions must match");
    TORCH_CHECK(Q.size(1) == K.size(1), "Q and K head counts must match");
    TORCH_CHECK(Q.size(1) == V.size(1), "Q and V head counts must match");
    TORCH_CHECK(K.size(2) == V.size(2), "K and V sequence lengths must match");
    TORCH_CHECK(Q.size(3) == K.size(3), "Q and K head dimensions must match");
    TORCH_CHECK(Q.size(3) == V.size(3), "Q and V head dimensions must match");
    TORCH_CHECK(Q.size(0) > 0, "B must be positive");
    TORCH_CHECK(Q.size(1) > 0, "H must be positive");
    TORCH_CHECK(Q.size(2) > 0, "Sq must be positive");
    TORCH_CHECK(K.size(2) > 0, "Skv must be positive");
    TORCH_CHECK(Q.size(3) > 0, "D must be positive");
    TORCH_CHECK((Q.size(3) & 7) == 0, "D must be divisible by 8");
    TORCH_CHECK(Q.size(3) <= 256, "D must be <= 256");
    TORCH_CHECK(Q.size(0) <= std::numeric_limits<int>::max(), "B is too large");
    TORCH_CHECK(Q.size(1) <= 65535, "H is too large");
    TORCH_CHECK(Q.size(2) <= std::numeric_limits<int>::max(), "Sq is too large");
    TORCH_CHECK(K.size(2) <= std::numeric_limits<int>::max(), "Skv is too large");
    TORCH_CHECK((Q.size(2) + 63) / 64 <= 65535, "Sq requires too many forward tiles");
}

static void check_backward_inputs(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    const torch::Tensor& O,
    const torch::Tensor& dO,
    const torch::Tensor& L
) {
    check_qkv(Q, K, V);
    TORCH_CHECK(O.is_cuda(), "O must be CUDA");
    TORCH_CHECK(dO.is_cuda(), "dO must be CUDA");
    TORCH_CHECK(L.is_cuda(), "L must be CUDA");
    TORCH_CHECK(O.device() == Q.device(), "O must be on the same CUDA device as Q");
    TORCH_CHECK(dO.device() == Q.device(), "dO must be on the same CUDA device as Q");
    TORCH_CHECK(L.device() == Q.device(), "L must be on the same CUDA device as Q");
    TORCH_CHECK(O.scalar_type() == torch::kFloat16, "O must be float16");
    TORCH_CHECK(dO.scalar_type() == torch::kFloat16, "dO must be float16");
    TORCH_CHECK(L.scalar_type() == torch::kFloat32, "L must be float32");
    TORCH_CHECK(O.is_contiguous(), "O must be contiguous");
    TORCH_CHECK(dO.is_contiguous(), "dO must be contiguous");
    TORCH_CHECK(L.is_contiguous(), "L must be contiguous");
    TORCH_CHECK(O.sizes() == Q.sizes(), "O shape must match Q");
    TORCH_CHECK(dO.sizes() == Q.sizes(), "dO shape must match Q");
    TORCH_CHECK(L.dim() == 3, "L must have shape [B,H,Sq]");
    TORCH_CHECK(L.size(0) == Q.size(0), "L batch dimension must match Q");
    TORCH_CHECK(L.size(1) == Q.size(1), "L head count must match Q");
    TORCH_CHECK(L.size(2) == Q.size(2), "L sequence length must match Q");
}

template<int D_PAD, int Bc, bool Masked>
static std::vector<torch::Tensor> launch_fwd_impl(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V
) {
    constexpr int Br = 64;
    constexpr int PAD = 8;
    constexpr int D_STRIDE = D_PAD + PAD;
    constexpr size_t smem_bytes =
        (Br * D_STRIDE + 4 * Bc * D_STRIDE) * sizeof(__half) + 256;

    const int B = static_cast<int>(Q.size(0));
    const int H = static_cast<int>(Q.size(1));
    const int Sq = static_cast<int>(Q.size(2));
    const int Skv = static_cast<int>(K.size(2));
    const int actual_D = static_cast<int>(Q.size(3));

    auto O = torch::empty_like(Q);
    auto L = torch::empty(
        {Q.size(0), Q.size(1), Q.size(2)},
        Q.options().dtype(torch::kFloat32)
    );

    dim3 block(128);
    dim3 grid(B, H, (Sq + Br - 1) / Br);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(Q.get_device());

    C10_CUDA_CHECK(cudaFuncSetAttribute(
        flashattn_fwd<Br, Bc, D_PAD, Masked>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)
    ));

    flashattn_fwd<Br, Bc, D_PAD, Masked>
        <<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(K.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(V.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(O.data_ptr<at::Half>()),
            L.data_ptr<float>(),
            actual_D,
            Skv,
            Sq
        );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {O, L};
}

template<bool Masked>
static std::vector<torch::Tensor> dispatch_fwd(
    int d_pad,
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V
) {
    switch (d_pad) {
        case 32: return launch_fwd_impl<32, 64, Masked>(Q, K, V);
        case 64: return launch_fwd_impl<64, 64, Masked>(Q, K, V);
        case 80: return launch_fwd_impl<80, 64, Masked>(Q, K, V);
        case 96: return launch_fwd_impl<96, 64, Masked>(Q, K, V);
        case 128: return launch_fwd_impl<128, 64, Masked>(Q, K, V);
        case 160: return launch_fwd_impl<160, 32, Masked>(Q, K, V);
        case 192: return launch_fwd_impl<192, 32, Masked>(Q, K, V);
        case 224: return launch_fwd_impl<224, 32, Masked>(Q, K, V);
        case 256: return launch_fwd_impl<256, 16, Masked>(Q, K, V);
    }
    TORCH_CHECK(false, "unsupported D_PAD");
    return {};
}

template<int Br, int Bc, int D_PAD, bool Masked>
static std::vector<torch::Tensor> launch_bwd_impl(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    const torch::Tensor& O,
    const torch::Tensor& dO,
    const torch::Tensor& L
) {
    const int B = static_cast<int>(Q.size(0));
    const int H = static_cast<int>(Q.size(1));
    const int Sq = static_cast<int>(Q.size(2));
    const int Skv = static_cast<int>(K.size(2));
    const int actual_D = static_cast<int>(Q.size(3));

    const int64_t total_rows_64 =
        static_cast<int64_t>(B) * H * Sq;
    TORCH_CHECK(
        total_rows_64 <= std::numeric_limits<int>::max(),
        "B*H*Sq is too large"
    );
    const int total_rows = static_cast<int>(total_rows_64);

    TORCH_CHECK((Skv + Bc - 1) / Bc <= 65535, "Skv requires too many backward tiles");
    TORCH_CHECK((Sq + Br - 1) / Br <= 65535, "Sq requires too many backward tiles");

    auto dQ = torch::empty_like(Q);
    auto dK = torch::empty_like(K);
    auto dV = torch::empty_like(V);
    auto Delta = torch::empty(
        {Q.size(0), Q.size(1), Q.size(2)},
        Q.options().dtype(torch::kFloat32)
    );

    dim3 block(128);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(Q.get_device());

    const int rows_per_delta_block = 4;
    const int delta_blocks =
        (total_rows + rows_per_delta_block - 1) /
        rows_per_delta_block;

    flashattn_bwd_delta_kernel<D_PAD>
        <<<delta_blocks, block, 0, stream>>>(
            reinterpret_cast<const __half*>(O.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(dO.data_ptr<at::Half>()),
            Delta.data_ptr<float>(),
            actual_D,
            total_rows
        );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    constexpr size_t dkdv_smem =
        flashattn_bwd_dkdv_smem_bytes<Br, Bc, D_PAD>();

    C10_CUDA_CHECK(cudaFuncSetAttribute(
        flashattn_bwd_dkdv_kernel<Br, Bc, D_PAD, Masked>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(dkdv_smem)
    ));

    flashattn_bwd_dkdv_kernel<Br, Bc, D_PAD, Masked>
        <<<dim3(B, H, (Skv + Bc - 1) / Bc),
           block, dkdv_smem, stream>>>(
            reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(K.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(V.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(dO.data_ptr<at::Half>()),
            L.data_ptr<float>(),
            Delta.data_ptr<float>(),
            reinterpret_cast<__half*>(dK.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(dV.data_ptr<at::Half>()),
            actual_D,
            Skv,
            Sq
        );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    constexpr size_t dq_smem =
        flashattn_bwd_dq_smem_bytes<Br, Bc, D_PAD>();

    C10_CUDA_CHECK(cudaFuncSetAttribute(
        flashattn_bwd_dq_kernel<Br, Bc, D_PAD, Masked>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(dq_smem)
    ));

    flashattn_bwd_dq_kernel<Br, Bc, D_PAD, Masked>
        <<<dim3(B, H, (Sq + Br - 1) / Br),
           block, dq_smem, stream>>>(
            reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(K.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(V.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(dO.data_ptr<at::Half>()),
            L.data_ptr<float>(),
            Delta.data_ptr<float>(),
            reinterpret_cast<__half*>(dQ.data_ptr<at::Half>()),
            actual_D,
            Skv,
            Sq
        );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dQ, dK, dV};
}

template<bool Masked>
static std::vector<torch::Tensor> dispatch_bwd(
    int d_pad,
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    const torch::Tensor& O,
    const torch::Tensor& dO,
    const torch::Tensor& L
) {
    switch (d_pad) {
        case 32: return launch_bwd_impl<32, 32, 32, Masked>(Q, K, V, O, dO, L);
        case 64: return launch_bwd_impl<32, 32, 64, Masked>(Q, K, V, O, dO, L);
        case 80: return launch_bwd_impl<32, 32, 80, Masked>(Q, K, V, O, dO, L);
        case 96: return launch_bwd_impl<32, 32, 96, Masked>(Q, K, V, O, dO, L);
        case 128: return launch_bwd_impl<16, 32, 128, Masked>(Q, K, V, O, dO, L);
        case 160: return launch_bwd_impl<16, 32, 160, Masked>(Q, K, V, O, dO, L);
        case 192: return launch_bwd_impl<16, 16, 192, Masked>(Q, K, V, O, dO, L);
        case 224: return launch_bwd_impl<16, 16, 224, Masked>(Q, K, V, O, dO, L);
        case 256: return launch_bwd_impl<16, 16, 256, Masked>(Q, K, V, O, dO, L);
    }
    TORCH_CHECK(false, "unsupported D_PAD");
    return {};
}

std::vector<torch::Tensor> flash_fwd_cuda(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    bool causal
) {
    check_qkv(Q, K, V);
    c10::cuda::CUDAGuard device_guard(Q.device());
    const int d_pad = select_d_pad(static_cast<int>(Q.size(3)));
    if (causal) return dispatch_fwd<true>(d_pad, Q, K, V);
    return dispatch_fwd<false>(d_pad, Q, K, V);
}

std::vector<torch::Tensor> flash_bwd_cuda(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    const torch::Tensor& O,
    const torch::Tensor& dO,
    const torch::Tensor& L,
    bool causal
) {
    check_backward_inputs(Q, K, V, O, dO, L);
    c10::cuda::CUDAGuard device_guard(Q.device());
    const int d_pad = select_d_pad(static_cast<int>(Q.size(3)));
    if (causal) return dispatch_bwd<true>(d_pad, Q, K, V, O, dO, L);
    return dispatch_bwd<false>(d_pad, Q, K, V, O, dO, L);
}