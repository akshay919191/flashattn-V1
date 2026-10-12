#pragma once

#include "mma_helpers.cuh"

// Guarded tile load: rows >= rows_total are zero-filled (cp.async src-size = 0)
// instead of being read, so ragged Sq / Skv never touch memory past the tensor.
template<int ROWS, int D_PAD, int STRIDE>
__device__ __forceinline__ void load_tile_guarded(
    const __half* __restrict__ gbase, uint32_t smem_tile, int tid, int tile, int rows_total)
{
    constexpr int CHUNKS_PER_ROW = D_PAD / 8;              // 16-byte chunks per row
    constexpr int TOTAL          = ROWS * CHUNKS_PER_ROW;
    constexpr int ITERS          = (TOTAL + 127) / 128;
    #pragma unroll
    for (int it = 0; it < ITERS; ++it) {
        const int i = tid + it * 128;
        if (i < TOTAL) {
            const int r    = i / CHUNKS_PER_ROW;
            const int c    = (i % CHUNKS_PER_ROW) * 8;
            const int grow = tile * ROWS + r;
            const bool ok  = grow < rows_total;
            const __half* src = gbase + static_cast<size_t>(ok ? grow : 0) * D_PAD + c;
            const uint32_t dst =
                smem_tile + static_cast<uint32_t>((r * STRIDE + c) * sizeof(__half));
            const int src_bytes = ok ? 16 : 0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                         :: "r"(dst), "l"(src), "r"(src_bytes) : "memory");
        }
    }
}

template<int ROWS, int D_PAD, int STRIDE, bool FULL_TILES>
__device__ __forceinline__ void load_tile_dispatch(
    const __half* gbase, uint32_t smem_tile, int tid, int tile, int rows_total)
{
    if constexpr (FULL_TILES) {
        load_tile_full<ROWS, D_PAD, STRIDE>(gbase, smem_tile, tid, tile);
    } else {
        load_tile_guarded<ROWS, D_PAD, STRIDE>(gbase, smem_tile, tid, tile, rows_total);
    }
}

// Dynamic smem for flashattn_fwd. Q is staged in the same region that later holds
// K|V (Q -> registers, sync, then K/V overwrite it), so the region is
// max(Br, 2*Bc) rows. The launcher must allocate exactly this much.
template<int Br, int Bc, int D_PAD>
__host__ __device__ constexpr size_t flashattn_fwd_smem_bytes() {
    return static_cast<size_t>((Br > 2 * Bc) ? Br : 2 * Bc) * (D_PAD + 8) * sizeof(__half);
}

template<int Br, int Bc, int D_PAD, bool masked, bool FULL_TILES>
__global__ void __launch_bounds__(128, 2)
flashattn_fwd(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
          __half* __restrict__ output,
          float*  __restrict__ Logsum,
    const int actual_D,
    const int Skv,
    const int Sq,
    const int numKVheads
) {
    static_assert(Br == 64, "This kernel requires Br == 64");
    static_assert(Bc > 0 && Bc % 16 == 0, "Bc must be positive and divisible by 16");
    static_assert(D_PAD > 0 && D_PAD % 16 == 0, "D_PAD must be positive and divisible by 16");

    if (blockDim.x != 128) return;
    if (actual_D != D_PAD || Sq <= 0 || Skv <= 0) return;
    if constexpr (FULL_TILES) {
        if ((Sq % Br) != 0 || (Skv % Bc) != 0) return;   // caller promised full tiles
    }

    constexpr float kLog2e = 1.4426950408889634f;
    constexpr float kLn2   = 0.6931471805599453f;

    const int tid   = threadIdx.x;
    const int warp  = tid >> 5;
    const int lane  = tid & 31;
    const int lane4 = lane & 3;
    const int lane16 = lane & 15;

    const int tileid    = blockIdx.x;
    const int headid    = blockIdx.y;
    const int batchid   = blockIdx.z;
    const int num_heads = gridDim.y;

    if (numKVheads <= 0 || num_heads % numKVheads != 0) return;
    const int kv_headid = headid / (num_heads / numKVheads);

    const long long q_base =
        (static_cast<long long>(batchid) * num_heads + headid) * Sq * D_PAD;
    const long long kv_base =
        (static_cast<long long>(batchid) * numKVheads + kv_headid) * Skv * D_PAD;
    const long long stat_base =
        (static_cast<long long>(batchid) * num_heads + headid) * Sq;

    const __half* Qptr = Q      + q_base;
    const __half* Kptr = K      + kv_base;
    const __half* Vptr = V      + kv_base;
          __half* Optr = output + q_base;
          float*  Lptr = Logsum + stat_base;

    const int Tr = (Sq + Br - 1) / Br;
    if (tileid >= Tr) return;

    const int q_block_start = tileid * Br;

    // Bottom-right causal alignment (same as the kStages kernel):
    // query row i may see keys 0 .. i + (Skv - Sq).
    const int kv_off = Skv - Sq;
    int kv_end = Skv;
    if constexpr (masked) {
        const int causal_end = q_block_start + Br + kv_off;
        if (causal_end < kv_end) kv_end = causal_end;
    }
    const int kv_tiles_to_process = kv_end > 0 ? (kv_end + Bc - 1) / Bc : 0;

    constexpr int PAD      = 8;
    constexpr int Q_STRIDE = D_PAD + PAD;   // same stride as K/V: Q aliases their region
    constexpr int K_STRIDE = D_PAD + PAD;
    constexpr int V_STRIDE = D_PAD + PAD;
    constexpr int Dk = D_PAD / 16;
    constexpr int Bk = Bc / 8;
    constexpr int Dv = D_PAD / 8;

    // smem layout (see flashattn_fwd_smem_bytes):
    //   phase 1: Q tile  = rows [0, Br)
    //   phase 2: K tile  = rows [0, Bc),  V tile = rows [Bc, 2*Bc)   (overwrites Q)
    extern __shared__ __align__(16) char smem_raw[];
    const uint32_t QS = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
    const uint32_t K0 = QS;
    const uint32_t V0 = K0 + Bc * K_STRIDE * sizeof(__half);

    // per-lane offsets (bytes), computed once
    const uint32_t q_lane =
        static_cast<uint32_t>(((warp * 16 + lane16) * Q_STRIDE + ((lane < 16) ? 0 : 8)) * sizeof(__half));
    const uint32_t k_lane =
        static_cast<uint32_t>(((lane16 & 7) * K_STRIDE + (lane16 >> 3) * 8) * sizeof(__half));
    const uint32_t v_lane =
        static_cast<uint32_t>((lane16 * V_STRIDE) * sizeof(__half));

    float O_frag[Dv * 4];
    #pragma unroll
    for (int i = 0; i < Dv * 4; ++i) O_frag[i] = 0.0f;
    float m_frag[2] = {-INFINITY, -INFINITY};   // running max, log2 domain
    float l_frag[2] = {0.0f, 0.0f};

    // scale * log2(e): scores live in the log2 domain, softmax uses exp2f.
    const float scale_log2 = (1.0f / sqrtf(static_cast<float>(D_PAD))) * kLog2e;

    // Load Q into the K/V region; it is consumed into registers below.
    load_tile_dispatch<Br, D_PAD, Q_STRIDE, FULL_TILES>(Qptr, QS, tid, tileid, Sq);
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    uint32_t q_frag[Dk][4];
    #pragma unroll
    for (int ks = 0; ks < Dk; ++ks) {
        ldmatrix_x4(q_frag[ks],
                    QS + q_lane + static_cast<uint32_t>(ks * 16 * sizeof(__half)));
    }
    // Every warp holds its Q fragments now; the Q rows may be overwritten by K/V.
    __syncthreads();

    // Prefetch K tile 0 into K buffer
    if (kv_tiles_to_process > 0) {
        load_tile_dispatch<Bc, D_PAD, K_STRIDE, FULL_TILES>(Kptr, K0, tid, 0, Skv);
        asm volatile("cp.async.commit_group;\n");
    }

    // Warp's first row in key coordinates; row0 sees keys <= row_lim0, row1 <= row_lim0 + 8.
    const int warp_row0 = q_block_start + warp * 16 + kv_off;
    const int row_lim0  = warp_row0 + lane / 4;
    const int row_lim1  = row_lim0 + 8;

    for (int kv_tile = 0; kv_tile < kv_tiles_to_process; ++kv_tile) {
        // K_j has landed; every warp finished P V_{j-1}, so the V buffer is free
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();

        // V_j streams in while we compute S = Q K_j^T
        load_tile_dispatch<Bc, D_PAD, V_STRIDE, FULL_TILES>(Vptr, V0, tid, kv_tile, Skv);
        asm volatile("cp.async.commit_group;\n");

        float S_frag[Bk * 4];
        #pragma unroll
        for (int i = 0; i < Bk * 4; ++i) S_frag[i] = 0.0f;

        #pragma unroll
        for (int ks = 0; ks < Dk; ++ks) {
            #pragma unroll
            for (int kb = 0; kb < Bk; ++kb) {
                uint32_t k_frag[2];
                ldmatrix_x2(k_frag,
                    K0 + k_lane +
                    static_cast<uint32_t>((kb * 8 * K_STRIDE + ks * 16) * sizeof(__half)));

                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(S_frag[kb*4+0]), "+f"(S_frag[kb*4+1]),
                      "+f"(S_frag[kb*4+2]), "+f"(S_frag[kb*4+3])
                    : "r"(q_frag[ks][0]), "r"(q_frag[ks][1]),
                      "r"(q_frag[ks][2]), "r"(q_frag[ks][3]),
                      "r"(k_frag[0]), "r"(k_frag[1])
                );
            }
        }

        // V_j has landed; every warp finished reading K_j, so K can be overwritten
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();

        // K_{j+1} streams in while we do softmax + P V_j
        const int next_kv_tile = kv_tile + 1;
        if (next_kv_tile < kv_tiles_to_process) {
            load_tile_dispatch<Bc, D_PAD, K_STRIDE, FULL_TILES>(Kptr, K0, tid, next_kv_tile, Skv);
            asm volatile("cp.async.commit_group;\n");
        }

        const int kv_start = kv_tile * Bc;

        // S = Q K^T * scale * log2(e)
        #pragma unroll
        for (int i = 0; i < Bk * 4; ++i) S_frag[i] *= scale_log2;

        // Mask keys past the end of K/V (ragged last tile).
        if constexpr (!FULL_TILES) {
            if (kv_start + Bc > Skv) {
                #pragma unroll
                for (int kb = 0; kb < Bk; ++kb) {
                    const int key0 = kv_start + kb * 8 + lane4 * 2;
                    if (key0     >= Skv) { S_frag[kb*4+0] = -INFINITY; S_frag[kb*4+2] = -INFINITY; }
                    if (key0 + 1 >= Skv) { S_frag[kb*4+1] = -INFINITY; S_frag[kb*4+3] = -INFINITY; }
                }
            }
        }

        // Causal mask, only for tiles that cross this warp's diagonal.
        if constexpr (masked) {
            if (kv_start + Bc - 1 > warp_row0) {
                #pragma unroll
                for (int kb = 0; kb < Bk; ++kb) {
                    const int key0 = kv_start + kb * 8 + lane4 * 2;
                    if (key0     > row_lim0) S_frag[kb*4+0] = -INFINITY;
                    if (key0 + 1 > row_lim0) S_frag[kb*4+1] = -INFINITY;
                    if (key0     > row_lim1) S_frag[kb*4+2] = -INFINITY;
                    if (key0 + 1 > row_lim1) S_frag[kb*4+3] = -INFINITY;
                }
            }
        }

        float local_max0 = -INFINITY;
        float local_max1 = -INFINITY;
        #pragma unroll
        for (int kb = 0; kb < Bk; ++kb) {
            local_max0 = fmaxf(local_max0, fmaxf(S_frag[kb*4+0], S_frag[kb*4+1]));
            local_max1 = fmaxf(local_max1, fmaxf(S_frag[kb*4+2], S_frag[kb*4+3]));
        }
        local_max0 = fmaxf(local_max0, __shfl_xor_sync(0xffffffffu, local_max0, 1, 4));
        local_max0 = fmaxf(local_max0, __shfl_xor_sync(0xffffffffu, local_max0, 2, 4));
        local_max1 = fmaxf(local_max1, __shfl_xor_sync(0xffffffffu, local_max1, 1, 4));
        local_max1 = fmaxf(local_max1, __shfl_xor_sync(0xffffffffu, local_max1, 2, 4));

        const float new_max0 = fmaxf(m_frag[0], local_max0);
        const float new_max1 = fmaxf(m_frag[1], local_max1);
        // A row with no visible key yet has max == -inf; subtract 0 instead so
        // exp2f(-inf - -inf) never produces NaN (or a bogus probability of 1).
        const float use_max0 = (new_max0 == -INFINITY) ? 0.0f : new_max0;
        const float use_max1 = (new_max1 == -INFINITY) ? 0.0f : new_max1;

        const float alpha0 = exp2f(m_frag[0] - use_max0);
        const float alpha1 = exp2f(m_frag[1] - use_max1);

        #pragma unroll
        for (int vs = 0; vs < Dv; ++vs) {
            O_frag[vs*4+0] *= alpha0;  O_frag[vs*4+1] *= alpha0;
            O_frag[vs*4+2] *= alpha1;  O_frag[vs*4+3] *= alpha1;
        }
        m_frag[0] = new_max0;
        m_frag[1] = new_max1;

        float tile_sum0 = 0.0f, tile_sum1 = 0.0f;

        #pragma unroll
        for (int kb = 0; kb < Bk; kb += 2) {
            const float p0 = exp2f(S_frag[ kb    * 4 + 0] - use_max0);
            const float p1 = exp2f(S_frag[ kb    * 4 + 1] - use_max0);
            const float p2 = exp2f(S_frag[ kb    * 4 + 2] - use_max1);
            const float p3 = exp2f(S_frag[ kb    * 4 + 3] - use_max1);
            const float p4 = exp2f(S_frag[(kb+1) * 4 + 0] - use_max0);
            const float p5 = exp2f(S_frag[(kb+1) * 4 + 1] - use_max0);
            const float p6 = exp2f(S_frag[(kb+1) * 4 + 2] - use_max1);
            const float p7 = exp2f(S_frag[(kb+1) * 4 + 3] - use_max1);

            tile_sum0 += p0 + p1 + p4 + p5;
            tile_sum1 += p2 + p3 + p6 + p7;

            uint32_t p_frag[4];
            p_frag[0] = pack_float2_to_half2_u32(p0, p1);
            p_frag[1] = pack_float2_to_half2_u32(p2, p3);
            p_frag[2] = pack_float2_to_half2_u32(p4, p5);
            p_frag[3] = pack_float2_to_half2_u32(p6, p7);

            #pragma unroll
            for (int vs = 0; vs < Dv; ++vs) {
                uint32_t v_frag[2];
                ldmatrix_x2_trans(v_frag,
                    V0 + v_lane +
                    static_cast<uint32_t>(((kb / 2) * 16 * V_STRIDE + vs * 8) * sizeof(__half)));

                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(O_frag[vs*4+0]), "+f"(O_frag[vs*4+1]),
                      "+f"(O_frag[vs*4+2]), "+f"(O_frag[vs*4+3])
                    : "r"(p_frag[0]), "r"(p_frag[1]),
                      "r"(p_frag[2]), "r"(p_frag[3]),
                      "r"(v_frag[0]), "r"(v_frag[1])
                );
            }
        }

        tile_sum0 += __shfl_xor_sync(0xffffffffu, tile_sum0, 1, 4);
        tile_sum0 += __shfl_xor_sync(0xffffffffu, tile_sum0, 2, 4);
        tile_sum1 += __shfl_xor_sync(0xffffffffu, tile_sum1, 1, 4);
        tile_sum1 += __shfl_xor_sync(0xffffffffu, tile_sum1, 2, 4);

        l_frag[0] = l_frag[0] * alpha0 + tile_sum0;
        l_frag[1] = l_frag[1] * alpha1 + tile_sum1;
    }

    // Epilogue: normalize once, guard ragged rows, convert base-2 max to natural-log LSE.
    const int row0 = q_block_start + warp * 16 + lane / 4;
    const int row1 = row0 + 8;
    const int col0 = lane4 * 2;
    const float inv_l0 = 1.0f / l_frag[0];
    const float inv_l1 = 1.0f / l_frag[1];

    #pragma unroll
    for (int vs = 0; vs < Dv; ++vs) {
        const int c = vs * 8 + col0;
        if (FULL_TILES || row0 < Sq) {
            *reinterpret_cast<__half2*>(&Optr[static_cast<size_t>(row0) * D_PAD + c]) =
                __floats2half2_rn(O_frag[vs*4+0] * inv_l0, O_frag[vs*4+1] * inv_l0);
        }
        if (FULL_TILES || row1 < Sq) {
            *reinterpret_cast<__half2*>(&Optr[static_cast<size_t>(row1) * D_PAD + c]) =
                __floats2half2_rn(O_frag[vs*4+2] * inv_l1, O_frag[vs*4+3] * inv_l1);
        }
    }

    if (lane4 == 0) {
        if (FULL_TILES || row0 < Sq) Lptr[row0] = m_frag[0] * kLn2 + logf(l_frag[0]);
        if (FULL_TILES || row1 < Sq) Lptr[row1] = m_frag[1] * kLn2 + logf(l_frag[1]);
    }
}
