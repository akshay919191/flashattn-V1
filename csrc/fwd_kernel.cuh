#pragma once

#include "mma_helpers.cuh"
template<int Br, int Bc, int D_PAD, bool masked , bool FULL_TILES>
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
    // Bc no longer needs to be >= Br: Q has its own buffer now.

    if (blockDim.x != 128) return;
    if (actual_D != D_PAD || Sq <= 0 || Skv <= 0) return;
    if ((Sq % Br) != 0 || (Skv % Bc) != 0) return;   // full tiles only

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

    const int Tr = Sq / Br;
    const int Tc = Skv / Bc;
    if (tileid >= Tr) return;

    const int q_block_start = tileid * Br;
    const int q_block_last  = q_block_start + Br - 1;

    int kv_tiles_to_process = Tc;
    if constexpr (masked) {
        const int causal_tiles = q_block_last / Bc + 1;
        if (causal_tiles < kv_tiles_to_process) kv_tiles_to_process = causal_tiles;
    }

    constexpr int PAD      = 8;
    constexpr int Q_STRIDE = D_PAD + PAD;   // Q has its own buffer
    constexpr int K_STRIDE = D_PAD + PAD;
    constexpr int V_STRIDE = D_PAD + PAD;
    constexpr int Dk = D_PAD / 16;
    constexpr int Bk = Bc / 8;
    constexpr int Dv = D_PAD / 8;

    // smem layout: [ Q tile (Br x Q_STRIDE) | K tile (Bc x K_STRIDE) | V tile (Bc x V_STRIDE) ]
    extern __shared__ __align__(16) char smem_raw[];
    const uint32_t QS = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
    const uint32_t K0 = QS + Br * Q_STRIDE * sizeof(__half);
    const uint32_t V0 = K0 + Bc * K_STRIDE * sizeof(__half);

    // per-lane offsets (bytes), computed once
    // Q offset: same row/col layout as before, but into the Q buffer
    const uint32_t q_lane =
        static_cast<uint32_t>(((warp * 16 + lane16) * Q_STRIDE + ((lane < 16) ? 0 : 8)) * sizeof(__half));
    const uint32_t k_lane =
        static_cast<uint32_t>(((lane16 & 7) * K_STRIDE + (lane16 >> 3) * 8) * sizeof(__half));
    const uint32_t v_lane =
        static_cast<uint32_t>((lane16 * V_STRIDE) * sizeof(__half));

    float O_frag[Dv * 4];
    #pragma unroll
    for (int i = 0; i < Dv * 4; ++i) O_frag[i] = 0.0f;
    float m_frag[2] = {-FLT_MAX, -FLT_MAX};
    float l_frag[2] = {0.0f, 0.0f};

    const float scale = 1.0f / sqrtf(static_cast<float>(D_PAD));

    // Load Q into its own dedicated buffer (never overwritten)
    load_tile_full<Br, D_PAD, Q_STRIDE>(Qptr, QS, tid, tileid);
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    // Read Q fragments from the Q buffer
    uint32_t q_frag[Dk][4];
    #pragma unroll
    for (int ks = 0; ks < Dk; ++ks) {
        ldmatrix_x4(q_frag[ks],
                    QS + q_lane + static_cast<uint32_t>(ks * 16 * sizeof(__half)));
    }
    // Prefetch K tile 0 into K buffer
    load_tile_full<Bc, D_PAD, K_STRIDE>(Kptr, K0, tid, 0);
    asm volatile("cp.async.commit_group;\n");

    for (int kv_tile = 0; kv_tile < kv_tiles_to_process; ++kv_tile) {
        // K_j has landed; every warp finished P V_{j-1}, so the V buffer is free
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();

        // V_j streams in while we compute S = Q K_j^T
        load_tile_full<Bc, D_PAD, V_STRIDE>(Vptr, V0, tid, kv_tile);
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
            load_tile_full<Bc, D_PAD, K_STRIDE>(Kptr, K0, tid, next_kv_tile);
            asm volatile("cp.async.commit_group;\n");
        }

        const int kv_start = kv_tile * Bc;
        bool needs_causal_mask = false;
        if constexpr (masked) {
            needs_causal_mask = !((kv_start + Bc - 1) <= q_block_start);
        }
        float tile_max[2] = {-FLT_MAX, -FLT_MAX};

        float local_max0 = -FLT_MAX;
        float local_max1 = -FLT_MAX;

        if (!needs_causal_mask) {
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

                local_max0 = fmaxf(local_max0, fmaxf(s0, s1));
                local_max1 = fmaxf(local_max1, fmaxf(s2, s3));
            }
        } else {
            const int query0 = q_block_start + warp * 16 + lane / 4;
            const int query1 = query0 + 8;

            #pragma unroll
            for (int kb = 0; kb < Bk; ++kb) {
                const int key0 = kv_start + kb * 8 + lane4 * 2;
                const int key1 = key0 + 1;

                float s0 = S_frag[kb * 4 + 0] * scale;
                float s1 = S_frag[kb * 4 + 1] * scale;
                float s2 = S_frag[kb * 4 + 2] * scale;
                float s3 = S_frag[kb * 4 + 3] * scale;

                if (key0 > query0) s0 = -FLT_MAX;
                if (key1 > query0) s1 = -FLT_MAX;
                if (key0 > query1) s2 = -FLT_MAX;
                if (key1 > query1) s3 = -FLT_MAX;

                S_frag[kb * 4 + 0] = s0;
                S_frag[kb * 4 + 1] = s1;
                S_frag[kb * 4 + 2] = s2;
                S_frag[kb * 4 + 3] = s3;

                local_max0 = fmaxf(local_max0, fmaxf(s0, s1));
                local_max1 = fmaxf(local_max1, fmaxf(s2, s3));
            }
        }

        local_max0 = fmaxf(
            local_max0,
            __shfl_xor_sync(0xffffffffu, local_max0, 1, 4)
        );
        local_max0 = fmaxf(
            local_max0,
            __shfl_xor_sync(0xffffffffu, local_max0, 2, 4)
        );

        local_max1 = fmaxf(
            local_max1,
            __shfl_xor_sync(0xffffffffu, local_max1, 1, 4)
        );
        local_max1 = fmaxf(
            local_max1,
            __shfl_xor_sync(0xffffffffu, local_max1, 2, 4)
        );

        tile_max[0] = local_max0;
        tile_max[1] = local_max1;

        const float old_max0 = m_frag[0];
        const float old_max1 = m_frag[1];
        const float new_max0 = fmaxf(old_max0, tile_max[0]);
        const float new_max1 = fmaxf(old_max1, tile_max[1]);

        const float alpha0 = __expf(old_max0 - new_max0);
        const float alpha1 = __expf(old_max1 - new_max1);

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
            const float p0 = __expf(S_frag[ kb    * 4 + 0] - new_max0);
            const float p1 = __expf(S_frag[ kb    * 4 + 1] - new_max0);
            const float p2 = __expf(S_frag[ kb    * 4 + 2] - new_max1);
            const float p3 = __expf(S_frag[ kb    * 4 + 3] - new_max1);
            const float p4 = __expf(S_frag[(kb+1) * 4 + 0] - new_max0);
            const float p5 = __expf(S_frag[(kb+1) * 4 + 1] - new_max0);
            const float p6 = __expf(S_frag[(kb+1) * 4 + 2] - new_max1);
            const float p7 = __expf(S_frag[(kb+1) * 4 + 3] - new_max1);

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

    const int row0 = q_block_start + warp * 16 + lane / 4;
    const int row1 = row0 + 8;
    const int col0 = lane4 * 2;
    const float inv_l0 = 1.0f / l_frag[0];
    const float inv_l1 = 1.0f / l_frag[1];

    #pragma unroll
    for (int vs = 0; vs < Dv; ++vs) {
        const int c = vs * 8 + col0;
        *reinterpret_cast<__half2*>(&Optr[static_cast<size_t>(row0) * D_PAD + c]) =
            __floats2half2_rn(O_frag[vs*4+0] * inv_l0, O_frag[vs*4+1] * inv_l0);
        *reinterpret_cast<__half2*>(&Optr[static_cast<size_t>(row1) * D_PAD + c]) =
            __floats2half2_rn(O_frag[vs*4+2] * inv_l1, O_frag[vs*4+3] * inv_l1);
    }

    if (lane4 == 0) {
        Lptr[row0] = m_frag[0] + logf(l_frag[0]);
        Lptr[row1] = m_frag[1] + logf(l_frag[1]);
    }
}
