#pragma once

#include "mma_helpers.cuh"

template<int Br, int Bc, int D_PAD,
         bool masked, bool FULL_TILES = false>
__global__ void __launch_bounds__(128)
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
    static_assert(Br == 64,
                  "This kernel requires Br == 64");
    static_assert(Bc > 0 && Bc % 16 == 0,
                  "Bc must be positive and divisible by 16");
    static_assert(D_PAD > 0 && D_PAD % 16 == 0,
                  "D_PAD must be positive and divisible by 16");

    if (blockDim.x != 128) return;
    if (actual_D <= 0 || actual_D > D_PAD || Sq <= 0 || Skv <= 0) return;

    const int tid   = threadIdx.x;
    const int warp  = tid >> 5;
    const int lane  = tid & 31;
    const int lane4 = lane & 3;


const int tileid    = blockIdx.x;
const int headid    = blockIdx.y;
const int batchid   = blockIdx.z;
const int num_heads = gridDim.y;

    if (numKVheads <= 0 || num_heads % numKVheads != 0) return;
    const int kv_headid = headid / (num_heads / numKVheads);

    const long long q_base =
        (static_cast<long long>(batchid) * num_heads + headid) * Sq * actual_D;
    const long long kv_base =
        (static_cast<long long>(batchid) * numKVheads + kv_headid) * Skv * actual_D;
    const long long stat_base =
        (static_cast<long long>(batchid) * num_heads + headid) * Sq;

    const __half* Qptr  = Q      + q_base;
    const __half* Kptr  = K      + kv_base;
    const __half* Vptr  = V      + kv_base;
          __half* Optr  = output + q_base;
          float*  Lptr  = Logsum + stat_base;

    const int Tr = (Sq  + Br - 1) / Br;
    const int Tc = (Skv + Bc - 1) / Bc;

    if (tileid >= Tr) return;

    const int q_block_start         = tileid * Br;
    const int q_block_last_unclamped = q_block_start + Br - 1;
    const int q_block_last =
        q_block_last_unclamped < Sq ? q_block_last_unclamped : Sq - 1;

    int kv_tiles_to_process = Tc;
    if constexpr (masked) {
        const int causal_tiles = q_block_last / Bc + 1;
        if (causal_tiles < kv_tiles_to_process)
            kv_tiles_to_process = causal_tiles;
    }
 
    constexpr int PAD      = 8;
    constexpr int Q_STRIDE = D_PAD + PAD;
    constexpr int K_STRIDE = D_PAD + PAD;
    constexpr int V_STRIDE = D_PAD + PAD;

    extern __shared__ char smem_raw[];
    char* ptr = smem_raw;

    auto align_ptr = [&](size_t alignment = 16) {
        const uintptr_t value = reinterpret_cast<uintptr_t>(ptr);
        ptr = reinterpret_cast<char*>((value + alignment - 1) & ~(alignment - 1));
    };

    align_ptr();
    __half* Qsmem  = reinterpret_cast<__half*>(ptr);
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
 
    constexpr int Dk = D_PAD / 16; // #K-tiles along head dim
    constexpr int Bk = Bc / 8;     // #score tiles along KV dim
    constexpr int Dv = D_PAD / 8;  // #V-tiles along head dim

    float O_frag[Dv * 4] = {0.0f};
    float m_frag[2]       = {-FLT_MAX, -FLT_MAX};
    float l_frag[2]       = {0.0f, 0.0f};

    const float scale = 1.0f / sqrtf(static_cast<float>(actual_D));
 
    if constexpr (FULL_TILES) {
        load_tile_full<Br, D_PAD, Q_STRIDE>(
            Qptr, smem_u32_ptr(Qsmem), tid, tileid);

        load_tile_full<Bc, D_PAD, K_STRIDE>(
            Kptr, smem_u32_ptr(Ksmem0), tid, 0);

        load_tile_full<Bc, D_PAD, V_STRIDE>(
            Vptr, smem_u32_ptr(Vsmem0), tid, 0);
    } else {
        asyncLOAD_2D_TILE<Br, D_PAD, 128>(
            Qptr, smem_u32_ptr(Qsmem), tid, Q_STRIDE,
            Sq, actual_D, actual_D, tileid, 0);

        asyncLOAD_2D_TILE<Bc, D_PAD, 128>(
            Kptr, smem_u32_ptr(Ksmem[0]), tid, K_STRIDE,
            Skv, actual_D, actual_D, 0, 0);

        asyncLOAD_2D_TILE<Bc, D_PAD, 128>(
            Vptr, smem_u32_ptr(Vsmem[0]), tid, V_STRIDE,
            Skv, actual_D, actual_D, 0, 0);
    }

    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
 
    for (int kv_tile = 0; kv_tile < kv_tiles_to_process; ++kv_tile) {
        const int current_stage = kv_tile & 1;
        const int next_stage    = current_stage ^ 1;
        const int next_kv_tile  = kv_tile + 1;

        // prefetch next tile
        if constexpr (FULL_TILES) {
            constexpr uint32_t TILE_BYTES =
                Bc * K_STRIDE * sizeof(__half);

            const uint32_t K0 = smem_u32_ptr(Ksmem0);
            const uint32_t V0 = smem_u32_ptr(Vsmem0);

            load_tile_full<Bc, D_PAD, K_STRIDE>(
                Kptr,
                K0 + static_cast<uint32_t>(next_stage) * TILE_BYTES,
                tid,
                next_kv_tile);

            load_tile_full<Bc, D_PAD, V_STRIDE>(
                Vptr,
                V0 + static_cast<uint32_t>(next_stage) * TILE_BYTES,
                tid,
                next_kv_tile);
        } else {
            asyncLOAD_2D_TILE<Bc, D_PAD, 128>(
                Kptr, smem_u32_ptr(Ksmem[next_stage]), tid, K_STRIDE,
                Skv, actual_D, actual_D, next_kv_tile, 0);

            asyncLOAD_2D_TILE<Bc, D_PAD, 128>(
                Vptr, smem_u32_ptr(Vsmem[next_stage]), tid, V_STRIDE,
                Skv, actual_D, actual_D, next_kv_tile, 0);
        }

        asm volatile("cp.async.commit_group;\n");
        
        float S_frag[Bk * 4] = {0.0f};

        const int q_row  = warp * 16 + (lane & 15);
        const int lane16 = lane & 15;

        #pragma unroll
        for (int ks = 0; ks < Dk; ++ks) {
            uint32_t q_frag[4];
            const int q_col = ks * 16 + ((lane < 16) ? 0 : 8);
            ldmatrix_x4(q_frag, smem_u32_ptr(Qsmem + q_row * Q_STRIDE + q_col));

            #pragma unroll
            for (int kb = 0; kb < Bk; ++kb) {
                uint32_t k_frag[2];
                const int k_row = kb * 8 + (lane16 & 7);
                const int k_col = ks * 16 + ((lane16 >> 3) * 8);
                ldmatrix_x2(k_frag,
                    smem_u32_ptr(Ksmem[current_stage] + k_row * K_STRIDE + k_col));

                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(S_frag[kb*4+0]), "+f"(S_frag[kb*4+1]),
                      "+f"(S_frag[kb*4+2]), "+f"(S_frag[kb*4+3])
                    : "r"(q_frag[0]), "r"(q_frag[1]),
                      "r"(q_frag[2]), "r"(q_frag[3]),
                      "r"(k_frag[0]), "r"(k_frag[1])
                );
            }
        }

        const int kv_start           = kv_tile * Bc;
        const int kv_last_unclamped  = kv_start + Bc - 1;
        const int kv_real_last       = kv_last_unclamped < Skv
                                           ? kv_last_unclamped
                                           : Skv - 1;

        const bool q_tile_full  = q_block_last_unclamped < Sq;
        const bool kv_tile_full = kv_last_unclamped < Skv;

        bool needs_causal_mask = false;
        if constexpr (masked) {
            needs_causal_mask = !(kv_real_last <= q_block_start);
        }
        const bool needs_any_mask = !q_tile_full || !kv_tile_full || needs_causal_mask;

        float tile_max[2] = {-FLT_MAX, -FLT_MAX};

        if (!needs_any_mask) {
            #pragma unroll
            for (int kb = 0; kb < Bk; ++kb) {
                const float s0 = S_frag[kb*4+0] * scale;
                const float s1 = S_frag[kb*4+1] * scale;
                const float s2 = S_frag[kb*4+2] * scale;
                const float s3 = S_frag[kb*4+3] * scale;
                S_frag[kb*4+0] = s0; S_frag[kb*4+1] = s1;
                S_frag[kb*4+2] = s2; S_frag[kb*4+3] = s3;

                float max0 = fmaxf(s0, s1);
                float max1 = fmaxf(s2, s3);
                max0 = fmaxf(max0, __shfl_xor_sync(0xffffffffu, max0, 1, 4));
                max0 = fmaxf(max0, __shfl_xor_sync(0xffffffffu, max0, 2, 4));
                max1 = fmaxf(max1, __shfl_xor_sync(0xffffffffu, max1, 1, 4));
                max1 = fmaxf(max1, __shfl_xor_sync(0xffffffffu, max1, 2, 4));
                tile_max[0] = fmaxf(tile_max[0], max0);
                tile_max[1] = fmaxf(tile_max[1], max1);
            }
        } else {
            const int query0            = q_block_start + warp * 16 + lane / 4;
            const int query1            = query0 + 8;
            const bool query0_in_bounds = query0 < Sq;
            const bool query1_in_bounds = query1 < Sq;

            #pragma unroll
            for (int kb = 0; kb < Bk; ++kb) {
                const int key0           = kv_start + kb * 8 + lane4 * 2;
                const int key1           = key0 + 1;
                const bool key0_in_bounds = key0 < Skv;
                const bool key1_in_bounds = key1 < Skv;

                float s0 = S_frag[kb*4+0] * scale;
                float s1 = S_frag[kb*4+1] * scale;
                float s2 = S_frag[kb*4+2] * scale;
                float s3 = S_frag[kb*4+3] * scale;

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

                S_frag[kb*4+0] = s0; S_frag[kb*4+1] = s1;
                S_frag[kb*4+2] = s2; S_frag[kb*4+3] = s3;

                float max0 = fmaxf(s0, s1);
                float max1 = fmaxf(s2, s3);
                max0 = fmaxf(max0, __shfl_xor_sync(0xffffffffu, max0, 1, 4));
                max0 = fmaxf(max0, __shfl_xor_sync(0xffffffffu, max0, 2, 4));
                max1 = fmaxf(max1, __shfl_xor_sync(0xffffffffu, max1, 1, 4));
                max1 = fmaxf(max1, __shfl_xor_sync(0xffffffffu, max1, 2, 4));
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

            const int v_row = (kb / 2) * 16 + (lane & 15);

            #pragma unroll
            for (int vs = 0; vs < Dv; ++vs) {
                uint32_t v_frag[2];
                const int v_col = vs * 8;
                ldmatrix_x2_trans(v_frag,
                    smem_u32_ptr(Vsmem[current_stage] + v_row * V_STRIDE + v_col));

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

        if (next_kv_tile < kv_tiles_to_process) {
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
            __syncthreads();
        }
    }

    const int row0  = q_block_start + warp * 16 + lane / 4;
    const int row1  = row0 + 8;
    const int col0  = lane4 * 2;
    const int col1  = col0 + 1;
    const float inv_l0 = 1.0f / l_frag[0];
    const float inv_l1 = 1.0f / l_frag[1];

    #pragma unroll
    for (int vs = 0; vs < Dv; ++vs) {
        const int out_col0 = vs * 8 + col0;
        const int out_col1 = vs * 8 + col1;

        if (row0 < Sq && out_col0 < actual_D)
            Optr[static_cast<size_t>(row0) * actual_D + out_col0] =
                __float2half(O_frag[vs*4+0] * inv_l0);
        if (row0 < Sq && out_col1 < actual_D)
            Optr[static_cast<size_t>(row0) * actual_D + out_col1] =
                __float2half(O_frag[vs*4+1] * inv_l0);
        if (row1 < Sq && out_col0 < actual_D)
            Optr[static_cast<size_t>(row1) * actual_D + out_col0] =
                __float2half(O_frag[vs*4+2] * inv_l1);
        if (row1 < Sq && out_col1 < actual_D)
            Optr[static_cast<size_t>(row1) * actual_D + out_col1] =
                __float2half(O_frag[vs*4+3] * inv_l1);
    }

    if (lane4 == 0) {
        if (row0 < Sq) Lptr[row0] = m_frag[0] + logf(l_frag[0]);
        if (row1 < Sq) Lptr[row1] = m_frag[1] + logf(l_frag[1]);
    }
}
