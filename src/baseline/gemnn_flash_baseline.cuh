
#pragma once

#include <cuda.h>
#include <cuda_fp16.h>  
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <iostream>

#include "helpers.cuh"

#define PADDED_STRIDE 0
#define STAGES 2


template<int Br>
__device__ __forceinline__ void m16n8k16_qkT(
    float& d1, float& d2, float& d3, float& d4,
    const uint32_t* a_frag,
    const __half* smem_k,
    const int lane,
    const int group,
    int strideK,
    int whichcol,
    int maincol
) {
    uint32_t b_frag[2];

    int k0 = whichcol * 16 + (lane % 4) * 2;
    int k1 = k0 + 8;
    int n  = maincol * 8 + group;

    if (n < Br) {
        b_frag[0] =
            uint32_t(__half_as_ushort(smem_k[n * strideK + k0])) |
            (uint32_t(__half_as_ushort(smem_k[n * strideK + k0 + 1])) << 16);

        b_frag[1] =
            uint32_t(__half_as_ushort(smem_k[n * strideK + k1])) |
            (uint32_t(__half_as_ushort(smem_k[n * strideK + k1 + 1])) << 16);
    } else {
        b_frag[0] = 0;
        b_frag[1] = 0;
    }

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},"
        "{%4,%5,%6,%7},"
        "{%8,%9},"
        "{%10,%11,%12,%13};"
        : "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4)
        : "r"(a_frag[0]), "r"(a_frag[1]),
          "r"(a_frag[2]), "r"(a_frag[3]),
          "r"(b_frag[0]), "r"(b_frag[1]),
          "f"(d1), "f"(d2), "f"(d3), "f"(d4)
    );
}


template<int Br, int NUM_HEADS, int SEQLEN, int HEAD_DIM>
__global__ void flashattn_fwd_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ output,
    float* __restrict__ L_out
) {
    const int tid   = threadIdx.x;
    const int warp  = tid / 32;
    const int lane  = tid % 32;
    const int group = lane / 4;

    const int batchid = blockIdx.x;
    const int headid  = blockIdx.y;
    const int tileitr = blockIdx.z;

    const long long base =
        (long long)batchid * NUM_HEADS * SEQLEN * HEAD_DIM +
        (long long)headid  * SEQLEN * HEAD_DIM;

    const __half* Qptr = Q + base;
    const __half* Kptr = K + base;
    const __half* Vptr = V + base;
    __half* Optr = output + base;

    const long long stat_base =
        (long long)batchid * NUM_HEADS * SEQLEN +
        (long long)headid  * SEQLEN;

    float* Lptr = L_out + stat_base;

    extern __shared__ char smem_raw[];
    char* ptr = smem_raw;

    const int SMEM_STRIDE_KV = HEAD_DIM + PADDED_STRIDE;
    const int TILE_ELEMENTS = Br * SMEM_STRIDE_KV;

    // K is still double-buffered.
    __half* k_base = reinterpret_cast<__half*>(ptr);

    __half* k_stages[STAGES] = {
        k_base,
        k_base + TILE_ELEMENTS
    };

    // V is now SINGLE-buffered.
    // This saves one full V tile in shared memory.
    __half* v_stage = k_base + TILE_ELEMENTS * STAGES;

    ptr = reinterpret_cast<char*>(v_stage + TILE_ELEMENTS);
    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 15) & ~15ULL
    );


    float* s_mem = reinterpret_cast<float*>(ptr);
    float* o_acc = s_mem + Br * Br;
    float* m_shared = o_acc + Br * HEAD_DIM;
    float* l_shared = m_shared + Br;

    const float SCALE = 1.0f / sqrtf((float)HEAD_DIM);
    const int Qrowitr = tileitr;
    const int numtiles = (SEQLEN + Br - 1) / Br;

    const int kv_tiles = min(numtiles, Qrowitr + 1);


    constexpr int Tk = HEAD_DIM / 16;
    constexpr int Q_REGS = 4 * Tk;

    uint32_t q_reg[Q_REGS];

    const int q_row_base = warp * 16;

    #pragma unroll
    for (int k = 0; k < Tk; k++) {
        int row0 = q_row_base + (lane / 4);
        int row1 = row0 + 8;

        int col0 = k * 16 + (lane % 4) * 2;
        int col1 = col0 + 8;

        int base_idx = k * 4;

        int global_row0 = Qrowitr * Br + row0;
        int global_row1 = Qrowitr * Br + row1;

        if (row0 < Br && global_row0 < SEQLEN) {
            q_reg[base_idx + 0] =
                *reinterpret_cast<const uint32_t*>(
                    &Qptr[global_row0 * HEAD_DIM + col0]
                );

            q_reg[base_idx + 2] =
                *reinterpret_cast<const uint32_t*>(
                    &Qptr[global_row0 * HEAD_DIM + col1]
                );
        } else {
            q_reg[base_idx + 0] = 0;
            q_reg[base_idx + 2] = 0;
        }

        if (row1 < Br && global_row1 < SEQLEN) {
            q_reg[base_idx + 1] =
                *reinterpret_cast<const uint32_t*>(
                    &Qptr[global_row1 * HEAD_DIM + col0]
                );

            q_reg[base_idx + 3] =
                *reinterpret_cast<const uint32_t*>(
                    &Qptr[global_row1 * HEAD_DIM + col1]
                );
        } else {
            q_reg[base_idx + 1] = 0;
            q_reg[base_idx + 3] = 0;
        }
    }

    for (int i = tid; i < Br * HEAD_DIM; i += 128) {
        o_acc[i] = 0.f;
    }

    for (int i = tid; i < Br; i += 128) {
        m_shared[i] = -FLT_MAX;
        l_shared[i] = 0.f;
    }

    __syncthreads();

    const int float4s_per_tile = (Br * HEAD_DIM) / 8;

    uint32_t smenptr0 = __cvta_generic_to_shared(k_stages[0]);
    asyncLOAD<Br, HEAD_DIM, 128>(
        Kptr,
        smenptr0,
        tid,
        HEAD_DIM,
        SMEM_STRIDE_KV,
        float4s_per_tile,
        SEQLEN,
        HEAD_DIM,
        0,
        0
    );

    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n");

    for (int tile = 0; tile < kv_tiles; tile++)
    {
        int cur = tile % STAGES;
        int next = tile + 1;
        int next_stage = (tile + 1) % STAGES;

        // Prefetch next K only.
        // We do not wait for it here because current K is already ready.
        if (next < kv_tiles) {
            uint32_t smenptr = __cvta_generic_to_shared(k_stages[next_stage]);

            asyncLOAD<Br, HEAD_DIM, 128>(
                Kptr,
                smenptr,
                tid,
                HEAD_DIM,
                SMEM_STRIDE_KV,
                float4s_per_tile,
                SEQLEN,
                HEAD_DIM,
                next,
                0
            );

            asm volatile("cp.async.commit_group;\n");
        }

        constexpr int N_GROUPS = (Br + 7) / 8;

        if (q_row_base < Br) {
            #pragma unroll
            for (int mc = 0; mc < N_GROUPS; mc++) {
                float acc_s[4] = {0.f, 0.f, 0.f, 0.f};

                #pragma unroll
                for (int kc = 0; kc < Tk; kc++) {
                    m16n8k16_qkT<Br>(
                        acc_s[0],
                        acc_s[1],
                        acc_s[2],
                        acc_s[3],
                        &q_reg[kc * 4],
                        k_stages[cur],
                        lane,
                        group,
                        SMEM_STRIDE_KV,
                        kc,
                        mc
                    );
                }

                int local_row0 = q_row_base + (lane / 4);
                int local_row1 = local_row0 + 8;

                int col0 = mc * 8 + (lane % 4) * 2;
                int col1 = col0 + 1;

                int global_row0 = Qrowitr * Br + local_row0;
                int global_row1 = Qrowitr * Br + local_row1;

                int kv_col0 = tile * Br + col0;
                int kv_col1 = tile * Br + col1;

                float s00 = acc_s[0] * SCALE;
                float s01 = acc_s[1] * SCALE;
                float s10 = acc_s[2] * SCALE;
                float s11 = acc_s[3] * SCALE;

                bool row0_ok = (local_row0 < Br);
                bool row1_ok = (local_row1 < Br);

                bool col0_ok = (col0 < Br);
                bool col1_ok = (col1 < Br);

                bool row0_valid = (global_row0 < SEQLEN);
                bool row1_valid = (global_row1 < SEQLEN);

                bool col0_valid = (kv_col0 < SEQLEN);
                bool col1_valid = (kv_col1 < SEQLEN);

                bool diagonal_tile = (tile == Qrowitr);

                if (row0_ok && col0_ok) {
                    bool mask00 =
                        (!row0_valid) ||
                        (!col0_valid) ||
                        (diagonal_tile && kv_col0 > global_row0);

                    s_mem[local_row0 * Br + col0] =
                        mask00 ? -FLT_MAX : s00;
                }

                if (row0_ok && col1_ok) {
                    bool mask01 =
                        (!row0_valid) ||
                        (!col1_valid) ||
                        (diagonal_tile && kv_col1 > global_row0);

                    s_mem[local_row0 * Br + col1] =
                        mask01 ? -FLT_MAX : s01;
                }

                if (row1_ok && col0_ok) {
                    bool mask10 =
                        (!row1_valid) ||
                        (!col0_valid) ||
                        (diagonal_tile && kv_col0 > global_row1);

                    s_mem[local_row1 * Br + col0] =
                        mask10 ? -FLT_MAX : s10;
                }

                if (row1_ok && col1_ok) {
                    bool mask11 =
                        (!row1_valid) ||
                        (!col1_valid) ||
                        (diagonal_tile && kv_col1 > global_row1);

                    s_mem[local_row1 * Br + col1] =
                        mask11 ? -FLT_MAX : s11;
                }
            }
        }

        __syncthreads();

        {
            uint32_t v_smem = __cvta_generic_to_shared(v_stage);

            asyncLOAD<Br, HEAD_DIM, 128>(
                Vptr,
                v_smem,
                tid,
                HEAD_DIM,
                SMEM_STRIDE_KV,
                float4s_per_tile,
                SEQLEN,
                HEAD_DIM,
                tile,
                0
            );

            asm volatile("cp.async.commit_group;\n");
        }

        __half* p_half = k_stages[cur];

        // Online softmax with unnormalized O accumulator.
        for (int row = warp; row < Br; row += 4) {
            int global_row = Qrowitr * Br + row;

            if (global_row >= SEQLEN) {
                continue;
            }

            float m_val = -FLT_MAX;

            for (int c = lane; c < Br; c += 32) {
                m_val = fmaxf(m_val, s_mem[row * Br + c]);
            }

            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                m_val = fmaxf(
                    m_val,
                    __shfl_xor_sync(0xffffffff, m_val, offset)
                );
            }

            float l_val = 0.f;
            
            for (int c = lane; c < Br; c += 32) {
                l_val += __expf(s_mem[row * Br + c] - m_val);
            }

            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                l_val += __shfl_xor_sync(0xffffffff, l_val, offset);
            }

            float m_prev = m_shared[row];
            float l_prev = l_shared[row];

            float m_new = fmaxf(m_prev, m_val);

            float alpha = __expf(m_prev - m_new);
            float beta  = __expf(m_val  - m_new);

            float l_new = alpha * l_prev + beta * l_val;

            for (int c = lane; c < HEAD_DIM; c += 32) {
                o_acc[row * HEAD_DIM + c] *= alpha;
            }

            m_shared[row] = m_new;
            l_shared[row] = l_new;

            for (int c = lane; c < Br; c += 32) {
                float p = beta * __expf(s_mem[row * Br + c] - m_val);
                p_half[row * Br + c] = __float2half(p);
            }
        }

        asm volatile("cp.async.wait_group 1;\n" ::: "memory");
        __syncthreads();

        // O += P @ V
        {
            const int num_row_tiles = (Br + 15) / 16;
            const int num_col_tiles = (HEAD_DIM + 7) / 8;
            const int num_k_tiles_p = (Br + 15) / 16;

            for (
                int tile_idx = warp;
                tile_idx < num_row_tiles * num_col_tiles;
                tile_idx += 4
            ) {
                int tile_row = tile_idx / num_col_tiles;
                int tile_col = tile_idx % num_col_tiles;

                int row_start = tile_row * 16;
                int col_start = tile_col * 8;

                if (row_start >= Br || col_start >= HEAD_DIM) {
                    continue;
                }

                float d[4] = {0.f, 0.f, 0.f, 0.f};

                for (int k_tile = 0; k_tile < num_k_tiles_p; k_tile++) {
                    int k_start = k_tile * 16;

                    int c0 = (lane % 4) * 2;

                    uint32_t a_frag[4];

                    a_frag[0] =
                        *reinterpret_cast<const uint32_t*>(
                            &p_half[(row_start + group) * Br + (k_start + c0)]
                        );

                    a_frag[1] =
                        *reinterpret_cast<const uint32_t*>(
                            &p_half[(row_start + group + 8) * Br + (k_start + c0)]
                        );

                    a_frag[2] =
                        *reinterpret_cast<const uint32_t*>(
                            &p_half[(row_start + group) * Br + (k_start + c0 + 8)]
                        );

                    a_frag[3] =
                        *reinterpret_cast<const uint32_t*>(
                            &p_half[(row_start + group + 8) * Br + (k_start + c0 + 8)]
                        );

                    uint32_t b_frag[2];

                    int v_col = col_start + group;

                    int v_row0 = k_start + c0;
                    int v_row1 = k_start + c0 + 1;
                    int v_row2 = k_start + c0 + 8;
                    int v_row3 = k_start + c0 + 9;

                    uint32_t bv0 =
                        (v_row0 < Br && v_col < HEAD_DIM)
                        ? uint32_t(__half_as_ushort(v_stage[v_row0 * SMEM_STRIDE_KV + v_col]))
                        : 0u;

                    uint32_t bv1 =
                        (v_row1 < Br && v_col < HEAD_DIM)
                        ? uint32_t(__half_as_ushort(v_stage[v_row1 * SMEM_STRIDE_KV + v_col]))
                        : 0u;

                    uint32_t bv2 =
                        (v_row2 < Br && v_col < HEAD_DIM)
                        ? uint32_t(__half_as_ushort(v_stage[v_row2 * SMEM_STRIDE_KV + v_col]))
                        : 0u;

                    uint32_t bv3 =
                        (v_row3 < Br && v_col < HEAD_DIM)
                        ? uint32_t(__half_as_ushort(v_stage[v_row3 * SMEM_STRIDE_KV + v_col]))
                        : 0u;

                    b_frag[0] = bv0 | (bv1 << 16);
                    b_frag[1] = bv2 | (bv3 << 16);

                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3},"
                        "{%4,%5,%6,%7},"
                        "{%8,%9},"
                        "{%0,%1,%2,%3};\n"
                        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]),
                          "r"(a_frag[2]), "r"(a_frag[3]),
                          "r"(b_frag[0]), "r"(b_frag[1])
                    );
                }

                int row0 = group;
                int row1 = row0 + 8;

                int col0 = (lane % 4) * 2;
                int col1 = col0 + 1;

                int out_row0 = row_start + row0;
                int out_row1 = row_start + row1;

                int out_col0 = col_start + col0;
                int out_col1 = col_start + col1;

                if (out_row0 < Br && out_col0 < HEAD_DIM) {
                    o_acc[out_row0 * HEAD_DIM + out_col0] += d[0];
                }

                if (out_row0 < Br && out_col1 < HEAD_DIM) {
                    o_acc[out_row0 * HEAD_DIM + out_col1] += d[1];
                }

                if (out_row1 < Br && out_col0 < HEAD_DIM) {
                    o_acc[out_row1 * HEAD_DIM + out_col0] += d[2];
                }

                if (out_row1 < Br && out_col1 < HEAD_DIM) {
                    o_acc[out_row1 * HEAD_DIM + out_col1] += d[3];
                }
            }
        }

        __syncthreads();
    }

    for (int i = tid; i < Br * HEAD_DIM; i += 128) {
        int row = i / HEAD_DIM;
        int col = i % HEAD_DIM;

        int global_row = Qrowitr * Br + row;

        if (global_row < SEQLEN && col < HEAD_DIM) {
            float denom = l_shared[row];

            float out_val = 0.0f;
            if (denom > 0.0f) {
                out_val = o_acc[i] / denom;
            }

            Optr[global_row * HEAD_DIM + col] =
                __float2half(out_val);
        }
    }

    for (int row = tid; row < Br; row += 128) {
        int global_row = Qrowitr * Br + row;

        if (global_row < SEQLEN) {
            Lptr[global_row] =
                m_shared[row] + logf(l_shared[row]);
        }
    }
}
