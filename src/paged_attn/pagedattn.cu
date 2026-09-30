

#include "helper.cuh"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <vector>

namespace PAGEDATTN {

template<int Rows, int Cols, int NT>
__device__ __forceinline__ void asyncLOAD_PAGED_TILE(
    const __half* __restrict__ pool,
    uint32_t                   smemptr,
    int                        tid,
    int                        smem_stride,
    const int* __restrict__    table,
    int                        kv_start,
    int                        kv_len,
    int                        actual_D,
    int                        blk_shift,
    int                        max_blocks,
    long long                  token_stride,
    long long                  head_off
) {
    static_assert(Cols % 8 == 0, "Cols must be divisible by 8");
    constexpr int vecs_per_row = Cols / 8;
    constexpr int total_vecs   = Rows * vecs_per_row;
    const int blk_mask = (1 << blk_shift) - 1;

    for (int i = tid; i < total_vecs; i += NT) {
        const int r    = i / vecs_per_row;
        const int c    = (i % vecs_per_row) * 8;
        const int kpos = kv_start + r;

        bool valid = (kpos < kv_len) && (c + 8 <= actual_D);
        const __half* src = pool;

        if (valid) {
            const int page = kpos >> blk_shift;
            const int phys = (page < max_blocks) ? __ldg(table + page) : -1;
            valid = (phys >= 0);
            if (valid) {
                src = pool
                    + ((long long)phys * (blk_mask + 1) + (kpos & blk_mask)) * token_stride
                    + head_off + c;
            }
        }

        const uint32_t dst = smemptr + (r * smem_stride + c) * sizeof(__half);
        const int pred = valid ? 1 : 0;

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
            : "r"(dst), "l"(src), "r"(pred)
            : "memory"
        );
    }
}

__device__ __forceinline__ void atomic_add_paged_half2(
    __half* __restrict__ pool,
    const int* __restrict__ table,
    int  kv_pos,
    int  kv_len,
    int  d,
    int  actual_D,
    int  blk_shift,
    int  max_blocks,
    int  blocksize,
    long long token_stride,
    long long head_off,
    float c0, float c1
) {
    if (kv_pos >= kv_len) return;
    if (d >= actual_D) return;
    const int page = kv_pos >> blk_shift;
    if (page >= max_blocks) return;
    const int phys = __ldg(table + page);
    if (phys < 0) return;
    const int off = kv_pos & (blocksize - 1);
    __half* addr = pool + ((long long)phys * blocksize + off) * token_stride
                 + head_off + d;
    atomicAdd(reinterpret_cast<__half2*>(addr), __floats2half2_rn(c0, c1));
}

// ---------------------------------------------------------------------------
// ldmatrix address helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t ldm_x4_fwd_addr(
    const __half* smem, int stride, int row_base, int col_base, int lane) {
    const int lane16 = lane & 15;
    const int r = row_base + lane16;
    const int c = col_base + ((lane < 16) ? 0 : 8);
    return smem_u32_ptr(smem + r * stride + c);
}

__device__ __forceinline__ uint32_t ldm_x4_trans_addr(
    const __half* smem, int stride, int row_base, int col_base, int lane) {
    int r, c;
    if (lane < 8)       { r = row_base + lane;              c = col_base; }
    else if (lane < 16) { r = row_base + (lane - 8);        c = col_base + 8; }
    else if (lane < 24) { r = row_base + (lane - 16) + 8;   c = col_base; }
    else                { r = row_base + (lane - 24) + 8;   c = col_base + 8; }
    return smem_u32_ptr(smem + r * stride + c);
}

__device__ __forceinline__ uint32_t ldm_x2_trans_addr(
    const __half* smem, int stride, int row_base, int col_base, int lane) {
    const int lane16 = lane & 15;
    return smem_u32_ptr(smem + (row_base + lane16) * stride + col_base);
}

__device__ __forceinline__ uint32_t ldm_x2_addr(
    const __half* smem, int stride, int row_base, int col_base, int lane) {
    const int lane16 = lane & 15;
    const int r = row_base + (lane16 & 7);
    const int c = col_base + ((lane16 >> 3) * 8);
    return smem_u32_ptr(smem + r * stride + c);
}


template<int Br, int Bc, int D_PAD, bool masked, bool cross>
__global__ void forward(
    __half*       __restrict__ O,
    float*        __restrict__ Lout,
    const __half* __restrict__ Q,
    const __half* __restrict__ Kpool,
    const __half* __restrict__ Vpool,
    const int blocksize,
    const int* __restrict__ seq_lens,
    const int* __restrict__ blocktable,
    const int* __restrict__ queryloc,
    const float scale,
    const int actual_D,
    const int max_blocks_per_seq,
    const int Hkv
) {
    static_assert(Br == 64, "This kernel requires Br == 64 (4 warps x 16 rows)");
    static_assert(Bc > 0 && Bc % 16 == 0, "Bc must be a positive multiple of 16");
    static_assert(D_PAD > 0 && D_PAD % 16 == 0, "D_PAD must be a multiple of 16");
    static_assert(!(masked && cross), "cross-attention is never causal");

    constexpr bool causal = masked && !cross;

    if (blockDim.x != 128) return;
    if (actual_D <= 0 || actual_D > D_PAD || (actual_D & 7) != 0) return;
    if (blocksize <= 0 || (blocksize & (blocksize - 1)) != 0) return;

    const int tid   = threadIdx.x;
    const int warp  = tid >> 5;
    const int lane  = tid & 31;
    const int lane4 = lane & 3;

    const int seqid  = blockIdx.x;
    const int q_head = blockIdx.y;
    const int tileid = blockIdx.z;
    const int Hq     = gridDim.y;

    if (Hkv <= 0 || (Hq % Hkv) != 0) return;
    const int kv_head = q_head / (Hq / Hkv);

    const int blk_shift = __ffs(blocksize) - 1;

    const int q_begin = queryloc[seqid];
    const int q_len   = queryloc[seqid + 1] - q_begin;

    int kv_len = seq_lens[seqid];
    const int kv_cap = max_blocks_per_seq * blocksize;
    if (kv_len > kv_cap) kv_len = kv_cap;

    const int q_tile_start = tileid * Br;
    if (q_tile_start >= q_len) return;

    const int* table = blocktable + (size_t)seqid * max_blocks_per_seq;

    int ctx_len = 0;
    if constexpr (causal) {
        ctx_len = kv_len - q_len;
        if (ctx_len < 0) ctx_len = 0;
    }

    int kv_limit = kv_len;
    if constexpr (causal) {
        const int q_last = (q_tile_start + Br < q_len ? q_tile_start + Br : q_len) - 1;
        const int vis    = ctx_len + q_last + 1;
        if (vis < kv_limit) kv_limit = vis;
    }
    const int kv_tiles = kv_limit > 0 ? (kv_limit + Bc - 1) / Bc : 0;

    const long long token_stride = (long long)Hkv * actual_D;
    const long long head_off     = (long long)kv_head * actual_D;

    const __half* Qptr = Q + ((size_t)q_begin * Hq + q_head) * actual_D;
          __half* Optr = O + ((size_t)q_begin * Hq + q_head) * actual_D;
    const int     q_row_stride = Hq * actual_D;

    constexpr int PAD = 8;
    constexpr int Q_STRIDE = D_PAD + PAD;
    constexpr int K_STRIDE = D_PAD + PAD;
    constexpr int V_STRIDE = D_PAD + PAD;

    extern __shared__ __align__(16) char smem_raw[];
    __half* Qsmem  = reinterpret_cast<__half*>(smem_raw);
    __half* Ksmem0 = Qsmem  + Br * Q_STRIDE;
    __half* Ksmem1 = Ksmem0 + Bc * K_STRIDE;
    __half* Vsmem0 = Ksmem1 + Bc * K_STRIDE;
    __half* Vsmem1 = Vsmem0 + Bc * V_STRIDE;
    __half* Ksmem[2] = {Ksmem0, Ksmem1};
    __half* Vsmem[2] = {Vsmem0, Vsmem1};

    constexpr int Dk = D_PAD / 16;
    constexpr int Bk = Bc / 8;
    constexpr int Dv = D_PAD / 8;

    float O_frag[Dv * 4] = {0.0f};
    float m_frag[2] = {-INFINITY, -INFINITY};
    float l_frag[2] = {0.0f, 0.0f};

    asyncLOAD_2D_TILE<Br, D_PAD, 128>(
        Qptr, smem_u32_ptr(Qsmem), tid, Q_STRIDE,
        q_len, actual_D, q_row_stride, tileid, 0);

    if (kv_tiles > 0) {
        asyncLOAD_PAGED_TILE<Bc, D_PAD, 128>(
            Kpool, smem_u32_ptr(Ksmem[0]), tid, K_STRIDE, table,
            0, kv_len, actual_D, blk_shift, max_blocks_per_seq, token_stride, head_off);
        asyncLOAD_PAGED_TILE<Bc, D_PAD, 128>(
            Vpool, smem_u32_ptr(Vsmem[0]), tid, V_STRIDE, table,
            0, kv_len, actual_D, blk_shift, max_blocks_per_seq, token_stride, head_off);
    }
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    const int lane16 = lane & 15;
    const int qpos0 = ctx_len + q_tile_start + warp * 16 + (lane >> 2);
    const int qpos1 = qpos0 + 8;

    for (int kv_tile = 0; kv_tile < kv_tiles; ++kv_tile) {
        const int cur  = kv_tile & 1;
        const int nxt  = cur ^ 1;
        const int next_tile = kv_tile + 1;

        if (next_tile < kv_tiles) {
            asyncLOAD_PAGED_TILE<Bc, D_PAD, 128>(
                Kpool, smem_u32_ptr(Ksmem[nxt]), tid, K_STRIDE, table,
                next_tile * Bc, kv_len, actual_D, blk_shift, max_blocks_per_seq,
                token_stride, head_off);
            asyncLOAD_PAGED_TILE<Bc, D_PAD, 128>(
                Vpool, smem_u32_ptr(Vsmem[nxt]), tid, V_STRIDE, table,
                next_tile * Bc, kv_len, actual_D, blk_shift, max_blocks_per_seq,
                token_stride, head_off);
            asm volatile("cp.async.commit_group;\n");
        }

        float S_frag[Bk * 4] = {0.0f};

        #pragma unroll
        for (int ks = 0; ks < Dk; ++ks) {
            uint32_t q_frag[4];
            ldmatrix_x4(q_frag, ldm_x4_fwd_addr(Qsmem, Q_STRIDE, warp*16, ks*16, lane));

            #pragma unroll
            for (int kb = 0; kb < Bk; ++kb) {
                uint32_t k_frag[2];
                ldmatrix_x2(k_frag, ldm_x2_addr(Ksmem[cur], K_STRIDE, kb*8, ks*16, lane));

                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(S_frag[kb*4+0]), "+f"(S_frag[kb*4+1]),
                      "+f"(S_frag[kb*4+2]), "+f"(S_frag[kb*4+3])
                    : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]),
                      "r"(k_frag[0]), "r"(k_frag[1]));
            }
        }

        const int kv_start  = kv_tile * Bc;
        const bool tile_full = (kv_start + Bc) <= kv_len;
        const int kv_last    = (kv_start + Bc < kv_len ? kv_start + Bc : kv_len) - 1;

        bool need_causal = false;
        if constexpr (causal) need_causal = kv_last > (ctx_len + q_tile_start);
        const bool needs_mask = !tile_full || need_causal;

        float tile_max[2] = {-INFINITY, -INFINITY};

        #pragma unroll
        for (int kb = 0; kb < Bk; ++kb) {
            float s0 = S_frag[kb*4+0] * scale;
            float s1 = S_frag[kb*4+1] * scale;
            float s2 = S_frag[kb*4+2] * scale;
            float s3 = S_frag[kb*4+3] * scale;

            if (needs_mask) {
                const int key0 = kv_start + kb * 8 + lane4 * 2;
                const int key1 = key0 + 1;
                bool v00 = key0 < kv_len;
                bool v01 = key1 < kv_len;
                bool v10 = v00, v11 = v01;
                if constexpr (causal) {
                    v00 = v00 && key0 <= qpos0;  v01 = v01 && key1 <= qpos0;
                    v10 = v10 && key0 <= qpos1;  v11 = v11 && key1 <= qpos1;
                }
                if (!v00) s0 = -INFINITY;
                if (!v01) s1 = -INFINITY;
                if (!v10) s2 = -INFINITY;
                if (!v11) s3 = -INFINITY;
            }

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

        const float new_max0 = fmaxf(m_frag[0], tile_max[0]);
        const float new_max1 = fmaxf(m_frag[1], tile_max[1]);
        const float ref0 = (new_max0 == -INFINITY) ? 0.0f : new_max0;
        const float ref1 = (new_max1 == -INFINITY) ? 0.0f : new_max1;

        const float alpha0 = __expf(m_frag[0] - ref0);
        const float alpha1 = __expf(m_frag[1] - ref1);

        #pragma unroll
        for (int vs = 0; vs < Dv; ++vs) {
            O_frag[vs*4+0] *= alpha0;
            O_frag[vs*4+1] *= alpha0;
            O_frag[vs*4+2] *= alpha1;
            O_frag[vs*4+3] *= alpha1;
        }

        m_frag[0] = new_max0; m_frag[1] = new_max1;
        float tile_sum0 = 0.0f, tile_sum1 = 0.0f;

        #pragma unroll
        for (int kb = 0; kb < Bk; kb += 2) {
            const float p0 = __expf(S_frag[kb*4+0] - ref0);
            const float p1 = __expf(S_frag[kb*4+1] - ref0);
            const float p2 = __expf(S_frag[kb*4+2] - ref1);
            const float p3 = __expf(S_frag[kb*4+3] - ref1);
            const float p4 = __expf(S_frag[(kb+1)*4+0] - ref0);
            const float p5 = __expf(S_frag[(kb+1)*4+1] - ref0);
            const float p6 = __expf(S_frag[(kb+1)*4+2] - ref1);
            const float p7 = __expf(S_frag[(kb+1)*4+3] - ref1);

            tile_sum0 += p0 + p1 + p4 + p5;
            tile_sum1 += p2 + p3 + p6 + p7;

            uint32_t p_frag[4];
            p_frag[0] = pack_float2_to_half2_u32(p0, p1);
            p_frag[1] = pack_float2_to_half2_u32(p2, p3);
            p_frag[2] = pack_float2_to_half2_u32(p4, p5);
            p_frag[3] = pack_float2_to_half2_u32(p6, p7);

            const int v_row = (kb / 2) * 16 + lane16;

            #pragma unroll
            for (int vs = 0; vs < Dv; ++vs) {
                uint32_t v_frag[2];
                ldmatrix_x2_trans(v_frag,
                    smem_u32_ptr(Vsmem[cur] + v_row * V_STRIDE + vs * 8));

                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(O_frag[vs*4+0]), "+f"(O_frag[vs*4+1]),
                      "+f"(O_frag[vs*4+2]), "+f"(O_frag[vs*4+3])
                    : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]),
                      "r"(v_frag[0]), "r"(v_frag[1]));
            }
        }

        tile_sum0 += __shfl_xor_sync(0xffffffffu, tile_sum0, 1, 4);
        tile_sum0 += __shfl_xor_sync(0xffffffffu, tile_sum0, 2, 4);
        tile_sum1 += __shfl_xor_sync(0xffffffffu, tile_sum1, 1, 4);
        tile_sum1 += __shfl_xor_sync(0xffffffffu, tile_sum1, 2, 4);

        l_frag[0] = l_frag[0] * alpha0 + tile_sum0;
        l_frag[1] = l_frag[1] * alpha1 + tile_sum1;

        if (next_tile < kv_tiles) {
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
            __syncthreads();
        }
    }

    const int row0 = q_tile_start + warp * 16 + (lane >> 2);
    const int row1 = row0 + 8;
    const float inv_l0 = l_frag[0] > 0.0f ? 1.0f / l_frag[0] : 0.0f;
    const float inv_l1 = l_frag[1] > 0.0f ? 1.0f / l_frag[1] : 0.0f;

    #pragma unroll
    for (int vs = 0; vs < Dv; ++vs) {
        const int col = vs * 8 + lane4 * 2;
        if (col >= actual_D) continue;
        if (row0 < q_len) {
            *reinterpret_cast<__half2*>(Optr + (size_t)row0 * q_row_stride + col) =
                __floats2half2_rn(O_frag[vs*4+0] * inv_l0, O_frag[vs*4+1] * inv_l0);
        }
        if (row1 < q_len) {
            *reinterpret_cast<__half2*>(Optr + (size_t)row1 * q_row_stride + col) =
                __floats2half2_rn(O_frag[vs*4+2] * inv_l1, O_frag[vs*4+3] * inv_l1);
        }
    }

    if (lane4 == 0 && Lout != nullptr) {
        if (row0 < q_len) {
            float lv = (l_frag[0] > 0.0f)
                     ? (m_frag[0] + __logf(l_frag[0])) : INFINITY;
            Lout[((size_t)q_begin + row0) * Hq + q_head] = lv;
        }
        if (row1 < q_len) {
            float lv = (l_frag[1] > 0.0f)
                     ? (m_frag[1] + __logf(l_frag[1])) : INFINITY;
            Lout[((size_t)q_begin + row1) * Hq + q_head] = lv;
        }
    }
}

__global__ void paged_attn_bwd_preprocess(
    float*        __restrict__ D_attn,
    const __half* __restrict__ O,
    const __half* __restrict__ dO,
    int D)
{
    const int64_t row = blockIdx.x;
    const __half* Op  = O  + row * D;
    const __half* dOp = dO + row * D;

    float sum = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        sum += __half2float(Op[d]) * __half2float(dOp[d]);
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, off);

    __shared__ float warp_sums[32];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        const int nwarps = (blockDim.x + 31) >> 5;
        sum = (lane < nwarps) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, off);
        if (lane == 0) D_attn[row] = sum;
    }
}

template<int Br, int Bc, int D_PAD, bool masked, bool cross>
__global__ void __launch_bounds__(128) backward(
    __half*       __restrict__ dQ,
    __half*       __restrict__ dK,
    __half*       __restrict__ dV,
    const __half* __restrict__ Kpool_in,
    const __half* __restrict__ Vpool_in,
    const __half* __restrict__ Q,
    const __half* __restrict__ dO,
    const float*  __restrict__ Lp,
    const float*  __restrict__ Dp,
    const int blocksize,
    const int* __restrict__ seq_lens,
    const int* __restrict__ blocktable,
    const int* __restrict__ queryloc,
    const float scale,
    const int actual_D,
    const int max_blocks_per_seq,
    const int Hkv
) {
    static_assert(Br == 64, "requires Br == 64");
    static_assert(Bc > 0 && Bc % 16 == 0, "Bc must be a positive multiple of 16");
    static_assert(D_PAD > 0 && D_PAD % 16 == 0, "D_PAD must be a multiple of 16");
    static_assert(!(masked && cross), "cross-attention is never causal");

    constexpr bool causal = masked && !cross;

    if (blockDim.x != 128) return;
    if (actual_D <= 0 || actual_D > D_PAD || (actual_D & 7) != 0) return;
    if (blocksize <= 0 || (blocksize & (blocksize - 1)) != 0) return;

    const int tid   = threadIdx.x;
    const int warp  = tid >> 5;
    const int lane  = tid & 31;
    const int lane4 = lane & 3;
    const int g     = lane >> 2;

    const int seqid  = blockIdx.x;
    const int q_head = blockIdx.y;
    const int tileid = blockIdx.z;
    const int Hq     = gridDim.y;

    if (Hkv <= 0 || (Hq % Hkv) != 0) return;
    const int kv_head = q_head / (Hq / Hkv);

    const int blk_shift = __ffs(blocksize) - 1;

    const int q_begin = queryloc[seqid];
    const int q_len   = queryloc[seqid + 1] - q_begin;

    int kv_len = seq_lens[seqid];
    const int kv_cap = max_blocks_per_seq * blocksize;
    if (kv_len > kv_cap) kv_len = kv_cap;

    const int q_tile_start = tileid * Br;
    if (q_tile_start >= q_len) return;

    const int* table = blocktable + (size_t)seqid * max_blocks_per_seq;

    int ctx_len = 0;
    if constexpr (causal) {
        ctx_len = kv_len - q_len;
        if (ctx_len < 0) ctx_len = 0;
    }

    int kv_limit = kv_len;
    if constexpr (causal) {
        const int q_last = (q_tile_start + Br < q_len ? q_tile_start + Br : q_len) - 1;
        const int vis    = ctx_len + q_last + 1;
        if (vis < kv_limit) kv_limit = vis;
    }
    const int kv_tiles = kv_limit > 0 ? (kv_limit + Bc - 1) / Bc : 0;

    const long long token_stride = (long long)Hkv * actual_D;
    const long long head_off     = (long long)kv_head * actual_D;

    const __half* Qptr  = Q  + ((size_t)q_begin * Hq + q_head) * actual_D;
    const __half* dOptr = dO + ((size_t)q_begin * Hq + q_head) * actual_D;
    __half*       dQptr = dQ + ((size_t)q_begin * Hq + q_head) * actual_D;
    const int q_row_stride = Hq * actual_D;

    constexpr int PAD = 8;
    constexpr int Q_STRIDE  = D_PAD + PAD;
    constexpr int K_STRIDE  = D_PAD + PAD;
    constexpr int V_STRIDE  = D_PAD + PAD;
    constexpr int Bc_STRIDE = Bc + PAD;

    extern __shared__ __align__(16) char smem_raw_b[];
    __half* Qsmem  = reinterpret_cast<__half*>(smem_raw_b);
    __half* dOsmem = Qsmem  + Br * Q_STRIDE;
    __half* Ksmem  = dOsmem + Br * Q_STRIDE;
    __half* Vsmem  = Ksmem  + Bc * K_STRIDE;
    __half* Psmem  = Vsmem  + Bc * V_STRIDE;
    __half* dSsmem = Psmem  + Br * Bc_STRIDE;

    constexpr int Dk   = D_PAD / 16;
    constexpr int Bk8  = Bc / 8;
    constexpr int Dv   = D_PAD / 8;
    constexpr int Bc16 = Bc / 16;
    const int num_dtiles = actual_D / 8;

    asyncLOAD_2D_TILE<Br, D_PAD, 128>(
        Qptr, smem_u32_ptr(Qsmem), tid, Q_STRIDE,
        q_len, actual_D, q_row_stride, tileid, 0);
    asyncLOAD_2D_TILE<Br, D_PAD, 128>(
        dOptr, smem_u32_ptr(dOsmem), tid, Q_STRIDE,
        q_len, actual_D, q_row_stride, tileid, 0);
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    const int q_row0 = q_tile_start + warp * 16 + g;
    const int q_row1 = q_row0 + 8;
    float L0 = INFINITY, L1 = INFINITY;
    if (q_row0 < q_len) L0 = __ldg(Lp + ((size_t)q_begin + q_row0) * Hq + q_head);
    if (q_row1 < q_len) L1 = __ldg(Lp + ((size_t)q_begin + q_row1) * Hq + q_head);

    float D_row0 = 0.0f, D_row1 = 0.0f;
    if (q_row0 < q_len) D_row0 = __ldg(Dp + ((size_t)q_begin + q_row0) * Hq + q_head);
    if (q_row1 < q_len) D_row1 = __ldg(Dp + ((size_t)q_begin + q_row1) * Hq + q_head);

    const int qpos0 = ctx_len + q_row0;
    const int qpos1 = ctx_len + q_row1;

    float dQ_acc[Dv * 4];
    #pragma unroll
    for (int i = 0; i < Dv*4; ++i) dQ_acc[i] = 0.0f;

    for (int kv_tile = 0; kv_tile < kv_tiles; ++kv_tile) {
        const int kv_start = kv_tile * Bc;

        // Read K/V from the INPUT pool (not from dK/dV).
        asyncLOAD_PAGED_TILE<Bc, D_PAD, 128>(
            Kpool_in, smem_u32_ptr(Ksmem), tid, K_STRIDE, table,
            kv_start, kv_len, actual_D, blk_shift, max_blocks_per_seq,
            token_stride, head_off);
        asyncLOAD_PAGED_TILE<Bc, D_PAD, 128>(
            Vpool_in, smem_u32_ptr(Vsmem), tid, V_STRIDE, table,
            kv_start, kv_len, actual_D, blk_shift, max_blocks_per_seq,
            token_stride, head_off);
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();

        // S = Q K^T
        float S_frag[Bk8 * 4];
        #pragma unroll
        for (int i = 0; i < Bk8*4; ++i) S_frag[i] = 0.0f;

        #pragma unroll
        for (int ks = 0; ks < Dk; ++ks) {
            uint32_t q_frag[4];
            ldmatrix_x4(q_frag, ldm_x4_fwd_addr(Qsmem, Q_STRIDE, warp*16, ks*16, lane));
            #pragma unroll
            for (int kb = 0; kb < Bk8; ++kb) {
                uint32_t k_frag[2];
                ldmatrix_x2(k_frag, ldm_x2_addr(Ksmem, K_STRIDE, kb*8, ks*16, lane));
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(S_frag[kb*4+0]), "+f"(S_frag[kb*4+1]),
                      "+f"(S_frag[kb*4+2]), "+f"(S_frag[kb*4+3])
                    : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]),
                      "r"(k_frag[0]), "r"(k_frag[1]));
            }
        }

        const bool tile_full = (kv_start + Bc) <= kv_len;
        const int  kv_last   = (kv_start + Bc < kv_len ? kv_start + Bc : kv_len) - 1;
        bool need_causal = false;
        if constexpr (causal) need_causal = kv_last > (ctx_len + q_tile_start);
        const bool needs_mask = !tile_full || need_causal;

        // P = exp(S*scale - L) and write to Psmem
        #pragma unroll
        for (int kb = 0; kb < Bk8; ++kb) {
            float s0 = S_frag[kb*4+0] * scale;
            float s1 = S_frag[kb*4+1] * scale;
            float s2 = S_frag[kb*4+2] * scale;
            float s3 = S_frag[kb*4+3] * scale;

            if (needs_mask) {
                const int key0 = kv_start + kb*8 + lane4*2;
                const int key1 = key0 + 1;
                bool v00 = key0 < kv_len;
                bool v01 = key1 < kv_len;
                bool v10 = v00, v11 = v01;
                if constexpr (causal) {
                    v00 = v00 && key0 <= qpos0;  v01 = v01 && key1 <= qpos0;
                    v10 = v10 && key0 <= qpos1;  v11 = v11 && key1 <= qpos1;
                }
                if (!v00) s0 = -INFINITY;
                if (!v01) s1 = -INFINITY;
                if (!v10) s2 = -INFINITY;
                if (!v11) s3 = -INFINITY;
            }

            const float p0 = __expf(s0 - L0);
            const float p1 = __expf(s1 - L0);
            const float p2 = __expf(s2 - L1);
            const float p3 = __expf(s3 - L1);

            S_frag[kb*4+0] = p0; S_frag[kb*4+1] = p1;
            S_frag[kb*4+2] = p2; S_frag[kb*4+3] = p3;

            const int prow0 = warp*16 + g;
            const int prow1 = prow0 + 8;
            const int pcol  = kb*8 + lane4*2;
            *reinterpret_cast<__half2*>(Psmem + prow0*Bc_STRIDE + pcol) =
                __floats2half2_rn(p0, p1);
            *reinterpret_cast<__half2*>(Psmem + prow1*Bc_STRIDE + pcol) =
                __floats2half2_rn(p2, p3);
        }

        // dP = dO V^T
        float dP_frag[Bk8 * 4];
        #pragma unroll
        for (int i = 0; i < Bk8*4; ++i) dP_frag[i] = 0.0f;

        #pragma unroll
        for (int ks = 0; ks < Dk; ++ks) {
            uint32_t do_frag[4];
            ldmatrix_x4(do_frag, ldm_x4_fwd_addr(dOsmem, Q_STRIDE, warp*16, ks*16, lane));
            #pragma unroll
            for (int kb = 0; kb < Bk8; ++kb) {
                uint32_t v_frag[2];
                ldmatrix_x2(v_frag, ldm_x2_addr(Vsmem, V_STRIDE, kb*8, ks*16, lane));
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(dP_frag[kb*4+0]), "+f"(dP_frag[kb*4+1]),
                      "+f"(dP_frag[kb*4+2]), "+f"(dP_frag[kb*4+3])
                    : "r"(do_frag[0]), "r"(do_frag[1]), "r"(do_frag[2]), "r"(do_frag[3]),
                      "r"(v_frag[0]), "r"(v_frag[1]));
            }
        }

        // dS = P * (dP - D) * scale, stored to dSsmem
        #pragma unroll
        for (int kb = 0; kb < Bk8; ++kb) {
            const float ds0 = S_frag[kb*4+0] * (dP_frag[kb*4+0] - D_row0) * scale;
            const float ds1 = S_frag[kb*4+1] * (dP_frag[kb*4+1] - D_row0) * scale;
            const float ds2 = S_frag[kb*4+2] * (dP_frag[kb*4+2] - D_row1) * scale;
            const float ds3 = S_frag[kb*4+3] * (dP_frag[kb*4+3] - D_row1) * scale;

            const int drow0 = warp*16 + g;
            const int drow1 = drow0 + 8;
            const int dcol  = kb*8 + lane4*2;
            *reinterpret_cast<__half2*>(dSsmem + drow0*Bc_STRIDE + dcol) =
                __floats2half2_rn(ds0, ds1);
            *reinterpret_cast<__half2*>(dSsmem + drow1*Bc_STRIDE + dcol) =
                __floats2half2_rn(ds2, ds3);
        }

        __syncthreads();

        // dV += P^T @ dO    (accumulate into dV, NOT Vpool_in)
        #pragma unroll 1
        for (int bc_tile = 0; bc_tile < Bc16; ++bc_tile) {
            const int bc_base = bc_tile * 16;
            for (int d_tile = 0; d_tile < num_dtiles; ++d_tile) {
                const int d_base = d_tile * 8;

                uint32_t a_frag[4], b_frag[2];
                ldmatrix_x4_trans(a_frag,
                    ldm_x4_trans_addr(Psmem, Bc_STRIDE, warp*16, bc_base, lane));
                ldmatrix_x2_trans(b_frag,
                    ldm_x2_trans_addr(dOsmem, Q_STRIDE, warp*16, d_base, lane));

                float dv[4] = {0.f, 0.f, 0.f, 0.f};
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(dv[0]), "+f"(dv[1]), "+f"(dv[2]), "+f"(dv[3])
                    : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                      "r"(b_frag[0]), "r"(b_frag[1]));

                const int d_col = d_base + lane4 * 2;
                const int kv0 = kv_start + bc_base + g;
                const int kv1 = kv_start + bc_base + g + 8;

                atomic_add_paged_half2(dV, table, kv0, kv_len, d_col,
                    actual_D, blk_shift, max_blocks_per_seq, blocksize,
                    token_stride, head_off, dv[0], dv[1]);
                atomic_add_paged_half2(dV, table, kv1, kv_len, d_col,
                    actual_D, blk_shift, max_blocks_per_seq, blocksize,
                    token_stride, head_off, dv[2], dv[3]);
            }
        }

        // dK += dS^T @ Q    (accumulate into dK, NOT Kpool_in)
        #pragma unroll 1
        for (int bc_tile = 0; bc_tile < Bc16; ++bc_tile) {
            const int bc_base = bc_tile * 16;
            for (int d_tile = 0; d_tile < num_dtiles; ++d_tile) {
                const int d_base = d_tile * 8;

                uint32_t a_frag[4], b_frag[2];
                ldmatrix_x4_trans(a_frag,
                    ldm_x4_trans_addr(dSsmem, Bc_STRIDE, warp*16, bc_base, lane));
                ldmatrix_x2_trans(b_frag,
                    ldm_x2_trans_addr(Qsmem, Q_STRIDE, warp*16, d_base, lane));

                float dk[4] = {0.f, 0.f, 0.f, 0.f};
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(dk[0]), "+f"(dk[1]), "+f"(dk[2]), "+f"(dk[3])
                    : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                      "r"(b_frag[0]), "r"(b_frag[1]));

                const int d_col = d_base + lane4 * 2;
                const int kv0 = kv_start + bc_base + g;
                const int kv1 = kv_start + bc_base + g + 8;

                atomic_add_paged_half2(dK, table, kv0, kv_len, d_col,
                    actual_D, blk_shift, max_blocks_per_seq, blocksize,
                    token_stride, head_off, dk[0], dk[1]);
                atomic_add_paged_half2(dK, table, kv1, kv_len, d_col,
                    actual_D, blk_shift, max_blocks_per_seq, blocksize,
                    token_stride, head_off, dk[2], dk[3]);
            }
        }

        // dQ += dS @ K
        #pragma unroll 1
        for (int bc_tile = 0; bc_tile < Bc16; ++bc_tile) {
            const int bc_base = bc_tile * 16;

            uint32_t a_frag[4];
            {
                const int lane16l = lane & 15;
                const int r = warp * 16 + lane16l;
                const int c = bc_base + ((lane < 16) ? 0 : 8);
                ldmatrix_x4(a_frag, smem_u32_ptr(dSsmem + r * Bc_STRIDE + c));
            }

            #pragma unroll
            for (int d_tile = 0; d_tile < Dv; ++d_tile) {
                uint32_t b_frag[2];
                ldmatrix_x2_trans(b_frag,
                    ldm_x2_trans_addr(Ksmem, K_STRIDE, bc_base, d_tile*8, lane));

                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(dQ_acc[d_tile*4+0]), "+f"(dQ_acc[d_tile*4+1]),
                      "+f"(dQ_acc[d_tile*4+2]), "+f"(dQ_acc[d_tile*4+3])
                    : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                      "r"(b_frag[0]), "r"(b_frag[1]));
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int d_tile = 0; d_tile < Dv; ++d_tile) {
        const int col = d_tile * 8 + lane4 * 2;
        if (col >= actual_D) continue;
        if (q_row0 < q_len) {
            *reinterpret_cast<__half2*>(dQptr + (size_t)q_row0 * q_row_stride + col)
                = __floats2half2_rn(dQ_acc[d_tile*4+0], dQ_acc[d_tile*4+1]);
        }
        if (q_row1 < q_len) {
            *reinterpret_cast<__half2*>(dQptr + (size_t)q_row1 * q_row_stride + col)
                = __floats2half2_rn(dQ_acc[d_tile*4+2], dQ_acc[d_tile*4+3]);
        }
    }
}

} // namespace PAGEDATTN


namespace PAGEDATTN {

static int select_d_pad_paged(int d) {
    static const int pads[] = {32, 64, 80, 96, 128, 160, 192, 224, 256};
    for (int p : pads) if (d <= p) return p;
    TORCH_CHECK(false, "head dim > 256 not supported");
    return 0;
}

static void check_paged_inputs(
    const torch::Tensor& Q, const torch::Tensor& Kpool, const torch::Tensor& Vpool,
    const torch::Tensor& blocktable, const torch::Tensor& seq_lens,
    const torch::Tensor& queryloc)
{
    TORCH_CHECK(Q.is_cuda() && Kpool.is_cuda() && Vpool.is_cuda() &&
                blocktable.is_cuda() && seq_lens.is_cuda() && queryloc.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat16 &&
                Kpool.scalar_type() == torch::kFloat16 &&
                Vpool.scalar_type() == torch::kFloat16, "Q/Kpool/Vpool must be float16");
    TORCH_CHECK(blocktable.scalar_type() == torch::kInt32 &&
                seq_lens.scalar_type() == torch::kInt32 &&
                queryloc.scalar_type() == torch::kInt32, "metadata must be int32");
    TORCH_CHECK(Q.is_contiguous() && Kpool.is_contiguous() && Vpool.is_contiguous() &&
                blocktable.is_contiguous() && seq_lens.is_contiguous() &&
                queryloc.is_contiguous(), "all tensors must be contiguous");
    TORCH_CHECK(Q.dim() == 3, "Q must be [total_q_tokens, Hq, D]");
    TORCH_CHECK(Kpool.dim() == 4 && Kpool.sizes() == Vpool.sizes(),
                "Kpool/Vpool must be [num_blocks, blocksize, Hkv, D] and equal shape");
    TORCH_CHECK(Kpool.size(3) == Q.size(2), "head dim mismatch");
    TORCH_CHECK(Q.size(1) % Kpool.size(2) == 0, "Hq must be a multiple of Hkv");
    TORCH_CHECK(Q.size(2) % 8 == 0, "head dim must be a multiple of 8");
    const int64_t bs = Kpool.size(1);
    TORCH_CHECK(bs > 0 && (bs & (bs - 1)) == 0, "blocksize must be a power of two");
    TORCH_CHECK(blocktable.dim() == 2 && blocktable.size(0) == seq_lens.size(0),
                "blocktable must be [num_seqs, max_blocks_per_seq]");
    TORCH_CHECK(queryloc.numel() == seq_lens.numel() + 1, "queryloc must be [num_seqs + 1]");
}

template<int D_PAD, int Bc, bool Masked, bool Cross>
static void launch_fwd_impl(
    torch::Tensor& O, torch::Tensor& Lse,
    const torch::Tensor& Q, const torch::Tensor& Kpool, const torch::Tensor& Vpool,
    const torch::Tensor& blocktable, const torch::Tensor& seq_lens,
    const torch::Tensor& queryloc, int64_t max_q_len, double scale_in
) {
    constexpr int Br = 64;
    constexpr int D_STRIDE = D_PAD + 8;
    constexpr size_t smem_bytes = (size_t)(Br + 4 * Bc) * D_STRIDE * sizeof(__half);

    const auto* props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(smem_bytes <= props->sharedMemPerBlockOptin,
                "not enough shared memory for this head dim on this GPU");

    const int num_seqs = static_cast<int>(seq_lens.size(0));
    const int Hq       = static_cast<int>(Q.size(1));
    const int D        = static_cast<int>(Q.size(2));
    const int Hkv      = static_cast<int>(Kpool.size(2));
    const int bs       = static_cast<int>(Kpool.size(1));
    const int max_blk  = static_cast<int>(blocktable.size(1));

    if (num_seqs == 0 || max_q_len <= 0 || Q.size(0) == 0) return;

    dim3 block(128);
    dim3 grid(num_seqs, Hq, (unsigned)((max_q_len + Br - 1) / Br));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(Q.get_device());

    C10_CUDA_CHECK(cudaFuncSetAttribute(
        forward<Br, Bc, D_PAD, Masked, Cross>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));

    const float scale = scale_in > 0.0 ? static_cast<float>(scale_in)
                                       : 1.0f / sqrtf(static_cast<float>(D));

    forward<Br, Bc, D_PAD, Masked, Cross><<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<__half*>(O.data_ptr<at::Half>()),
        Lse.numel() > 0 ? Lse.data_ptr<float>() : nullptr,
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(Kpool.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(Vpool.data_ptr<at::Half>()),
        bs,
        seq_lens.data_ptr<int>(),
        blocktable.data_ptr<int>(),
        queryloc.data_ptr<int>(),
        scale, D, max_blk, Hkv);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int D_PAD, int Bc, bool Masked, bool Cross>
static void launch_bwd_impl(
    torch::Tensor& dQ, torch::Tensor& dK, torch::Tensor& dV,
    const torch::Tensor& Q,
    const torch::Tensor& Kpool, const torch::Tensor& Vpool,
    const torch::Tensor& dO, const torch::Tensor& Lse, const torch::Tensor& D_attn,
    const torch::Tensor& blocktable, const torch::Tensor& seq_lens,
    const torch::Tensor& queryloc, int64_t max_q_len, double scale_in
) {
    constexpr int Br = 64;
    constexpr int D_STRIDE = D_PAD + 8;
    constexpr int Bc_STRIDE = Bc + 8;
    constexpr size_t smem_bytes =
        (size_t)(2 * Br * D_STRIDE + 2 * Bc * D_STRIDE + 2 * Br * Bc_STRIDE)
        * sizeof(__half);

    const auto* props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(smem_bytes <= props->sharedMemPerBlockOptin,
                "not enough shared memory for backward on this GPU");

    const int num_seqs = static_cast<int>(seq_lens.size(0));
    const int Hq       = static_cast<int>(Q.size(1));
    const int D        = static_cast<int>(Q.size(2));
    const int Hkv      = static_cast<int>(Kpool.size(2));
    const int bs       = static_cast<int>(Kpool.size(1));
    const int max_blk  = static_cast<int>(blocktable.size(1));

    if (num_seqs == 0 || max_q_len <= 0 || Q.size(0) == 0) return;

    dim3 block(128);
    dim3 grid(num_seqs, Hq, (unsigned)((max_q_len + Br - 1) / Br));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(Q.get_device());

    C10_CUDA_CHECK(cudaFuncSetAttribute(
        backward<Br, Bc, D_PAD, Masked, Cross>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));

    const float scale = scale_in > 0.0 ? static_cast<float>(scale_in)
                                       : 1.0f / sqrtf(static_cast<float>(D));

    backward<Br, Bc, D_PAD, Masked, Cross><<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<__half*>(dQ.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(dK.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(dV.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(Kpool.data_ptr<at::Half>()),   // READ
        reinterpret_cast<const __half*>(Vpool.data_ptr<at::Half>()),   // READ
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(dO.data_ptr<at::Half>()),
        Lse.data_ptr<float>(),
        D_attn.data_ptr<float>(),
        bs,
        seq_lens.data_ptr<int>(),
        blocktable.data_ptr<int>(),
        queryloc.data_ptr<int>(),
        scale, D, max_blk, Hkv);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<bool Masked, bool Cross>
static void dispatch_fwd(
    int d_pad,
    torch::Tensor& O, torch::Tensor& Lse,
    const torch::Tensor& Q, const torch::Tensor& Kpool, const torch::Tensor& Vpool,
    const torch::Tensor& blocktable, const torch::Tensor& seq_lens,
    const torch::Tensor& queryloc, int64_t max_q_len, double scale
) {
#define PA_FWD(DP, BC) \
    case DP: launch_fwd_impl<DP, BC, Masked, Cross>(O, Lse, Q, Kpool, Vpool, \
        blocktable, seq_lens, queryloc, max_q_len, scale); break;
    switch (d_pad) {
        PA_FWD(32, 64)  PA_FWD(64, 64)  PA_FWD(80, 64)  PA_FWD(96, 64)
        PA_FWD(128, 64) PA_FWD(160, 32) PA_FWD(192, 32) PA_FWD(224, 32)
        PA_FWD(256, 16)
    }
#undef PA_FWD
}

template<bool Masked, bool Cross>
static void dispatch_bwd(
    int d_pad,
    torch::Tensor& dQ, torch::Tensor& dK, torch::Tensor& dV,
    const torch::Tensor& Q,
    const torch::Tensor& Kpool, const torch::Tensor& Vpool,
    const torch::Tensor& dO, const torch::Tensor& Lse, const torch::Tensor& D_attn,
    const torch::Tensor& blocktable, const torch::Tensor& seq_lens,
    const torch::Tensor& queryloc, int64_t max_q_len, double scale
) {
#define PA_BWD(DP, BC) \
    case DP: launch_bwd_impl<DP, BC, Masked, Cross>(dQ, dK, dV, Q, Kpool, Vpool, \
        dO, Lse, D_attn, blocktable, seq_lens, queryloc, max_q_len, scale); break;
    switch (d_pad) {
        PA_BWD(32, 64)  PA_BWD(64, 64)  PA_BWD(80, 64)  PA_BWD(96, 64)
        PA_BWD(128, 64) PA_BWD(160, 32) PA_BWD(192, 32) PA_BWD(224, 32)
        PA_BWD(256, 16)
    }
#undef PA_BWD
}

} // namespace PAGEDATTN



std::vector<torch::Tensor> paged_attn_fwd_cuda(
    const torch::Tensor& Q, const torch::Tensor& Kpool, const torch::Tensor& Vpool,
    const torch::Tensor& blocktable, const torch::Tensor& seq_lens,
    const torch::Tensor& queryloc, int64_t max_q_len, double scale, bool causal,
    bool return_lse
) {
    using namespace PAGEDATTN;
    check_paged_inputs(Q, Kpool, Vpool, blocktable, seq_lens, queryloc);
    c10::cuda::CUDAGuard device_guard(Q.device());

    auto O = torch::empty_like(Q);
    torch::Tensor Lse;
    if (return_lse) {
        Lse = torch::empty({Q.size(0), Q.size(1)},
                           Q.options().dtype(torch::kFloat32));
    } else {
        Lse = torch::empty({0}, Q.options().dtype(torch::kFloat32));
    }

    const int d_pad = select_d_pad_paged(static_cast<int>(Q.size(2)));
    if (causal) dispatch_fwd<true,  false>(d_pad, O, Lse, Q, Kpool, Vpool,
                                           blocktable, seq_lens, queryloc,
                                           max_q_len, scale);
    else        dispatch_fwd<false, false>(d_pad, O, Lse, Q, Kpool, Vpool,
                                           blocktable, seq_lens, queryloc,
                                           max_q_len, scale);
    return {O, Lse};
}

std::vector<torch::Tensor> paged_attn_bwd_cuda(
    const torch::Tensor& Q, const torch::Tensor& Kpool, const torch::Tensor& Vpool,
    const torch::Tensor& blocktable, const torch::Tensor& seq_lens,
    const torch::Tensor& queryloc,
    const torch::Tensor& dO, const torch::Tensor& Lse,
    const torch::Tensor& O,
    int64_t max_q_len, double scale, bool causal
) {
    using namespace PAGEDATTN;
    check_paged_inputs(Q, Kpool, Vpool, blocktable, seq_lens, queryloc);
    TORCH_CHECK(dO.sizes() == Q.sizes() && dO.scalar_type() == Q.scalar_type() &&
                dO.is_contiguous(), "dO must match Q");
    TORCH_CHECK(O.sizes() == Q.sizes() && O.scalar_type() == Q.scalar_type() &&
                O.is_contiguous(), "O must match Q");
    TORCH_CHECK(Lse.scalar_type() == torch::kFloat32 &&
                Lse.numel() == Q.size(0) * Q.size(1) && Lse.is_contiguous(),
                "Lse must be float32 [total_q_tokens, Hq]");
    c10::cuda::CUDAGuard device_guard(Q.device());

    auto dQ  = torch::zeros_like(Q);
    auto dK  = torch::zeros_like(Kpool);
    auto dV  = torch::zeros_like(Vpool);

    // Preprocess D_attn[token,head] = sum_d O * dO
    auto D_attn = torch::empty({Q.size(0), Q.size(1)},
                               Q.options().dtype(torch::kFloat32));
    if (Q.size(0) > 0) {
        const int D = static_cast<int>(Q.size(2));
        const int64_t rows = Q.size(0) * Q.size(1);
        const int threads = 128;
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(Q.get_device());
        paged_attn_bwd_preprocess<<<(unsigned)rows, threads, 0, stream>>>(
            D_attn.data_ptr<float>(),
            reinterpret_cast<const __half*>(O.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(dO.data_ptr<at::Half>()),
            D);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    const int d_pad = select_d_pad_paged(static_cast<int>(Q.size(2)));
    if (causal) dispatch_bwd<true,  false>(d_pad, dQ, dK, dV, Q, Kpool, Vpool,
                                           dO, Lse, D_attn, blocktable, seq_lens,
                                           queryloc, max_q_len, scale);
    else        dispatch_bwd<false, false>(d_pad, dQ, dK, dV, Q, Kpool, Vpool,
                                           dO, Lse, D_attn, blocktable, seq_lens,
                                           queryloc, max_q_len, scale);
    return {dQ, dK, dV};
}

torch::Tensor paged_cross_attn_fwd_cuda(
    const torch::Tensor& Q, const torch::Tensor& Kpool, const torch::Tensor& Vpool,
    const torch::Tensor& blocktable, const torch::Tensor& enc_lens,
    const torch::Tensor& queryloc, int64_t max_q_len, double scale
) {
    using namespace PAGEDATTN;
    check_paged_inputs(Q, Kpool, Vpool, blocktable, enc_lens, queryloc);
    c10::cuda::CUDAGuard device_guard(Q.device());
    auto O = torch::empty_like(Q);
    torch::Tensor Lse;
    const int d_pad = select_d_pad_paged(static_cast<int>(Q.size(2)));
    dispatch_fwd<false, true>(d_pad, O, Lse, Q, Kpool, Vpool,
                              blocktable, enc_lens, queryloc, max_q_len, scale);
    return O;
}