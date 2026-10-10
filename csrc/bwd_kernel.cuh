#pragma once

#include "mma_helpers.cuh"
 
namespace flashattn_masked_bwd_detail {

template<int M, int N, int ACC_COUNT>
__device__ __forceinline__ void store_f16(
    const float (&accumulator)[ACC_COUNT],
    __half*     __restrict__ output,
    int row_offset,
    int total_rows,
    int actual_cols
) {
    constexpr int kWarps      = 4;
    constexpr int kMTiles     = M / 16;
    constexpr int kNTiles     = N / 8;
    constexpr int kTotalTiles = kMTiles * kNTiles;
    constexpr int kTilesPerWarp =
        (kTotalTiles + kWarps - 1) / kWarps;

    static_assert(ACC_COUNT == kTilesPerWarp * 4, "Wrong accumulator size");

    const int warp      = threadIdx.x >> 5;
    const int lane      = threadIdx.x & 31;
    const int row_group = lane >> 2;
    const int lane4     = lane & 3;

    #pragma unroll
    for (int slot = 0; slot < kTilesPerWarp; ++slot) {
        const int tile = warp + slot * kWarps;
        if (tile < kTotalTiles) {
            const int m_tile = tile / kNTiles;
            const int n_tile = tile % kNTiles;

            const int local_row0  = m_tile * 16 + row_group;
            const int local_row1  = local_row0 + 8;
            const int global_row0 = row_offset + local_row0;
            const int global_row1 = row_offset + local_row1;
            const int col0        = n_tile * 8 + lane4 * 2;
            const int col1        = col0 + 1;

            if (global_row0 < total_rows && col0 < actual_cols)
                output[static_cast<size_t>(global_row0) * actual_cols + col0] =
                    __float2half(accumulator[slot * 4 + 0]);
            if (global_row0 < total_rows && col1 < actual_cols)
                output[static_cast<size_t>(global_row0) * actual_cols + col1] =
                    __float2half(accumulator[slot * 4 + 1]);
            if (global_row1 < total_rows && col0 < actual_cols)
                output[static_cast<size_t>(global_row1) * actual_cols + col0] =
                    __float2half(accumulator[slot * 4 + 2]);
            if (global_row1 < total_rows && col1 < actual_cols)
                output[static_cast<size_t>(global_row1) * actual_cols + col1] =
                    __float2half(accumulator[slot * 4 + 3]);
        }
    }
}

template<int M, int N, int ACC_COUNT>
__device__ __forceinline__ void accumulate_f16(
    const float (&accumulator)[ACC_COUNT],
    __half*     __restrict__ output,
    int row_offset,
    int total_rows,
    int actual_cols
) {
    constexpr int kWarps      = 4;
    constexpr int kMTiles     = M / 16;
    constexpr int kNTiles     = N / 8;
    constexpr int kTotalTiles = kMTiles * kNTiles;
    constexpr int kTilesPerWarp =
        (kTotalTiles + kWarps - 1) / kWarps;

    static_assert(ACC_COUNT == kTilesPerWarp * 4, "Wrong accumulator size");

    const int warp      = threadIdx.x >> 5;
    const int lane      = threadIdx.x & 31;
    const int row_group = lane >> 2;
    const int lane4     = lane & 3;

    #pragma unroll
    for (int slot = 0; slot < kTilesPerWarp; ++slot) {
        const int tile = warp + slot * kWarps;
        if (tile < kTotalTiles) {
            const int m_tile = tile / kNTiles;
            const int n_tile = tile % kNTiles;

            const int local_row0  = m_tile * 16 + row_group;
            const int local_row1  = local_row0 + 8;
            const int global_row0 = row_offset + local_row0;
            const int global_row1 = row_offset + local_row1;
            const int col0        = n_tile * 8 + lane4 * 2;
            const int col1        = col0 + 1;

            if (global_row0 < total_rows && col0 < actual_cols)
                atomicAdd(output + static_cast<size_t>(global_row0) * actual_cols + col0,
                          __float2half(accumulator[slot * 4 + 0]));
            if (global_row0 < total_rows && col1 < actual_cols)
                atomicAdd(output + static_cast<size_t>(global_row0) * actual_cols + col1,
                          __float2half(accumulator[slot * 4 + 1]));
            if (global_row1 < total_rows && col0 < actual_cols)
                atomicAdd(output + static_cast<size_t>(global_row1) * actual_cols + col0,
                          __float2half(accumulator[slot * 4 + 2]));
            if (global_row1 < total_rows && col1 < actual_cols)
                atomicAdd(output + static_cast<size_t>(global_row1) * actual_cols + col1,
                          __float2half(accumulator[slot * 4 + 3]));
        }
    }
}

} // namespace flashattn_masked_bwd_detail

 
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

    const int warp             = threadIdx.x >> 5;
    const int lane             = threadIdx.x & 31;
    const int warps_per_block  = blockDim.x >> 5;
    const int row              = blockIdx.x * warps_per_block + warp;

    if (row < total_q_rows) {
        const __half* o_row  = O  + static_cast<size_t>(row) * actual_D;
        const __half* do_row = dO + static_cast<size_t>(row) * actual_D;

        float sum = 0.0f;
        for (int col = lane; col < actual_D; col += 32)
            sum += __half2float(o_row[col]) * __half2float(do_row[col]);

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);

        if (lane == 0) Delta[row] = sum;
    }
}

 
template<int Br, int Bc, int D_PAD, bool masked>
__global__ void __launch_bounds__(128)
flashattn_bwd_dkdv_kernel(
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
          int Sq,
    const int numKVheads
) {
    static_assert(Br > 0 && Br % 16 == 0, "Br must be divisible by 16");
    static_assert(Bc > 0 && Bc % 16 == 0, "Bc must be divisible by 16");
    static_assert(D_PAD > 0 && D_PAD % 16 == 0, "D_PAD must be divisible by 16");

    if (blockDim.x != 128) return;
    if (actual_D <= 0 || actual_D > D_PAD || Sq <= 0 || Skv <= 0) return;

    const int tid   = threadIdx.x;
    const float scale = 1.0f / sqrtf(static_cast<float>(actual_D));

    constexpr int PAD       = 8;
    constexpr int Q_STRIDE  = D_PAD + PAD;
    constexpr int K_STRIDE  = D_PAD + PAD;
    constexpr int V_STRIDE  = D_PAD + PAD;
    constexpr int DO_STRIDE = D_PAD + PAD;
    constexpr int S_STRIDE  = Bc + PAD;
    constexpr int PDS_STRIDE = Br + PAD;   // transposed: [Bc, Br]

    constexpr int kOutputTiles      = (Bc / 16) * (D_PAD / 8);
    constexpr int kTilesPerWarp     = (kOutputTiles + 3) / 4;
    constexpr int kAccumulatorCount = kTilesPerWarp * 4;

    float dK_fragment[kAccumulatorCount] = {0.0f};
    float dV_fragment[kAccumulatorCount] = {0.0f};

    const int batch = blockIdx.x;
    const int head  = blockIdx.y;
    const int kv_tile = blockIdx.z;
    const int heads = gridDim.y;

    if (numKVheads <= 0 || heads % numKVheads != 0) return;
    const int kv_headid = head / (heads / numKVheads);

    const long long q_base =
        (static_cast<long long>(batch) * heads + head) * Sq * actual_D;
    const long long kv_base =
        (static_cast<long long>(batch) * numKVheads + kv_headid) * Skv * actual_D;
    const long long stat_base =
        (static_cast<long long>(batch) * heads + head) * Sq;

    const __half* Qptr    = Q     + q_base;
    const __half* Kptr    = K     + kv_base;
    const __half* Vptr    = V     + kv_base;
    const __half* dOptr   = dO    + q_base;
    const float*  Lptr    = L     + stat_base;
    const float*  Deltaptr = Delta + stat_base;
          __half* dKptr   = dK    + kv_base;
          __half* dVptr   = dV    + kv_base;

    const int q_tiles  = (Sq  + Br - 1) / Br;
    const int kv_tiles = (Skv + Bc - 1) / Bc;
    if (kv_tile >= kv_tiles) return;

    const int kv_start          = kv_tile * Bc;
    const int kv_last_unclamped = kv_start + Bc - 1;
    const int kv_real_last      = kv_last_unclamped < Skv
                                      ? kv_last_unclamped
                                      : Skv - 1;
    const bool kv_tile_full = kv_last_unclamped < Skv;

    int first_q_tile = 0;
    if constexpr (masked) {
        first_q_tile = kv_start / Br;
        if (first_q_tile > q_tiles) first_q_tile = q_tiles;
    }
 
    extern __shared__ char shared_raw[];
    char* shared_ptr = shared_raw;

    auto align_ptr = [&](size_t alignment = 16) {
        const uintptr_t value = reinterpret_cast<uintptr_t>(shared_ptr);
        shared_ptr = reinterpret_cast<char*>(
            (value + alignment - 1) & ~(alignment - 1));
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
    shared_ptr += Bc * PDS_STRIDE * sizeof(__half);

    align_ptr();
    float* l_smem     = reinterpret_cast<float*>(shared_ptr);
    shared_ptr += Br * sizeof(float);
    align_ptr();
    float* delta_smem = reinterpret_cast<float*>(shared_ptr);

    // prefetch K, V, and first Q/dO tiles
    asyncLOAD_2D_TILE<Bc, D_PAD, 128>(
        Kptr, smem_u32_ptr(smemK), tid, K_STRIDE,
        Skv, actual_D, actual_D, kv_tile, 0);
    asyncLOAD_2D_TILE<Bc, D_PAD, 128>(
        Vptr, smem_u32_ptr(smemV), tid, V_STRIDE,
        Skv, actual_D, actual_D, kv_tile, 0);
    asm volatile("cp.async.commit_group;\n");

    asyncLOAD_2D_TILE<Br, D_PAD, 128>(
        Qptr, smem_u32_ptr(smemQ[0]), tid, Q_STRIDE,
        Sq, actual_D, actual_D, first_q_tile, 0);
    asyncLOAD_2D_TILE<Br, D_PAD, 128>(
        dOptr, smem_u32_ptr(smemdO[0]), tid, DO_STRIDE,
        Sq, actual_D, actual_D, first_q_tile, 0);
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
 
    const int q_steps = q_tiles - first_q_tile;
    for (int step = 0; step < q_steps; ++step) {
        const int qt            = first_q_tile + step;
        const int current_stage = step & 1;
        const int next_stage    = current_stage ^ 1;
        const int next_qt       = qt + 1;

        if (next_qt < q_tiles) {
            asyncLOAD_2D_TILE<Br, D_PAD, 128>(
                Qptr, smem_u32_ptr(smemQ[next_stage]), tid, Q_STRIDE,
                Sq, actual_D, actual_D, next_qt, 0);
            asyncLOAD_2D_TILE<Br, D_PAD, 128>(
                dOptr, smem_u32_ptr(smemdO[next_stage]), tid, DO_STRIDE,
                Sq, actual_D, actual_D, next_qt, 0);
            asm volatile("cp.async.commit_group;\n");
        }

        const int q_start          = qt * Br;
        const int q_last_unclamped = q_start + Br - 1;
        const bool q_tile_full     = q_last_unclamped < Sq;

        bool needs_causal_mask = false;
        if constexpr (masked)
            needs_causal_mask = !(kv_real_last <= q_start);
        const bool needs_any_mask = !q_tile_full || !kv_tile_full || needs_causal_mask;

        for (int row = tid; row < Br; row += blockDim.x) {
            const int gq = q_start + row;
            l_smem[row]     = gq < Sq ? Lptr[gq]     : 0.0f;
            delta_smem[row] = gq < Sq ? Deltaptr[gq] : 0.0f;
        }

        mma_score_f16_tiled<Br, D_PAD, Bc, Q_STRIDE, K_STRIDE, S_STRIDE>(
            smemQ[current_stage], smemK, score_smem);
        __syncthreads();
 
        if (!needs_any_mask) {
            #pragma unroll
            for (int index = tid; index < Br * Bc; index += 128) {
                const int row = index / Bc, col = index % Bc;
                p_ds_smem[col * PDS_STRIDE + row] = __float2half(
                    __expf(score_smem[row * S_STRIDE + col] * scale - l_smem[row]));
            }
        } else {
            #pragma unroll
            for (int index = tid; index < Br * Bc; index += 128) {
                const int row = index / Bc, col = index % Bc;
                const int gq  = q_start + row;
                const int gk  = kv_start + col;
                bool valid    = gq < Sq && gk < Skv;
                if constexpr (masked)
                    if (needs_causal_mask) valid = valid && gk <= gq;
                p_ds_smem[col * PDS_STRIDE + row] = __float2half(
                    valid
                    ? __expf(score_smem[row * S_STRIDE + col] * scale - l_smem[row])
                    : 0.0f);
            }
        }
        __syncthreads();

        // dV += P^T · dO
        mma_accum_f16_registers<Bc, Br, D_PAD, PDS_STRIDE, DO_STRIDE, kAccumulatorCount>(
            p_ds_smem, smemdO[current_stage], dV_fragment);

        // dS = P ∘ (dO·V^T − Delta)
        mma_score_f16_tiled<Br, D_PAD, Bc, DO_STRIDE, V_STRIDE, S_STRIDE>(
            smemdO[current_stage], smemV, score_smem);
        __syncthreads();

        if (!needs_any_mask) {
            #pragma unroll
            for (int index = tid; index < Br * Bc; index += 128) {
                const int row = index / Bc, col = index % Bc;
                const float p  = __half2float(p_ds_smem[col * PDS_STRIDE + row]);
                const float dp = score_smem[row * S_STRIDE + col];
                p_ds_smem[col * PDS_STRIDE + row] =
                    __float2half(p * (dp - delta_smem[row]) * scale);
            }
        } else {
            #pragma unroll
            for (int index = tid; index < Br * Bc; index += 128) {
                const int row = index / Bc, col = index % Bc;
                const int gq  = q_start + row;
                const int gk  = kv_start + col;
                bool valid    = gq < Sq && gk < Skv;
                if constexpr (masked)
                    if (needs_causal_mask) valid = valid && gk <= gq;
                float ds = 0.0f;
                if (valid) {
                    const float p  = __half2float(p_ds_smem[col * PDS_STRIDE + row]);
                    const float dp = score_smem[row * S_STRIDE + col];
                    ds = p * (dp - delta_smem[row]) * scale;
                }
                p_ds_smem[col * PDS_STRIDE + row] = __float2half(ds);
            }
        }
        __syncthreads();

        // dK += dS^T · Q
        mma_accum_f16_registers<Bc, Br, D_PAD, PDS_STRIDE, Q_STRIDE, kAccumulatorCount>(
            p_ds_smem, smemQ[current_stage], dK_fragment);

        if (next_qt < q_tiles) {
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
            __syncthreads();
        }
    }

    flashattn_masked_bwd_detail::store_f16<Bc, D_PAD, kAccumulatorCount>(
        dK_fragment, dKptr, kv_start, Skv, actual_D);
    flashattn_masked_bwd_detail::store_f16<Bc, D_PAD, kAccumulatorCount>(
        dV_fragment, dVptr, kv_start, Skv, actual_D);
}

 
template<int Br, int Bc, int D_PAD, bool masked>
__global__ void __launch_bounds__(128)
flashattn_bwd_dq_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    const __half* __restrict__ dO,
    const float*  __restrict__ L,
    const float*  __restrict__ Delta,
          __half* __restrict__ dQ,
          int actual_D,
          int Skv,
          int Sq,
    const int numKVheads
) {
    static_assert(Br > 0 && Br % 16 == 0, "Br must be divisible by 16");
    static_assert(Bc > 0 && Bc % 16 == 0, "Bc must be divisible by 16");
    static_assert(D_PAD > 0 && D_PAD % 16 == 0, "D_PAD must be divisible by 16");

    if (blockDim.x != 128) return;
    if (actual_D <= 0 || actual_D > D_PAD || Sq <= 0 || Skv <= 0) return;

    const int tid   = threadIdx.x;
    const float scale = 1.0f / sqrtf(static_cast<float>(actual_D));

    constexpr int PAD        = 8;
    constexpr int Q_STRIDE   = D_PAD + PAD;
    constexpr int K_STRIDE   = D_PAD + PAD;
    constexpr int V_STRIDE   = D_PAD + PAD;
    constexpr int DO_STRIDE  = D_PAD + PAD;
    constexpr int S_STRIDE   = Bc + PAD;
    constexpr int PDS_STRIDE = Bc + PAD;

    constexpr int kOutputTiles      = (Br / 16) * (D_PAD / 8);
    constexpr int kTilesPerWarp     = (kOutputTiles + 3) / 4;
    constexpr int kAccumulatorCount = kTilesPerWarp * 4;
    float dQ_fragment[kAccumulatorCount] = {0.0f};

    const int batch  = blockIdx.x;
    const int head   = blockIdx.y;
    const int q_tile = blockIdx.z;
    const int heads  = gridDim.y;

    if (numKVheads <= 0 || heads % numKVheads != 0) return;
    const int kv_headid = head / (heads / numKVheads);

    const long long q_base =
        (static_cast<long long>(batch) * heads + head) * Sq * actual_D;
    const long long kv_base =
        (static_cast<long long>(batch) * numKVheads + kv_headid) * Skv * actual_D;
    const long long stat_base =
        (static_cast<long long>(batch) * heads + head) * Sq;

    const __half* Qptr    = Q   + q_base;
    const __half* Kptr    = K   + kv_base;
    const __half* Vptr    = V   + kv_base;
    const __half* dOptr   = dO  + q_base;
    const float*  Lptr    = L   + stat_base;
    const float*  Deltaptr = Delta + stat_base;
          __half* dQptr   = dQ  + q_base;

    const int q_tiles  = (Sq  + Br - 1) / Br;
    const int kv_tiles = (Skv + Bc - 1) / Bc;
    if (q_tile >= q_tiles) return;

    const int q_start          = q_tile * Br;
    const int q_last_unclamped = q_start + Br - 1;
    const int q_real_last      = q_last_unclamped < Sq ? q_last_unclamped : Sq - 1;
    const bool q_tile_full     = q_last_unclamped < Sq;

    int kv_tiles_to_process = kv_tiles;
    if constexpr (masked) {
        const int causal_tiles = q_real_last / Bc + 1;
        if (causal_tiles < kv_tiles_to_process)
            kv_tiles_to_process = causal_tiles;
    }
 
    extern __shared__ char shared_raw[];
    char* shared_ptr = shared_raw;

    auto align_ptr = [&](size_t alignment = 16) {
        const uintptr_t value = reinterpret_cast<uintptr_t>(shared_ptr);
        shared_ptr = reinterpret_cast<char*>(
            (value + alignment - 1) & ~(alignment - 1));
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
    shared_ptr += Br * PDS_STRIDE * sizeof(__half);

    align_ptr();
    float* l_smem     = reinterpret_cast<float*>(shared_ptr);
    shared_ptr += Br * sizeof(float);
    align_ptr();
    float* delta_smem = reinterpret_cast<float*>(shared_ptr);

    // preload Q and dO (fixed for this block)
    asyncLOAD_2D_TILE<Br, D_PAD, 128>(
        Qptr, smem_u32_ptr(smemQ), tid, Q_STRIDE,
        Sq, actual_D, actual_D, q_tile, 0);
    asyncLOAD_2D_TILE<Br, D_PAD, 128>(
        dOptr, smem_u32_ptr(smemdO), tid, DO_STRIDE,
        Sq, actual_D, actual_D, q_tile, 0);
    asm volatile("cp.async.commit_group;\n");

    // preload scalars
    for (int row = tid; row < Br; row += blockDim.x) {
        const int gq = q_start + row;
        l_smem[row]     = gq < Sq ? Lptr[gq]     : 0.0f;
        delta_smem[row] = gq < Sq ? Deltaptr[gq] : 0.0f;
    }

    // prefetch first KV tile
    asyncLOAD_2D_TILE<Bc, D_PAD, 128>(
        Kptr, smem_u32_ptr(smemK[0]), tid, K_STRIDE,
        Skv, actual_D, actual_D, 0, 0);
    asyncLOAD_2D_TILE<Bc, D_PAD, 128>(
        Vptr, smem_u32_ptr(smemV[0]), tid, V_STRIDE,
        Skv, actual_D, actual_D, 0, 0);
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
 
    for (int kv_tile = 0; kv_tile < kv_tiles_to_process; ++kv_tile) {
        const int current_stage = kv_tile & 1;
        const int next_stage    = current_stage ^ 1;
        const int next_kv_tile  = kv_tile + 1;

        if (next_kv_tile < kv_tiles_to_process) {
            asyncLOAD_2D_TILE<Bc, D_PAD, 128>(
                Kptr, smem_u32_ptr(smemK[next_stage]), tid, K_STRIDE,
                Skv, actual_D, actual_D, next_kv_tile, 0);
            asyncLOAD_2D_TILE<Bc, D_PAD, 128>(
                Vptr, smem_u32_ptr(smemV[next_stage]), tid, V_STRIDE,
                Skv, actual_D, actual_D, next_kv_tile, 0);
            asm volatile("cp.async.commit_group;\n");
        }

        const int kv_start          = kv_tile * Bc;
        const int kv_last_unclamped = kv_start + Bc - 1;
        const int kv_real_last      = kv_last_unclamped < Skv
                                          ? kv_last_unclamped
                                          : Skv - 1;
        const bool kv_tile_full     = kv_last_unclamped < Skv;

        bool needs_causal_mask = false;
        if constexpr (masked)
            needs_causal_mask = !(kv_real_last <= q_start);
        const bool needs_any_mask = !q_tile_full || !kv_tile_full || needs_causal_mask;

        mma_score_f16_tiled<Br, D_PAD, Bc, Q_STRIDE, K_STRIDE, S_STRIDE>(
            smemQ, smemK[current_stage], score_smem);
        __syncthreads();

        // P = softmax(S)
        if (!needs_any_mask) {
            #pragma unroll
            for (int index = tid; index < Br * Bc; index += 128) {
                const int row = index / Bc, col = index % Bc;
                p_ds_smem[row * PDS_STRIDE + col] = __float2half(
                    __expf(score_smem[row * S_STRIDE + col] * scale - l_smem[row]));
            }
        } else {
            #pragma unroll
            for (int index = tid; index < Br * Bc; index += 128) {
                const int row = index / Bc, col = index % Bc;
                const int gq  = q_start + row;
                const int gk  = kv_start + col;
                bool valid    = gq < Sq && gk < Skv;
                if constexpr (masked)
                    if (needs_causal_mask) valid = valid && gk <= gq;
                p_ds_smem[row * PDS_STRIDE + col] = __float2half(
                    valid
                    ? __expf(score_smem[row * S_STRIDE + col] * scale - l_smem[row])
                    : 0.0f);
            }
        }
        __syncthreads();

        // dP = dO · V^T
        mma_score_f16_tiled<Br, D_PAD, Bc, DO_STRIDE, V_STRIDE, S_STRIDE>(
            smemdO, smemV[current_stage], score_smem);
        __syncthreads();

        // dS = P ∘ (dP − Delta)
        if (!needs_any_mask) {
            #pragma unroll
            for (int index = tid; index < Br * Bc; index += 128) {
                const int row = index / Bc, col = index % Bc;
                const float p  = __half2float(p_ds_smem[row * PDS_STRIDE + col]);
                const float dp = score_smem[row * S_STRIDE + col];
                p_ds_smem[row * PDS_STRIDE + col] =
                    __float2half(p * (dp - delta_smem[row]) * scale);
            }
        } else {
            #pragma unroll
            for (int index = tid; index < Br * Bc; index += 128) {
                const int row = index / Bc, col = index % Bc;
                const int gq  = q_start + row;
                const int gk  = kv_start + col;
                bool valid    = gq < Sq && gk < Skv;
                if constexpr (masked)
                    if (needs_causal_mask) valid = valid && gk <= gq;
                float ds = 0.0f;
                if (valid) {
                    const float p  = __half2float(p_ds_smem[row * PDS_STRIDE + col]);
                    const float dp = score_smem[row * S_STRIDE + col];
                    ds = p * (dp - delta_smem[row]) * scale;
                }
                p_ds_smem[row * PDS_STRIDE + col] = __float2half(ds);
            }
        }
        __syncthreads();

        // dQ += dS · K
        mma_accum_f16_registers<Br, Bc, D_PAD, PDS_STRIDE, K_STRIDE, kAccumulatorCount>(
            p_ds_smem, smemK[current_stage], dQ_fragment);

        if (next_kv_tile < kv_tiles_to_process) {
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
            __syncthreads();
        }
    }

    flashattn_masked_bwd_detail::store_f16<Br, D_PAD, kAccumulatorCount>(
        dQ_fragment, dQptr, q_start, Sq, actual_D);
}

 

template<int Br, int Bc, int D_PAD>
constexpr size_t flashattn_bwd_dkdv_smem_bytes() {
    constexpr int PAD       = 8;
    constexpr int D_STRIDE  = D_PAD + PAD;
    constexpr int S_STRIDE  = Bc + PAD;
    constexpr int PDS_STRIDE = Br + PAD;

    return
        2 * Br * D_STRIDE * sizeof(__half) +  // smemQ double-buffer
        2 * Br * D_STRIDE * sizeof(__half) +  // smemdO double-buffer
        Bc * D_STRIDE * sizeof(__half) +      // smemK
        Bc * D_STRIDE * sizeof(__half) +      // smemV
        Br * S_STRIDE * sizeof(float)  +      // score_smem
        Bc * PDS_STRIDE * sizeof(__half) +    // p_ds_smem [Bc,Br]
        2 * Br * sizeof(float) +              // l_smem, delta_smem
        160;                                  // alignment slack: 10 align_ptr()
                                               // calls in the kernel, each wastes
                                               // at most 15B rounding to 16B -> 150B
                                               // worst case, rounded up to 160B.
}

template<int Br, int Bc, int D_PAD>
constexpr size_t flashattn_bwd_dq_smem_bytes() {
    constexpr int PAD        = 8;
    constexpr int D_STRIDE   = D_PAD + PAD;
    constexpr int S_STRIDE   = Bc + PAD;
    constexpr int PDS_STRIDE = Bc + PAD;

    return
        Br * D_STRIDE * sizeof(__half) +      // smemQ
        Br * D_STRIDE * sizeof(__half) +      // smemdO
        2 * Bc * D_STRIDE * sizeof(__half) +  // smemK double-buffer
        2 * Bc * D_STRIDE * sizeof(__half) +  // smemV double-buffer
        Br * S_STRIDE * sizeof(float)  +      // score_smem
        Br * PDS_STRIDE * sizeof(__half) +    // p_ds_smem [Br,Bc]
        2 * Br * sizeof(float) +              // l_smem, delta_smem
        160;                                  // alignment slack: 10 align_ptr()
                                               // calls in the kernel, each wastes
                                               // at most 15B rounding to 16B -> 150B
                                               // worst case, rounded up to 160B.
}
