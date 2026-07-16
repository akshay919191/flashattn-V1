#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include "helper.cuh"


template<int Br, int Bc, int D>
__global__ void flashattn_fwd_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
          __half* __restrict__ output,
          float*  __restrict__ Logsum,
          int N
) {
    const int tid   = threadIdx.x;
    const int warp  = tid / 32;
    const int lane  = tid & 31;

    const float SCALE = 1.0f / sqrtf((float)D);

    extern __shared__ char smem_raw[];
    char* ptr = smem_raw;

    auto align = [&](char*& p, size_t a = 16) {
        p = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(p) + a - 1) & ~(a - 1));
    };

    constexpr int PAD = 8;
    constexpr int Q_STRIDE = D + PAD;
    constexpr int K_STRIDE = D + PAD;
    constexpr int V_STRIDE = D + PAD;

    align(ptr);
    __half* Qsmem = reinterpret_cast<__half*>(ptr); ptr += Br * Q_STRIDE * sizeof(__half);
    align(ptr);
    __half* Ksmem0 = reinterpret_cast<__half*>(ptr); ptr += Bc * K_STRIDE * sizeof(__half);
    align(ptr);
    __half* Ksmem1 = reinterpret_cast<__half*>(ptr); ptr += Bc * K_STRIDE * sizeof(__half);
    __half* Ksmem[2] = {Ksmem0, Ksmem1};
    align(ptr);
    __half* Vsmem0 = reinterpret_cast<__half*>(ptr); ptr += Bc * V_STRIDE * sizeof(__half);
    align(ptr);
    __half* Vsmem1 = reinterpret_cast<__half*>(ptr); ptr += Bc * V_STRIDE * sizeof(__half);
    __half* Vsmem[2] = {Vsmem0, Vsmem1};

    const int batchid = blockIdx.x;
    const int headid  = blockIdx.y;
    const int rowid   = blockIdx.z;
    const int H_runtime = gridDim.y;

    const long long base = ((long long)batchid * H_runtime + headid) * N * D;
    const long long statbase = ((long long)batchid * H_runtime + headid) * N;

    const __half* Qptr = Q + base;
    const __half* Kptr = K + base;
    const __half* Vptr = V + base;
          __half* Optr = output + base;
          float* Lptr = Logsum + statbase;

    const int Tr = (N + Br - 1) / Br;
    const int Tc = (N + Bc - 1) / Bc;

    if (rowid >= Tr) return;

    asyncLOAD_2D_TILE<Br, D, 128>(Qptr, smem_u32_ptr(Qsmem), tid, Q_STRIDE, N, D, D, rowid, 0);
    asyncLOAD_2D_TILE<Bc, D, 128>(Kptr, smem_u32_ptr(Ksmem[0]), tid, K_STRIDE, N, D, D, 0, 0);
    asyncLOAD_2D_TILE<Bc, D, 128>(Vptr, smem_u32_ptr(Vsmem[0]), tid, V_STRIDE, N, D, D, 0, 0);
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    constexpr int Dk = D / 16;
    constexpr int Bk = Bc / 8;
    constexpr int Dv = D / 8;

    float O_frag[Dv * 4] = {0.0f};
    float m_frag[2] = {-FLT_MAX, -FLT_MAX};
    float l_frag[2] = {0.0f};

    for (int kv_tile = 0; kv_tile < Tc; kv_tile++) {
        const int curstage  = kv_tile & 1;
        const int nextstage = curstage ^ 1;
        const int next_kv   = kv_tile + 1;

        if (next_kv < Tc) {
            asyncLOAD_2D_TILE<Bc, D, 128>(Vptr, smem_u32_ptr(Vsmem[nextstage]), tid, V_STRIDE, N, D, D, next_kv, 0);
            asyncLOAD_2D_TILE<Bc, D, 128>(Kptr, smem_u32_ptr(Ksmem[nextstage]), tid, K_STRIDE, N, D, D, next_kv, 0);
            asm volatile("cp.async.commit_group;\n");
        }

        float S_frag[Bk * 4] = {0.0f};

        // 1. S = Q @ K^T
        for (int kb = 0; kb < Bk; kb++) {
            for (int ks = 0; ks < Dk; ks++) {
                uint32_t q_frag[4];
                int q_r = warp * 16 + (lane % 16);
                int q_c = ks * 16 + ((lane < 16) ? 0 : 8);
                uint32_t q_addr = smem_u32_ptr(Qsmem + q_r * Q_STRIDE + q_c);
                ldmatrix_x4(q_frag, q_addr);

                uint32_t k_frag[2];
                const int lane16 = lane & 15;
                int k_r = kb * 8 + (lane16 & 7);
                int k_c = ks * 16 + ((lane16 >> 3) * 8);

                uint32_t k_addr =
                    smem_u32_ptr(Ksmem[curstage] + k_r * K_STRIDE + k_c);

                ldmatrix_x2(k_frag, k_addr);

                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(S_frag[kb * 4 + 0]), "+f"(S_frag[kb * 4 + 1]), "+f"(S_frag[kb * 4 + 2]), "+f"(S_frag[kb * 4 + 3])
                    : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]),
                      "r"(k_frag[0]), "r"(k_frag[1])
                );
            }
        }

        float m_val[2] = {-FLT_MAX, -FLT_MAX};
        const int lane4 = lane & 3;

        for (int kb = 0; kb < Bk; ++kb) {
            const int key0 = kv_tile * Bc + kb * 8 + lane4 * 2;
            const int key1 = key0 + 1;

            float s0 = S_frag[kb * 4 + 0] * SCALE;
            float s1 = S_frag[kb * 4 + 1] * SCALE;
            float s2 = S_frag[kb * 4 + 2] * SCALE;
            float s3 = S_frag[kb * 4 + 3] * SCALE;

            if (key0 >= N) {
                s0 = -FLT_MAX;
                s2 = -FLT_MAX;
            }

            if (key1 >= N) {
                s1 = -FLT_MAX;
                s3 = -FLT_MAX;
            }

            S_frag[kb * 4 + 0] = s0;
            S_frag[kb * 4 + 1] = s1;
            S_frag[kb * 4 + 2] = s2;
            S_frag[kb * 4 + 3] = s3;

            float m0 = fmaxf(s0, s1);
            float m1 = fmaxf(s2, s3);

            m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1, 4));
            m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2, 4));

            m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 1, 4));
            m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 2, 4));

            m_val[0] = fmaxf(m_val[0], m0);
            m_val[1] = fmaxf(m_val[1], m1);
        }

        float m_prev0 = m_frag[0];
        float m_prev1 = m_frag[1];
        float m_new0 = fmaxf(m_prev0, m_val[0]);
        float m_new1 = fmaxf(m_prev1, m_val[1]);

        float alpha0 = __expf(m_prev0 - m_new0);
        float alpha1 = __expf(m_prev1 - m_new1);

        for (int vs = 0; vs < Dv; vs++) {
            O_frag[vs * 4 + 0] *= alpha0;
            O_frag[vs * 4 + 1] *= alpha0;
            O_frag[vs * 4 + 2] *= alpha1;
            O_frag[vs * 4 + 3] *= alpha1;
        }

        m_frag[0] = m_new0;
        m_frag[1] = m_new1;

        float l0 = 0.0f, l1 = 0.0f;

        // 3. P @ V
        for (int kb = 0; kb < Bk; kb += 2) {
            uint32_t P_frag[4];
            
            float p0 = __expf(S_frag[kb * 4 + 0] - m_new0);
            float p1 = __expf(S_frag[kb * 4 + 1] - m_new0);
            float p2 = __expf(S_frag[kb * 4 + 2] - m_new1);
            float p3 = __expf(S_frag[kb * 4 + 3] - m_new1);
            
            l0 += (p0 + p1);
            l1 += (p2 + p3);

            // A-operand quadrant order is {top-left, bottom-left, top-right, bottom-right}
            P_frag[0] = pack_float2_to_half2_u32(p0, p1); // top-left
            P_frag[1] = pack_float2_to_half2_u32(p2, p3); // bottom-left

            if (kb + 1 < Bk) {
                float p4 = __expf(S_frag[(kb + 1) * 4 + 0] - m_new0);
                float p5 = __expf(S_frag[(kb + 1) * 4 + 1] - m_new0);
                float p6 = __expf(S_frag[(kb + 1) * 4 + 2] - m_new1);
                float p7 = __expf(S_frag[(kb + 1) * 4 + 3] - m_new1);
                
                l0 += (p4 + p5);
                l1 += (p6 + p7);

                P_frag[2] = pack_float2_to_half2_u32(p4, p5); // top-right
                P_frag[3] = pack_float2_to_half2_u32(p6, p7); // bottom-right
            } else {
                P_frag[2] = 0;
                P_frag[3] = 0;
            }

            int v_r = (kb / 2) * 16 + (lane % 16);
            for (int vs = 0; vs < Dv; vs++) {
                uint32_t v_frag[2];
                int v_c = vs * 8;
                uint32_t v_addr = smem_u32_ptr(Vsmem[curstage] + v_r * V_STRIDE + v_c);
                ldmatrix_x2_trans(v_frag, v_addr);

                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(O_frag[vs * 4 + 0]), "+f"(O_frag[vs * 4 + 1]), "+f"(O_frag[vs * 4 + 2]), "+f"(O_frag[vs * 4 + 3])
                    : "r"(P_frag[0]), "r"(P_frag[1]), "r"(P_frag[2]), "r"(P_frag[3]),
                      "r"(v_frag[0]), "r"(v_frag[1])
                );
            }
        }

        // Reduce l0 and l1 across the 8 columns
        l0 += __shfl_xor_sync(0xffffffffu, l0, 1, 4);
        l0 += __shfl_xor_sync(0xffffffffu, l0, 2, 4);

        l1 += __shfl_xor_sync(0xffffffffu, l1, 1, 4);
        l1 += __shfl_xor_sync(0xffffffffu, l1, 2, 4);

        l_frag[0] = l_frag[0] * alpha0 + l0;
        l_frag[1] = l_frag[1] * alpha1 + l1;

        if (next_kv < Tc) {
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
            __syncthreads();
        }
    }

    int row0 = rowid * Br + warp * 16 + (lane / 4);
    int row1 = row0 + 8;
    int col0 = (lane % 4) * 2;
    int col1 = col0 + 1;

    for (int vs = 0; vs < Dv; vs++) {
        int c0 = vs * 8 + col0;
        int c1 = vs * 8 + col1;
        if (row0 < N) {
            Optr[row0 * D + c0] = __float2half(O_frag[vs * 4 + 0] / l_frag[0]);
            Optr[row0 * D + c1] = __float2half(O_frag[vs * 4 + 1] / l_frag[0]);
        }
        if (row1 < N) {
            Optr[row1 * D + c0] = __float2half(O_frag[vs * 4 + 2] / l_frag[1]);
            Optr[row1 * D + c1] = __float2half(O_frag[vs * 4 + 3] / l_frag[1]);
        }
    }

    if ((lane % 4) == 0) {
        if (row0 < N) Lptr[row0] = m_frag[0] + logf(l_frag[0]);
        if (row1 < N) Lptr[row1] = m_frag[1] + logf(l_frag[1]);
    }
}

template<int D>
__global__ void calc_delta_kernel(
    const __half* __restrict__ O,
    const __half* __restrict__ dO,
    float*        __restrict__ Delta,
    int total_rows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < total_rows) {
        const __half* o_row = O + (size_t)row * D;
        const __half* do_row = dO + (size_t)row * D;
        
        float delta = 0.0f;
        for (int i = 0; i < D; i++) {
            delta += __half2float(o_row[i]) * __half2float(do_row[i]);
        }
        Delta[row] = delta;
    }
}

template<int Br, int Bc, int D>
__global__ void flashattn_bwd_dkdv_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    const __half* __restrict__ DO,
    const float*  __restrict__ L,
    const float*  __restrict__ delta,
          __half* __restrict__ DK,
          __half* __restrict__ DV,
          int N
) 
{
    static_assert(Br > 0 && Br % 16 == 0,
                  "Backward kernels require Br divisible by 16");
    static_assert(Bc > 0 && Bc % 16 == 0,
                  "Backward kernels require Bc divisible by 16");
    static_assert(D > 0 && D % 16 == 0,
                  "Backward kernels require D divisible by 16");

    if (blockDim.x != 128) return;

    const int tid   = threadIdx.x;
    const float SCALE = 1.0f / sqrtf((float)D);

    extern __shared__ char smem_raw[];
    char* ptr = smem_raw;

    auto align = [&](char*& p, size_t a = 16) {
        p = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(p) + a - 1) & ~(a - 1));
    };

    constexpr int PAD = 8;
    constexpr int Q_STRIDE  = D + PAD;
    constexpr int K_STRIDE  = D + PAD;
    constexpr int V_STRIDE  = D + PAD;
    constexpr int DO_STRIDE = D + PAD;
    constexpr int S_STRIDE  = Bc + PAD;

    align(ptr);
    __half* smemQ0 = reinterpret_cast<__half*>(ptr); ptr += Br * Q_STRIDE * sizeof(__half);
    align(ptr);
    __half* smemQ1 = reinterpret_cast<__half*>(ptr); ptr += Br * Q_STRIDE * sizeof(__half);
    __half* smemQ[2] = {smemQ0, smemQ1};

    align(ptr);
    __half* smemdO0 = reinterpret_cast<__half*>(ptr); ptr += Br * DO_STRIDE * sizeof(__half);
    align(ptr);
    __half* smemdO1 = reinterpret_cast<__half*>(ptr); ptr += Br * DO_STRIDE * sizeof(__half);
    __half* smemdO[2] = {smemdO0, smemdO1};

    align(ptr);
    __half* smemK = reinterpret_cast<__half*>(ptr); ptr += Bc * K_STRIDE * sizeof(__half);
    align(ptr);
    __half* smemV = reinterpret_cast<__half*>(ptr); ptr += Bc * V_STRIDE * sizeof(__half);

    align(ptr);
    float* dKacc = reinterpret_cast<float*>(ptr); ptr += Bc * K_STRIDE * sizeof(float);
    align(ptr);
    float* dVacc = reinterpret_cast<float*>(ptr); ptr += Bc * V_STRIDE * sizeof(float);

    align(ptr);
    float* WorkSmem = reinterpret_cast<float*>(ptr); ptr += Br * S_STRIDE * sizeof(float);
    align(ptr);
    __half* P_dS_smem = reinterpret_cast<__half*>(ptr); ptr += Bc * Br * sizeof(__half);

    align(ptr);
    float* Lsmem = reinterpret_cast<float*>(ptr); ptr += Br * sizeof(float);
    align(ptr);
    float* Deltasmem = reinterpret_cast<float*>(ptr); ptr += Br * sizeof(float);

    const int batchid = blockIdx.x;
    const int headid  = blockIdx.y;
    const int kvid    = blockIdx.z;
    const int H_runtime = gridDim.y;

    const long long base = ((long long)batchid * H_runtime + headid) * N * D;
    const long long statbase = ((long long)batchid * H_runtime + headid) * N;

    const __half* Qptr  = Q  + base;
    const __half* Kptr  = K  + base;
    const __half* Vptr  = V  + base;
    const __half* DOptr = DO + base;
    const float* Lptr     = L     + statbase;
    const float* Deltaptr = delta + statbase;
    __half* DKptr = DK + base;
    __half* DVptr = DV + base;

    const int Tr = (N + Br - 1) / Br;
    const int Tc = (N + Bc - 1) / Bc;

    if (kvid >= Tc) return;

    for (int i = tid; i < Bc * K_STRIDE; i += blockDim.x) dKacc[i] = 0.0f;
    for (int i = tid; i < Bc * V_STRIDE; i += blockDim.x) dVacc[i] = 0.0f;

    asyncLOAD_2D_TILE<Bc, D, 128>(Kptr, smem_u32_ptr(smemK), tid, K_STRIDE, N, D, D, kvid, 0);
    asyncLOAD_2D_TILE<Bc, D, 128>(Vptr, smem_u32_ptr(smemV), tid, V_STRIDE, N, D, D, kvid, 0);
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    asyncLOAD_2D_TILE<Br, D, 128>(Qptr, smem_u32_ptr(smemQ[0]), tid, Q_STRIDE, N, D, D, 0, 0);
    asyncLOAD_2D_TILE<Br, D, 128>(DOptr, smem_u32_ptr(smemdO[0]), tid, DO_STRIDE, N, D, D, 0, 0);
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    for (int Qtileid = 0; Qtileid < Tr; Qtileid++) {
        int curstage = Qtileid & 1;
        int nextstage = curstage ^ 1;
        int next_q = Qtileid + 1;

        if (next_q < Tr) {
            asyncLOAD_2D_TILE<Br, D, 128>(Qptr, smem_u32_ptr(smemQ[nextstage]), tid, Q_STRIDE, N, D, D, next_q, 0);
            asyncLOAD_2D_TILE<Br, D, 128>(DOptr, smem_u32_ptr(smemdO[nextstage]), tid, DO_STRIDE, N, D, D, next_q, 0);
            asm volatile("cp.async.commit_group;\n");
        }

        for (int r = tid; r < Br; r += blockDim.x) {
            int global_q = Qtileid * Br + r;
            if (global_q < N) {
                Lsmem[r]     = Lptr[global_q];
                Deltasmem[r] = Deltaptr[global_q];
            } else {
                Lsmem[r]     = 0.0f;
                Deltasmem[r] = 0.0f;
            }
        }
        __syncthreads();

        mma_score_strided(smemQ[curstage], smemK, WorkSmem, Br, D, Bc, Q_STRIDE, K_STRIDE, S_STRIDE);
        __syncthreads();

        for (int idx = tid; idx < Br * Bc; idx += blockDim.x) {
            int r = idx / Bc; int c = idx % Bc;
            int global_q = Qtileid * Br + r; int global_k = kvid * Bc + c;
            float p = 0.0f;
            if (global_q < N && global_k < N) {
                float score = WorkSmem[r * S_STRIDE + c];
                p = __expf(score * SCALE - Lsmem[r]);
            }
            P_dS_smem[c * Br + r] = __float2half(p);
        }
        __syncthreads();

        mma_accum_f16p_f16v_smem(P_dS_smem, smemdO[curstage], dVacc, Bc, Br, D, Br, DO_STRIDE, V_STRIDE);
        __syncthreads();

        mma_score_strided(smemdO[curstage], smemV, WorkSmem, Br, D, Bc, DO_STRIDE, V_STRIDE, S_STRIDE);
        __syncthreads();

        for (int idx = tid; idx < Br * Bc; idx += blockDim.x) {
            int r = idx / Bc; int c = idx % Bc;
            int global_q = Qtileid * Br + r; int global_k = kvid * Bc + c;
            float ds = 0.0f;
            if (global_q < N && global_k < N) {
                float dp = WorkSmem[r * S_STRIDE + c];
                float p  = __half2float(P_dS_smem[c * Br + r]);
                ds = p * (dp - Deltasmem[r]) * SCALE;
            }
            P_dS_smem[c * Br + r] = __float2half(ds);
        }
        __syncthreads();

        mma_accum_f16p_f16v_smem(P_dS_smem, smemQ[curstage], dKacc, Bc, Br, D, Br, Q_STRIDE, K_STRIDE);
        __syncthreads();

        if (next_q < Tr) {
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
            __syncthreads();
        }
    }

    for (int i = tid; i < Bc * D; i += blockDim.x) {
        int r = i / D; int c = i % D;
        int global_r = kvid * Bc + r;
        if (global_r < N && c < D) {
            DKptr[(size_t)global_r * D + c] = __float2half(dKacc[r * K_STRIDE + c]);
            DVptr[(size_t)global_r * D + c] = __float2half(dVacc[r * V_STRIDE + c]);
        }
    }
}

template<int Br, int Bc, int D>
__global__ void flashattn_bwd_dq_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    const __half* __restrict__ DO,
    const float*  __restrict__ L,
    const float*  __restrict__ delta,
          __half* __restrict__ DQ,
          int N
) 
{
    static_assert(Br > 0 && Br % 16 == 0,
                  "Backward kernels require Br divisible by 16");
    static_assert(Bc > 0 && Bc % 16 == 0,
                  "Backward kernels require Bc divisible by 16");
    static_assert(D > 0 && D % 16 == 0,
                  "Backward kernels require D divisible by 16");

    // The asynchronous loaders below are specialized for 128 threads.
    // This condition is uniform across the block, so returning is barrier-safe.
    if (blockDim.x != 128) return;

    const int tid   = threadIdx.x;
    const float SCALE = 1.0f / sqrtf((float)D);

    extern __shared__ char smem_raw[];
    char* ptr = smem_raw;

    auto align = [&](char*& p, size_t a = 16) {
        p = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(p) + a - 1) & ~(a - 1));
    };

    constexpr int PAD = 8;
    constexpr int Q_STRIDE  = D + PAD;
    constexpr int K_STRIDE  = D + PAD;
    constexpr int V_STRIDE  = D + PAD;
    constexpr int DO_STRIDE = D + PAD;
    constexpr int S_STRIDE  = Bc + PAD;

    align(ptr);
    __half* smemQ = reinterpret_cast<__half*>(ptr); ptr += Br * Q_STRIDE * sizeof(__half);
    align(ptr);
    __half* smemdO = reinterpret_cast<__half*>(ptr); ptr += Br * DO_STRIDE * sizeof(__half);

    align(ptr);
    __half* smemK0 = reinterpret_cast<__half*>(ptr); ptr += Bc * K_STRIDE * sizeof(__half);
    align(ptr);
    __half* smemK1 = reinterpret_cast<__half*>(ptr); ptr += Bc * K_STRIDE * sizeof(__half);
    __half* smemK[2] = {smemK0, smemK1};

    align(ptr);
    __half* smemV0 = reinterpret_cast<__half*>(ptr); ptr += Bc * V_STRIDE * sizeof(__half);
    align(ptr);
    __half* smemV1 = reinterpret_cast<__half*>(ptr); ptr += Bc * V_STRIDE * sizeof(__half);
    __half* smemV[2] = {smemV0, smemV1};

    align(ptr);
    float* dQacc = reinterpret_cast<float*>(ptr); ptr += Br * Q_STRIDE * sizeof(float);

    align(ptr);
    float* WorkSmem = reinterpret_cast<float*>(ptr); ptr += Br * S_STRIDE * sizeof(float);
    align(ptr);
    __half* P_dS_smem = reinterpret_cast<__half*>(ptr); ptr += Br * Bc * sizeof(__half);

    align(ptr);
    float* Lsmem = reinterpret_cast<float*>(ptr); ptr += Br * sizeof(float);
    align(ptr);
    float* Deltasmem = reinterpret_cast<float*>(ptr); ptr += Br * sizeof(float);

    const int batchid = blockIdx.x;
    const int headid  = blockIdx.y;
    const int qid     = blockIdx.z;
    const int H_runtime = gridDim.y;

    const long long base = ((long long)batchid * H_runtime + headid) * N * D;
    const long long statbase = ((long long)batchid * H_runtime + headid) * N;

    const __half* Qptr  = Q  + base;
    const __half* Kptr  = K  + base;
    const __half* Vptr  = V  + base;
    const __half* DOptr = DO + base;
    const float* Lptr     = L     + statbase;
    const float* Deltaptr = delta + statbase;
    __half* DQptr = DQ + base;

    const int Tr = (N + Br - 1) / Br;
    const int Tc = (N + Bc - 1) / Bc;

    if (qid >= Tr) return;

    for (int i = tid; i < Br * Q_STRIDE; i += blockDim.x) dQacc[i] = 0.0f;

    asyncLOAD_2D_TILE<Br, D, 128>(Qptr, smem_u32_ptr(smemQ), tid, Q_STRIDE, N, D, D, qid, 0);
    asyncLOAD_2D_TILE<Br, D, 128>(DOptr, smem_u32_ptr(smemdO), tid, DO_STRIDE, N, D, D, qid, 0);
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    for (int r = tid; r < Br; r += blockDim.x) {
        int global_q = qid * Br + r;
        if (global_q < N) {
            Lsmem[r]     = Lptr[global_q];
            Deltasmem[r] = Deltaptr[global_q];
        } else {
            Lsmem[r]     = 0.0f;
            Deltasmem[r] = 0.0f;
        }
    }
    __syncthreads();

    asyncLOAD_2D_TILE<Bc, D, 128>(Kptr, smem_u32_ptr(smemK[0]), tid, K_STRIDE, N, D, D, 0, 0);
    asyncLOAD_2D_TILE<Bc, D, 128>(Vptr, smem_u32_ptr(smemV[0]), tid, V_STRIDE, N, D, D, 0, 0);
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    for (int kvid = 0; kvid < Tc; kvid++) {
        int curstage = kvid & 1;
        int nextstage = curstage ^ 1;
        int next_k = kvid + 1;

        if (next_k < Tc) {
            asyncLOAD_2D_TILE<Bc, D, 128>(Kptr, smem_u32_ptr(smemK[nextstage]), tid, K_STRIDE, N, D, D, next_k, 0);
            asyncLOAD_2D_TILE<Bc, D, 128>(Vptr, smem_u32_ptr(smemV[nextstage]), tid, V_STRIDE, N, D, D, next_k, 0);
            asm volatile("cp.async.commit_group;\n");
        }

        mma_score_strided(smemQ, smemK[curstage], WorkSmem, Br, D, Bc, Q_STRIDE, K_STRIDE, S_STRIDE);
        __syncthreads();

        for (int idx = tid; idx < Br * Bc; idx += blockDim.x) {
            int r = idx / Bc; int c = idx % Bc;
            int global_q = qid * Br + r; int global_k = kvid * Bc + c;
            float p = 0.0f;
            if (global_q < N && global_k < N) {
                float score = WorkSmem[r * S_STRIDE + c];
                p = __expf(score * SCALE - Lsmem[r]);
            }
            P_dS_smem[r * Bc + c] = __float2half(p);
        }
        __syncthreads();

        mma_score_strided(smemdO, smemV[curstage], WorkSmem, Br, D, Bc, DO_STRIDE, V_STRIDE, S_STRIDE);
        __syncthreads();

        for (int idx = tid; idx < Br * Bc; idx += blockDim.x) {
            int r = idx / Bc; int c = idx % Bc;
            int global_q = qid * Br + r; int global_k = kvid * Bc + c;
            float ds = 0.0f;
            if (global_q < N && global_k < N) {
                float p  = __half2float(P_dS_smem[r * Bc + c]);
                float dp = WorkSmem[r * S_STRIDE + c];
                ds = p * (dp - Deltasmem[r]) * SCALE;
            }
            P_dS_smem[r * Bc + c] = __float2half(ds);
        }
        __syncthreads();

        mma_accum_f16p_f16v_smem(P_dS_smem, smemK[curstage], dQacc, Br, Bc, D, Bc, K_STRIDE, Q_STRIDE);
        __syncthreads();

        if (next_k < Tc) {
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
            __syncthreads();
        }
    }

    for (int i = tid; i < Br * D; i += blockDim.x) {
        int r = i / D; int c = i % D;
        int global_r = qid * Br + r;
        if (global_r < N && c < D) {
            DQptr[(size_t)global_r * D + c] = __float2half(dQacc[r * Q_STRIDE + c]);
        }
    }
}