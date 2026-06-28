/* 
####################################################################
this version is O accumulation in shared mem
####################################################################
*/
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <iostream>

#include "helper.cuh"
template<int Br, int Bc, int NUM_HEADS, int D, int N>
__global__ void flashattn_fwd_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
          __half* __restrict__ output,
          float*  __restrict__ Logsum
)
{
    const int tid   = threadIdx.x;
    const int warp  = tid / 32;
    const int lane  = tid % 32;
    const int group = lane / 4;

    const float SCALE = 1.0f / sqrtf((float)D);

    extern __shared__ char smem_raw[];

    char* ptr = smem_raw;

    constexpr int PAD = 8;

    constexpr int Q_STRIDE = D + PAD;
    constexpr int K_STRIDE = D + PAD;
    constexpr int V_STRIDE = D + PAD;
    constexpr int S_STRIDE = Bc;
    constexpr int O_STRIDE = D + PAD;

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 15) & ~15ULL
    );

    __half* Qsmem = reinterpret_cast<__half*>(ptr);
    ptr += Br * Q_STRIDE * sizeof(__half);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 15) & ~15ULL
    );

    __half* Ksmem0 = reinterpret_cast<__half*>(ptr);
    ptr += Bc * K_STRIDE * sizeof(__half);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 15) & ~15ULL
    );

    __half* Ksmem1 = reinterpret_cast<__half*>(ptr);
    ptr += Bc * K_STRIDE * sizeof(__half);

    __half* Ksmem[2] = {Ksmem0, Ksmem1};

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 15) & ~15ULL
    );

    __half* Vsmem0 = reinterpret_cast<__half*>(ptr);
    ptr += Bc * V_STRIDE * sizeof(__half);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 15) & ~15ULL
    );

    __half* Vsmem1 = reinterpret_cast<__half*>(ptr);
    ptr += Bc * V_STRIDE * sizeof(__half);

    __half* Vsmem[2] = {Vsmem0, Vsmem1};

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 15) & ~15ULL
    );

    float* Ssmem = reinterpret_cast<float*>(ptr);
    ptr += Br * S_STRIDE * sizeof(float);

    ptr = reinterpret_cast<char*>(
    (reinterpret_cast<uintptr_t>(ptr) + 15) & ~15ULL
    );

    float* Msmem = reinterpret_cast<float*>(ptr);
    ptr += Br * sizeof(float);

    ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(ptr) + 15) & ~15ULL
    );

    float* Lsmem = reinterpret_cast<float*>(ptr);
    ptr += Br * sizeof(float);

    ptr = reinterpret_cast<char*>(
    (reinterpret_cast<uintptr_t>(ptr) + 15) & ~15ULL
    );

    float* Oacc = reinterpret_cast<float*>(ptr);
    ptr += Br * O_STRIDE * sizeof(float);

    const int batchid = blockIdx.x;
    const int headid  = blockIdx.y;
    const int rowid   = blockIdx.z;

    const long long base =
        (long long)batchid * NUM_HEADS * N * D +
        (long long)headid  * N * D;
    
    const long long statbase =
    (long long)batchid * NUM_HEADS * N +
    (long long)headid  * N;

    const __half* Qptr = Q + base;
    const __half* Kptr = K + base;
    const __half* Vptr = V + base;
          __half* Optr = output + base;
          float* Lptr = Logsum + statbase;

    const int Tr = (N + Br - 1) / Br;
    const int Tc = (N + Bc - 1) / Bc;

    if (rowid >= Tr) return;

    {
        uint32_t q_smem = smem_u32_ptr(Qsmem);

        asyncLOAD_2D_TILE<Br, D, 128>(
            Qptr,
            q_smem,
            tid,
            Q_STRIDE,
            N,
            D,
            D,
            rowid,
            0
        );
    }

    {
        uint32_t k_smem = smem_u32_ptr(Ksmem[0]);

        asyncLOAD_2D_TILE<Bc, D, 128>(
            Kptr,
            k_smem,
            tid,
            K_STRIDE,
            N,
            D,
            D,
            0,
            0
        );
    }

    {
        uint32_t v_smem = smem_u32_ptr(Vsmem[0]);

        asyncLOAD_2D_TILE<Bc, D, 128>(
            Vptr,
            v_smem,
            tid,
            V_STRIDE,
            N,
            D,
            D,
            0,
            0
        );
    }

    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 1;\n" ::: "memory");
    __syncthreads();

    for (int r = tid; r < Br; r += blockDim.x) {
        Msmem[r] = -FLT_MAX;
        Lsmem[r] = 0.0f;
    }     
    for (int i = tid; i < Br * O_STRIDE; i += blockDim.x) {
        Oacc[i] = 0.0f;
    }

    __syncthreads();

    for (int kv_tile = 0; kv_tile < Tc; kv_tile++) {
        const int curstage  = kv_tile & 1;
        const int nextstage = curstage ^ 1;
        const int next_kv   = kv_tile + 1;

        for (int i = tid; i < Br * Bc; i += blockDim.x) {
            Ssmem[i] = 0.0f;
        }

        __syncthreads();

        if (next_kv < Tc) {

            {
                uint32_t v_smem = smem_u32_ptr(Vsmem[nextstage]);

                asyncLOAD_2D_TILE<Bc, D, 128>(
                    Vptr,
                    v_smem,
                    tid,
                    V_STRIDE,
                    N,
                    D,
                    D,
                    next_kv,
                    0
                );
            }
        
            {
                uint32_t k_smem = smem_u32_ptr(Ksmem[nextstage]);

                asyncLOAD_2D_TILE<Bc, D, 128>(
                    Kptr,
                    k_smem,
                    tid,
                    K_STRIDE,
                    N,
                    D,
                    D,
                    next_kv,
                    0
                );
            }


            asm volatile("cp.async.commit_group;\n");
        }

        mma_score_strided(
            Qsmem,
            Ksmem[curstage],
            Ssmem,
            Br,
            D,
            Bc,
            Q_STRIDE,
            K_STRIDE,
            S_STRIDE
        );

        if (next_kv < Tc) {
            asm volatile("cp.async.wait_group 2;\n" ::: "memory");
        }

        __syncthreads();

        for(int row = warp; row < Br; row += 4)
        {
            int globalitr = rowid * Br + row;

            if(globalitr >= N) continue;

            float m_val = -FLT_MAX;

            for(int i = lane; i < Bc; i += 32) {
                int global_col = kv_tile * Bc + i;

                if (global_col < N) {
                    float s = Ssmem[row * S_STRIDE + i] * SCALE;
                    m_val = fmaxf(m_val, s);
                }
            }

            #pragma unroll
            for(int offset = 16; offset > 0; offset >>= 1) {
                m_val = fmaxf(m_val, __shfl_xor_sync(0xffffffff, m_val, offset));
            }

            float l_val = 0.f;

            for(int i = lane; i < Bc; i += 32) {
                int global_col = kv_tile * Bc + i;

                if (global_col < N) {
                    float s = Ssmem[row * S_STRIDE + i] * SCALE;
                    l_val += __expf(s - m_val);
                }
            }

            #pragma unroll
            for(int offset = 16; offset > 0; offset >>= 1) {
                l_val += __shfl_xor_sync(0xffffffff, l_val, offset);
            }

            float m_prev = Msmem[row];
            float l_prev = Lsmem[row];

            float m_new = fmaxf(m_prev, m_val);

            float alpha = __expf(m_prev - m_new);
            float beta  = __expf(m_val  - m_new);

            float l_new = alpha * l_prev + beta * l_val;

            for(int c = lane; c < D; c += 32) {
                Oacc[row * O_STRIDE + c] *= alpha;
            }

            for (int i = lane; i < Bc; i += 32) {
                int global_col = kv_tile * Bc + i;

                if (global_col < N) {
                    float s = Ssmem[row * S_STRIDE + i] * SCALE;
                    float p = beta * __expf(s - m_val);
                    Ssmem[row * S_STRIDE + i] = p;
                } else {
                    Ssmem[row * S_STRIDE + i] = 0.0f;
                }
            }

            if (lane == 0) {
                Msmem[row] = m_new;
                Lsmem[row] = l_new;
            }
        }

        __syncthreads();
        /// we will create new function fr this , one is fload and other is float
        mma_pv_accum_f32p_f16v(
            Ssmem,
            Vsmem[curstage],
            Oacc,

            Br,
            Bc,
            D,

            S_STRIDE,
            V_STRIDE,
            O_STRIDE
        );
    }
    /// now we have the acc for O
    /// store it globally 
    for(int i = tid; i < Br * D; i += blockDim.x)
    {
        int r = i / D;
        int c = i % D;

        int mainrow = rowid * Br + r;

        if(mainrow < N && c < D)
        {
            float out = 0.f;

            if(Lsmem[r] > 0.f) {
                out = Oacc[r * O_STRIDE + c] / Lsmem[r];
            }

            Optr[mainrow * D + c] = __float2half(out);

            if (c == 0) {
                Lptr[mainrow] = Msmem[r] + logf(Lsmem[r]);
            }
        }
    }
    
}