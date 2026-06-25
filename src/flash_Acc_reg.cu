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
    const int tid4 = lane & 3;

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

    float* Alphasmem = reinterpret_cast<float*>(ptr);
    ptr += Br * sizeof(float);

    constexpr int WARPS = 4;
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;

    constexpr int NUM_M_TILES = (Br + MMA_M - 1) / MMA_M;
    constexpr int NUM_N_TILES = (D  + MMA_N - 1) / MMA_N;
    constexpr int TOTAL_O_TILES = NUM_M_TILES * NUM_N_TILES; /// total tiles
    constexpr int O_TILES_PER_WARP = (TOTAL_O_TILES + WARPS - 1) / WARPS; /// how many tiles per warp
    constexpr int O_REGS = O_TILES_PER_WARP * 4; /// per thread private register

    float Oreg[O_REGS];

    #pragma unroll
    for (int i = 0; i < O_REGS; i++) {
        Oreg[i] = 0.0f;
    }

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
        Alphasmem[r] = 1.0f;
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
                Alphasmem[row] = alpha;
                Msmem[row] = m_new;
                Lsmem[row] = l_new;
            }
        }

        __syncthreads();

        oaccSCALING<Br, D>(Oreg, Alphasmem);

        __syncthreads();

        mma_pv_accum_reg_f32p_f16v<Br, Bc, D>(
            Ssmem,
            Vsmem[curstage],
            Oreg,
            Br,
            Bc,
            D,
            Bc,
            V_STRIDE
        );
    }
    /// now we have the acc for O
    /// store it globally 
    #pragma unroll 1
    for (int t = 0; t < O_TILES_PER_WARP; t++)
    {
        int tile_idx = warp + t * WARPS;

        if (tile_idx >= TOTAL_O_TILES) continue;

        int mt = tile_idx / NUM_N_TILES;
        int nt = tile_idx % NUM_N_TILES;

        int row_start = mt * MMA_M;
        int col_start = nt * MMA_N;

        int r0 = row_start + group;
        int r1 = row_start + group + 8;

        int c0 = col_start + tid4 * 2;
        int c1 = c0 + 1;

        int global_r0 = rowid * Br + r0;
        int global_r1 = rowid * Br + r1;

        if (r0 < Br && global_r0 < N) {
            float inv_l0 = 1.0f / Lsmem[r0];

            if (c0 < D) {
                float out0 = Oreg[t * 4 + 0] * inv_l0;
                Optr[global_r0 * D + c0] = __float2half(out0);
            }

            if (c1 < D) {
                float out1 = Oreg[t * 4 + 1] * inv_l0;
                Optr[global_r0 * D + c1] = __float2half(out1);
            }
        }

        if (r1 < Br && global_r1 < N) {
            float inv_l1 = 1.0f / Lsmem[r1];

            if (c0 < D) {
                float out2 = Oreg[t * 4 + 2] * inv_l1;
                Optr[global_r1 * D + c0] = __float2half(out2);
            }

            if (c1 < D) {
                float out3 = Oreg[t * 4 + 3] * inv_l1;
                Optr[global_r1 * D + c1] = __float2half(out3);
            }
        }

        if (nt == 0 && tid4 == 0) {
            if (r0 < Br && global_r0 < N) {
                Lptr[global_r0] = Msmem[r0] + logf(Lsmem[r0]);
            }

            if (r1 < Br && global_r1 < N) {
                Lptr[global_r1] = Msmem[r1] + logf(Lsmem[r1]);
            }
        }
    }
    
}
