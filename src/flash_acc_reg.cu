#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <iostream>
#include <vector>
#include <algorithm>

#include "helper.cuh"

#define CUDA_CHECK(call)                                                        \
do {                                                                            \
    cudaError_t err = (call);                                                   \
    if (err != cudaSuccess) {                                                   \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__            \
                  << " : " << cudaGetErrorString(err) << std::endl;             \
        std::exit(1);                                                           \
    }                                                                           \
} while (0)



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

template<int Br, int Bc, int D, int N>
void cpu_flash_ref_online_mma_like(
    const std::vector<__half>& Q,
    const std::vector<__half>& K,
    const std::vector<__half>& V,
    std::vector<float>& Oref,
    std::vector<float>& Lref
)
{
    const float scale = 1.0f / std::sqrt((float)D);
    const int Tc = (N + Bc - 1) / Bc;

    Oref.assign(N * D, 0.0f);
    Lref.assign(N, 0.0f);

    std::vector<float> m(N, -FLT_MAX);
    std::vector<float> l(N, 0.0f);

    for (int row = 0; row < N; row++) {
        for (int kv_tile = 0; kv_tile < Tc; kv_tile++) {
            int kv_start = kv_tile * Bc;
            int valid = std::min(Bc, N - kv_start);

            float m_val = -FLT_MAX;

            for (int j = 0; j < valid; j++) {
                int col = kv_start + j;
                float s = 0.0f;

                for (int d = 0; d < D; d++) {
                    s += __half2float(Q[row * D + d]) * __half2float(K[col * D + d]);
                }

                s *= scale;
                m_val = std::max(m_val, s);
            }

            float l_val = 0.0f;
            std::vector<float> scores(valid);

            for (int j = 0; j < valid; j++) {
                int col = kv_start + j;
                float s = 0.0f;

                for (int d = 0; d < D; d++) {
                    s += __half2float(Q[row * D + d]) * __half2float(K[col * D + d]);
                }

                s *= scale;
                scores[j] = s;
                l_val += std::exp(s - m_val);
            }

            float m_new = std::max(m[row], m_val);
            float alpha = std::exp(m[row] - m_new);
            float beta  = std::exp(m_val - m_new);

            // Same as kernel: scale old O numerator by alpha.
            for (int d = 0; d < D; d++) {
                Oref[row * D + d] *= alpha;
            }

            // Kernel converts P float to half before tensor-core PV.
            for (int j = 0; j < valid; j++) {
                int col = kv_start + j;
                float p = beta * std::exp(scores[j] - m_val);
                float p_half = __half2float(__float2half(p));

                for (int d = 0; d < D; d++) {
                    Oref[row * D + d] += p_half * __half2float(V[col * D + d]);
                }
            }

            l[row] = alpha * l[row] + beta * l_val;
            m[row] = m_new;
        }

        for (int d = 0; d < D; d++) {
            Oref[row * D + d] /= l[row];
        }

        Lref[row] = m[row] + std::log(l[row]);
    }
}

int main()
{
    constexpr int Br = 32;
    constexpr int Bc = 16;
    constexpr int NUM_HEADS = 1;
    constexpr int D  = 128;
    constexpr int N  = 2048;

    std::vector<__half> hQ(N * D);
    std::vector<__half> hK(N * D);
    std::vector<__half> hV(N * D);
    std::vector<__half> hO(N * D);
    std::vector<float> hLogsum(N);

    for (int i = 0; i < N * D; i++) {
        float q = ((i % 29) - 14) * 0.025f;
        float k = ((i % 31) - 15) * 0.020f;
        float v = ((i % 17) - 8)  * 0.030f;

        hQ[i] = __float2half(q);
        hK[i] = __float2half(k);
        hV[i] = __float2half(v);
    }

    __half *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dO = nullptr;
    float* dLogsum = nullptr;

    CUDA_CHECK(cudaMalloc(&dQ, N * D * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dK, N * D * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dV, N * D * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dO, N * D * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dLogsum, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), N * D * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK.data(), N * D * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV.data(), N * D * sizeof(__half), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(dO, 0, N * D * sizeof(__half)));
    CUDA_CHECK(cudaMemset(dLogsum, 0, N * sizeof(float)));

    constexpr int PAD = 8;

    constexpr int Q_STRIDE = D + PAD;
    constexpr int K_STRIDE = D + PAD;
    constexpr int V_STRIDE = D + PAD;
    constexpr int S_STRIDE = Bc;

    size_t smem_size =
        Br * Q_STRIDE * sizeof(__half) +          // Qsmem
        2 * Bc * K_STRIDE * sizeof(__half) +      // K double buffer
        2 * Bc * V_STRIDE * sizeof(__half) +      // V double buffer
        Br * S_STRIDE * sizeof(float) +           // S/P
        Br * sizeof(float) +                      // M
        Br * sizeof(float) +                      // L
        Br * sizeof(float) +                      // Alpha
        512;                                      // alignment slack

    std::cout << "smem_size = " << smem_size / 1024.0 << " KB\n";

    CUDA_CHECK(cudaFuncSetAttribute(
        flashattn_fwd_kernel<Br, Bc, NUM_HEADS, D, N>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_size
    ));

    dim3 block(128);

    // kernel expects:
    // blockIdx.x = batch
    // blockIdx.y = head
    // blockIdx.z = row block
    dim3 grid(
        1,
        NUM_HEADS,
        (N + Br - 1) / Br
    );

    flashattn_fwd_kernel<Br, Bc, NUM_HEADS, D, N>
        <<<grid, block, smem_size>>>(
            dQ,
            dK,
            dV,
            dO,
            dLogsum
        );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hO.data(), dO, N * D * sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hLogsum.data(), dLogsum, N * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> refO;
    std::vector<float> refL;

    cpu_flash_ref_online_mma_like<Br, Bc, D, N>(
        hQ,
        hK,
        hV,
        refO,
        refL
    );

    float max_o_err = 0.0f;
    float max_l_err = 0.0f;
    int bad_o = 0;
    int bad_l = 0;

    for (int i = 0; i < N * D; i++) {
        float got = __half2float(hO[i]);
        float exp = refO[i];
        float diff = std::fabs(got - exp);

        max_o_err = std::max(max_o_err, diff);

        if (diff > 5e-2f) {
            if (bad_o < 10) {
                int r = i / D;
                int c = i % D;

                std::cout << "O mismatch (" << r << "," << c << "): "
                          << "GPU=" << got
                          << " CPU=" << exp
                          << " diff=" << diff << "\n";
            }

            bad_o++;
        }
    }

    for (int i = 0; i < N; i++) {
        float diff = std::fabs(hLogsum[i] - refL[i]);

        max_l_err = std::max(max_l_err, diff);

        if (diff > 5e-3f) {
            if (bad_l < 10) {
                std::cout << "L mismatch row " << i << ": "
                          << "GPU=" << hLogsum[i]
                          << " CPU=" << refL[i]
                          << " diff=" << diff << "\n";
            }

            bad_l++;
        }
    }

    std::cout << "max_o_err = " << max_o_err << "\n";
    std::cout << "max_l_err = " << max_l_err << "\n";

    if (bad_o == 0 && bad_l == 0) {
        std::cout << "PASSED\n";
    } else {
        std::cout << "FAILED bad_o=" << bad_o << " bad_l=" << bad_l << "\n";
    }

    CUDA_CHECK(cudaFree(dQ));
    CUDA_CHECK(cudaFree(dK));
    CUDA_CHECK(cudaFree(dV));
    CUDA_CHECK(cudaFree(dO));
    CUDA_CHECK(cudaFree(dLogsum));

    return (bad_o == 0 && bad_l == 0) ? 0 : 1;
}