#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <iostream>

// ── constants ─────────────────────────────────────────────────────────────────
#define BR       32     // Q rows
#define HEAD_DIM 256    // Q cols = K cols
#define BC       32     // K rows (one tile)
#define THREADS  128    // 4 warps

// ── mma function — Q in registers, K in smem ─────────────────────────────────
__device__ __forceinline__ void m16n8k16_reg(
    float& d1, float& d2, float& d3, float& d4,
    const uint32_t* a_frag,
    const __half*   smem_b,
    const int lane,
    const int group,
    int stride,
    int whichcol,
    int maincol
) {
    uint32_t b_frag[2];

    int r0 = (lane % 4) * 2;
    int r1 = r0 + 8;

    int k0 = whichcol * 16 + (lane % 4) * 2;   // k-dim, first pair
    int k1 = k0 + 8;                             // k-dim, second pair
    int n  = maincol * 8 + (lane / 4);          // n-dim = K row

    // pack 2 elements from same K row, adjacent k cols
    b_frag[0] = (uint32_t(__half_as_ushort(smem_b[n * stride + k0    ]))        |
                (uint32_t(__half_as_ushort(smem_b[n * stride + k0 + 1])) << 16));

    b_frag[1] = (uint32_t(__half_as_ushort(smem_b[n * stride + k1    ]))        |
                (uint32_t(__half_as_ushort(smem_b[n * stride + k1 + 1])) << 16));

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

__global__ void qk_mma_kernel(
    const __half* __restrict__ Q_gmem,
    const __half* __restrict__ K_gmem,
    float*        __restrict__ C_gmem
) {
    const int tid   = threadIdx.x;
    const int lane  = tid % 32;
    const int warp  = tid / 32;
    const int group = lane / 4;

    const int SMEM_STRIDE = HEAD_DIM + 8;
    __shared__ __half smem_k[BC][HEAD_DIM + 8];

    // load K into smem — all 128 threads
    for (int i = tid; i < BC * HEAD_DIM; i += THREADS) {
        int r = i / HEAD_DIM;
        int c = i % HEAD_DIM;
        smem_k[r][c] = K_gmem[r * HEAD_DIM + c];
    }
    __syncthreads();

    const int Tk = HEAD_DIM / 16;
    const int Tn = BC / 8;

    const int q_row_base = warp * 16;

    constexpr int Q_REGS = (HEAD_DIM / 16) * 4;
    uint32_t q_reg[Q_REGS];

    #pragma unroll
    for (int k = 0; k < Tk; k++) {
        int row0 = q_row_base + (lane / 4);      // rows 0-7 of this warp's block
        int row1 = row0 + 8;                      // rows 8-15 of this warp's block
        int col0 = k * 16 + (lane % 4) * 2;      // first k-pair
        int col1 = col0 + 8;                      // second k-pair

        int base = k * 4;
        q_reg[base+0] = *reinterpret_cast<const uint32_t*>(&Q_gmem[row0 * HEAD_DIM + col0]);
        q_reg[base+1] = *reinterpret_cast<const uint32_t*>(&Q_gmem[row1 * HEAD_DIM + col0]);
        q_reg[base+2] = *reinterpret_cast<const uint32_t*>(&Q_gmem[row0 * HEAD_DIM + col1]);
        q_reg[base+3] = *reinterpret_cast<const uint32_t*>(&Q_gmem[row1 * HEAD_DIM + col1]);
    }

    float acc[4][4] = {{0.f}};

    #pragma unroll
    for (int nc = 0; nc < Tn; nc++) {
        #pragma unroll
        for (int kc = 0; kc < Tk; kc++) {
            m16n8k16_reg(
                acc[nc][0], acc[nc][1], acc[nc][2], acc[nc][3],
                &q_reg[kc * 4],
                &smem_k[0][0],
                lane, group,
                SMEM_STRIDE,
                kc,
                nc
            );
        }
    }

    {
        int row0 = warp * 16 + (lane / 4);  
        int row1 = row0 + 8;

        #pragma unroll
        for (int nc = 0; nc < Tn; nc++) {
            int col0 = nc * 8 + (lane % 4) * 2;
            int col1 = col0 + 1;

            C_gmem[row0 * BC + col0] = acc[nc][0];
            C_gmem[row0 * BC + col1] = acc[nc][1];
            C_gmem[row1 * BC + col0] = acc[nc][2];
            C_gmem[row1 * BC + col1] = acc[nc][3];
        }
    }
}

// CPU reference must match exactly what GPU computes: C = Q @ K^T
void cpu_matmul_AT(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int r = 0; r < M; r++)
        for (int c = 0; c < N; c++) {
            float acc = 0.f;
            for (int k = 0; k < K; k++)
                acc += A[r * K + k] * B[c * K + k];  // B[c][k] = B^T[k][c]
            C[r * N + c] = acc;
        }
}

// ── main ──────────────────────────────────────────────────────────────────────
int main() {
    // host buffers
    __half* h_Q   = new __half[BR  * HEAD_DIM];
    __half* h_K   = new __half[BC  * HEAD_DIM];
    float*  h_C   = new float [BR  * BC];
    float*  h_Qf  = new float [BR  * HEAD_DIM];
    float*  h_Kf  = new float [BC  * HEAD_DIM];
    float*  h_ref = new float [BR  * BC];

    // init — small values to avoid fp16 overflow
    srand(42);
    for (int i = 0; i < BR * HEAD_DIM; i++) {
        float v = ((float)(rand() % 16) - 8.f) * 0.125f;  // -1.0 to 1.0, fits fp16
        h_Q[i]  = __float2half(v);
        h_Qf[i] = v;
    }
    for (int i = 0; i < BC * HEAD_DIM; i++) {
        float v = ((float)(rand() % 16) - 8.f) * 0.125f;
        h_K[i]  = __float2half(v);
        h_Kf[i] = v;
    }

// call it as:
cpu_matmul_AT(h_Qf, h_Kf, h_ref, BR, BC, HEAD_DIM);

    // device
    __half *d_Q, *d_K;
    float  *d_C;
    cudaMalloc(&d_Q, BR  * HEAD_DIM * sizeof(__half));
    cudaMalloc(&d_K, BC  * HEAD_DIM * sizeof(__half));
    cudaMalloc(&d_C, BR  * BC       * sizeof(float));
    cudaMemset(d_C, 0,  BR * BC * sizeof(float));

    cudaMemcpy(d_Q, h_Q, BR  * HEAD_DIM * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, BC  * HEAD_DIM * sizeof(__half), cudaMemcpyHostToDevice);

    // launch — 1 block, 128 threads (4 warps)
    qk_mma_kernel<<<1, THREADS>>>(d_Q, d_K, d_C);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    cudaMemcpy(h_C, d_C, BR * BC * sizeof(float), cudaMemcpyDeviceToHost);

    // verify
    int   errors  = 0;
    float max_err = 0.f;
    std::cout << "\n--- First 4x4 of C ---\n";
    for (int r = 0; r < BR; r++) {
        for (int c = 0; c < BC; c++) {
            float got  = h_C [r * BC + c];
            float exp  = h_ref[r * BC + c];
            float diff = fabsf(got - exp);
            if (diff > max_err) max_err = diff;
            if (diff > 0.5f)    errors++;   // fp16 accumulation has some error
            if (r < 4 && c < 4)
                printf("C[%d][%d] exp=%.4f got=%.4f\n", r, c, exp, got);
        }
    }

    std::cout << "\n--- Diagonal of C (should all be 1.0) ---\n";
for (int r = 0; r < min(BR, BC); r++) {
    float got = h_C[r * BC + r];
    float exp = h_ref[r * BC + r];
    printf("C[%d][%d] exp=%.4f got=%.4f\n", r, r, exp, got);
}

// also print full error locations
std::cout << "\n--- All errors ---\n";
for (int r = 0; r < BR; r++)
    for (int c = 0; c < BC; c++) {
        float got  = h_C[r * BC + c];
        float exp  = h_ref[r * BC + c];
        if (fabsf(got - exp) > 0.1f)
            printf("C[%d][%d] exp=%.4f got=%.4f\n", r, c, exp, got);
    }

    printf("\nMax error: %.4f\n", max_err);
    if (errors == 0)
        printf("SUCCESS: 0 errors\n");
    else
        printf("FAILURE: %d errors\n", errors);

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_C);
    delete[] h_Q; delete[] h_K; delete[] h_C;
    delete[] h_Qf; delete[] h_Kf; delete[] h_ref;
    return 0;
}