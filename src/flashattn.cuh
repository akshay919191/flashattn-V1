#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include "helper.cuh"


template<int Br, int Bc, int D_PAD , bool masked>  ///  D_PAD is headdim
__global__ void flashattn_fwd(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
          __half* __restrict__ output,
          float*  __restrict__ Logsum,
          int actual_D , int Skv , int Sq /// Skv is seq len for key and value , where Sq is for query 
)
{
    const int batchid = blockIdx.x;
    const int headid  = blockIdx.y;
    const int tileid  = blockIdx.z;
    const int numhead = gridDim.y ;

    /// now as strides are different so do for PTR's

    const long long Qstat = (long long)batchid * numhead * Sq * actual_D + 
                            (long long)headid  * Sq * actual_D;

    const long long KVstat = (long long)batchid * numhead * Skv * actual_D + 
                            (long long)headid  * Skv * actual_D;

    const long long statbase = (long long)batchid * numhead * Sq + headid * Sq;

    /// now ptrs towards data
    const __half* Qptr = Q + Qstat; const __half* Kptr = K + KVstat; const __half* Vptr = V + KVstat;

    /// shared mem allocate

    auto align = [&](char*& p , size_t a = 15){
        p = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(p) + a) & ~a);
    };

    extern __shared__ char smem[];
    char* ptr = smem;

    align(ptr);
    __half* Qsmem = reinterpret_cast<__half*>(ptr); ptr += Br * D_PAD * sizeof(__half);
    align(ptr);
    __half* Ksmem0 = reinterpret_cast<__half*>(ptr); ptr += Bc * D_PAD * sizeof(__half);
    align(ptr);
    __half* Ksmem1 = reinterpret_cast<__half*>(ptr); ptr += Bc * D_PAD * sizeof(__half);
    __half* Ksmem[2] = {Ksmem0, Ksmem1};
    align(ptr);
    __half* Vsmem0 = reinterpret_cast<__half*>(ptr); ptr += Bc * D_PAD * sizeof(__half);
    align(ptr);
    __half* Vsmem1 = reinterpret_cast<__half*>(ptr); ptr += Bc * D_PAD * sizeof(__half);
    __half* Vsmem[2] = {Vsmem0, Vsmem1};

    /// registers for Q and l , m
    /*
        total elements in O will be Br * D, 
        one mma give 4 elements , 2 different rows , so each threads should process 2 rows for max and log exp sum

        shape for O
        Br * D , means Br * D / 32(pre warp)
        and D / 8 , as output will be 16 * 8 per mma , means D / 8 , and each threads gives 4 output  means D / 8 * 4
        
        like 16 * 16 @ 16 * 8 -> 16 * 8
        16 * 8 -> 8 / 8 = 1 * 4 -> 4 * 32 = 128 -> 16 * 8 , 128 
    */

    const int tid   = threadIdx.x;
    const int warp  = tid / 32;
    const int lane  = tid & 31;

    const float SCALE = 1.0f / sqrtf((float)D);

    constexpr int Dk = Sq / 16;
    constexpr int Bk = Bc / 8;
    constexpr int Dv = Skv / 8;

    float O_frag[Dv * 4] = {0.0f};
    float m_frag[2] = {-FLT_MAX, -FLT_MAX};
    float l_frag[2] = {0.0f};

    const int Tr = (Sq + Br - 1) / Br;
    const int Tc = (Skv + Bc - 1) / Bc;

    if (tileid >= Tr) return;

    asyncLOAD_2D_TILE<Br , D_PAD , 128>(Qptr , smem_u32_ptr(Qsmem) , tid , D_PAD , Sq , actual_D , actual_D , tileid , 0);
    asyncLOAD_2D_TILE<Bc , D_PAD , 128>(Kptr , smem_u32_ptr(Ksmem[0]) , tid , D_PAD , Skv , actual_D , actual_D , 0 , 0);

    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    for(int kv_tile = 0 ; kv_tile < Tc ; kv_tile++){
        const int curstage  = kv_tile & 1;  /// % 2
        const int nextstage = curstage ^ 1; /// for 0 it gives 1 and 1 gives 2
        const int next_kv   = kv_tile + 1;

        if(next_kv < Tc){
            asyncLOAD_2D_TILE<Bc , D_PAD , 128>(Kptr , smem_u32_ptr(Ksmem[next_stage]) , tid , D_PAD , Skv , actual_D , actual_D , 0 , 0);
            asyncLOAD_2D_TILE<Bc , D_PAD , 128>(Vptr , smem_u32_ptr(Vsmem[curstage]) , tid , D_PAD , Skv , actual_D , actual_D , 0 , 0);
            asm volatile("cp.async.commit_group;\n");
        }

        S_frag[Bk * 4] = {0.f};

        for (int kb = 0; kb < Bk; kb++) {
            for (int ks = 0; ks < Dk; ks++) {
                uint32_t q_frag[4];
                int q_r = warp * 16 + (lane % 16);
                int q_c = ks * 16 + ((lane < 16) ? 0 : 8);
                uint32_t q_addr = smem_u32_ptr(Qsmem + q_r * D_PAD + q_c);
                ldmatrix_x4(q_frag, q_addr);

                uint32_t k_frag[2];
                const int lane16 = lane & 15;
                int k_r = kb * 8 + (lane16 & 7);
                int k_c = ks * 16 + ((lane16 >> 3) * 8);

                uint32_t k_addr =
                    smem_u32_ptr(Ksmem[curstage] + k_r * D_PAD + k_c);

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
        /// now it has some rough fake values too , as we have used padded headdim , we can pad it 
    }

}