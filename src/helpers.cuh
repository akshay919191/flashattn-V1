#ifndef MMA_HELPERS_CUH
#define MMA_HELPERS_CUH

#include <cuda_fp16.h>
#include <stdint.h>

#define WARP_FULL_MASK 0xffffffff

/// we are defining mma inline ptx here so we can reuse and each thread writes in own acc

__device__ __forceinline__ void m16n8k16(
    float& d1, float& d2, float& d3, float& d4,
    const __half* smem_a,
    const __half* smem_b,
    const int lane,
    const int group,
    int strideA,
    int strideB,
    int whichrow , int whichcol , int maincol/// this tells which we need from Br * Br
)
{

    int c0 = (lane % 4) * 2;
    int c1 = c0 + 8;

    int r0 = (lane % 4) * 2;
    int r1 = r0 + 8;

    /// we will use lane and create group
    uint32_t a_frag[4];
    uint32_t b_frag[2];

    /// uint32_t fragments to store floats in fragment for mma
    a_frag[0] = *reinterpret_cast<const uint32_t*>(&smem_a[whichrow * 16 * strideA + whichcol * 16 +  group      * strideA + c0]);
    a_frag[1] = *reinterpret_cast<const uint32_t*>(&smem_a[whichrow * 16 * strideA + whichcol * 16 + (group + 8) * strideA + c0]);
    a_frag[2] = *reinterpret_cast<const uint32_t*>(&smem_a[whichrow * 16 * strideA + whichcol * 16 +  group      * strideA + c1]);
    a_frag[3] = *reinterpret_cast<const uint32_t*>(&smem_a[whichrow * 16 * strideA + whichcol * 16 + (group + 8) * strideA + c1]);

    b_frag[0] = (uint32_t(__half_as_ushort(smem_b[whichcol * 16 * strideB + maincol * 8 + r0 * strideB + group])) | uint32_t(__half_as_ushort(smem_b[whichcol * 16 * strideB + maincol * 8 + (r0 + 1) * strideB + group])) << 16);
    b_frag[1] = (uint32_t(__half_as_ushort(smem_b[whichcol * 16 * strideB + maincol * 8 + r1 * strideB + group])) | uint32_t(__half_as_ushort(smem_b[whichcol * 16 * strideB + maincol * 8 + (r1 + 1) * strideB + group])) << 16);
    
    __syncthreads();

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},"
        "{%4,%5,%6,%7},"
        "{%8,%9},"
        "{%10,%11,%12,%13};"
        :"=f"(d1) , "=f"(d2) , "=f"(d3) , "=f"(d4)
        : "r"(a_frag[0]) , "r"(a_frag[1]) , "r"(a_frag[2]) , "r"(a_frag[3]) , 
          "r"(b_frag[0]) , "r"(b_frag[1]),
          "f"(d1) , "f"(d2) , "f"(d3) , "f"(d4)
    );
}

template<int Br, int headdim>
__device__ __forceinline__ void matmul_tile(const __half* __restrict__ A,
                             const __half* __restrict__ B,
                                   float*  __restrict__ C)
{
    extern __shared__ __half smem[];
    __half* smem_a = smem;           
    __half* smem_b = smem + 64 * Br; 
    const int tid     = threadIdx.x;
    const int warp = tid / 32;     
    const int lane    = tid % 32;
    const int group   = lane / 4;

    const int block_m_base = blockIdx.x * 4; 

    for (int i = tid; i < 64 * Br; i += blockDim.x)
    {
        int row = i / Br;
        int col = i % Br;
        smem_a[i] = A[(block_m_base * 16 + row) * Br + col];
    }

    for (int i = tid; i < Br * headdim; i += blockDim.x)
        smem_b[i] = B[i];

    __syncthreads();

    constexpr int kTiles = Br / 16;
    constexpr int nTiles = headdim / 8;

    __half* my_smem_a = smem_a + warp * 16 * Br;

    for (int nt = 0; nt < nTiles; ++nt)
    {
        float d1 = 0.f, d2 = 0.f, d3 = 0.f, d4 = 0.f;

        for (int kt = 0; kt < kTiles; ++kt)
        {
            m16n8k16(d1, d2, d3, d4,
                     my_smem_a, smem_b,
                     lane, group,
                     /*strideA=*/Br, /*strideB=*/headdim,
                     /*whichrow=*/0, /*whichcol=*/kt, /*maincol=*/nt);
        }

        int c0      = (lane % 4) * 2;
        int rowbase = (block_m_base + warp) * 16;
        int colbase = nt * 8;

        C[(rowbase + group)     * headdim + colbase + c0]     = d1;
        C[(rowbase + group)     * headdim + colbase + c0 + 1] = d2;
        C[(rowbase + group + 8) * headdim + colbase + c0]     = d3;
        C[(rowbase + group + 8) * headdim + colbase + c0 + 1] = d4;
    }
}

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

    int k0 = whichcol * 16 + (lane % 4) * 2;   
    int k1 = k0 + 8;                            
    int n  = maincol * 8 + (lane / 4);         

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


/// async load
template <int _TILE_M, int _TILE_N, int _BLOCK_DIM_X>
__device__ __forceinline__ void asyncLOAD(
    const __half* matA,
    uint32_t      smem_a,
    int           tid,
    const int     stride_,           // Global matrix row width  
    const int     padstr_,           // Physical smem row width   
    const int     float4s_per_tile,  // Total 16-byte vectors to load
    const int     M, const int N,    // Global matrix boundary limits
    const int     block_row_start,   // Tile row index 
    const int     block_col_start    // Tile col index  
)
{
    const int elements_per_vector = 8; // 16 bytes / sizeof(__half)

    #pragma unroll
    for (int i = tid; i < float4s_per_tile; i += _BLOCK_DIM_X)
    {
        int total_element_offset = i * elements_per_vector;
        int local_row = total_element_offset / _TILE_N;
        int local_col = total_element_offset % _TILE_N;

        int absolute_global_row = block_row_start * _TILE_M + local_row;
        int absolute_global_col = block_col_start * _TILE_N + local_col;


        uint32_t dst_smem_addr = smem_a +
                                 (local_row * padstr_ + local_col) * (uint32_t)sizeof(__half);

        bool is_valid = (absolute_global_row < M) && (absolute_global_col < N);
        const __half* src_global_ptr =
            matA + (size_t)absolute_global_row * stride_ + absolute_global_col;

        int predicate = is_valid ? 1 : 0;

        asm volatile(
            "{\n"
            "  .reg .pred p;\n"
            "  setp.ne.b32 p, %2, 0;\n"
            "  @p cp.async.cg.shared.global [%0], [%1], 16;\n"
            "  @!p st.shared.v4.b32 [%0], {0, 0, 0, 0};\n"
            "}\n"
            :
            : "r"(dst_smem_addr), "l"(src_global_ptr), "r"(predicate)
            : "memory"
        );
    }
}


/// for softmax reduction
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(WARP_FULL_MASK, val, offset));
    }
    return val; 
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(WARP_FULL_MASK, val, offset);
    }
    return val; 
}


#endif // MMA_HELPERS_CUH