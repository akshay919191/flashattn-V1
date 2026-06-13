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
    int stride,
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
    a_frag[0] = *reinterpret_cast<const uint32_t*>(&smem_a[whichrow * 16 * stride + whichcol * 16 +  group      * stride + c0]);
    a_frag[1] = *reinterpret_cast<const uint32_t*>(&smem_a[whichrow * 16 * stride + whichcol * 16 + (group + 8) * stride + c0]);
    a_frag[2] = *reinterpret_cast<const uint32_t*>(&smem_a[whichrow * 16 * stride + whichcol * 16 +  group      * stride + c1]);
    a_frag[3] = *reinterpret_cast<const uint32_t*>(&smem_a[whichrow * 16 * stride + whichcol * 16 + (group + 8) * stride + c1]);

    b_frag[0] = (uint32_t(__half_as_ushort(smem_b[whichcol * 16 * stride + maincol * 8 + r0 * stride + group])) | uint32_t(__half_as_ushort(smem_b[whichcol * 16 * stride + maincol * 8 + (r0 + 1) * stride + group])) << 16);
    b_frag[1] = (uint32_t(__half_as_ushort(smem_b[whichcol * 16 * stride + maincol * 8 + r1 * stride + group])) | uint32_t(__half_as_ushort(smem_b[whichcol * 16 * stride + maincol * 8 + (r1 + 1) * stride + group])) << 16);
    
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
    const int     stride_,           // Global matrix row width  (N_GLOBAL = 256)
    const int     padstr_,           // Physical smem row width   (PADSTR  = 40)
    const int     float4s_per_tile,  // Total 16-byte vectors to load
    const int     M, const int N,    // Global matrix boundary limits
    const int     block_row_start,   // Tile row index  (0-based)
    const int     block_col_start    // Tile col index  (0-based)
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