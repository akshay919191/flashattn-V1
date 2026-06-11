/// cuda modules
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

/// for maths operation
#include <math.h>

/// for float control
#include <float.h>

/// for some cpp style 
#include <iostream>

/// for building it up
#include <torch/extension.h>

/// to include our prebuilt function from other file
#include "mma_helpers.cuh"

/// Pad size for accuratly handling odd sizes and no bank conflicts
#define PADDED_STRIDE 8

/// we will use Br and Bc same here 
template<int Br , int NUM_HEADS , int SEQLEN , int HEAD_DIM>
__global__ void flashattn_fwd_kernel(
    /// Query matrix
    const half* __restrict__ Q, 

    /// key matrix
    const half* __restrict__ K, 

    /// Value matrix
    const half* __restrict__ V,

    /// holder for final output
    half* __restrict__ output ,

    /// log exp sum for backward pass
    float* __restrict__ L_out 
)
{
    /// thread id's
    const int tid = threadIdx.x;

    /// warpping here , 32 threads makes one warp
    const int warpid = tid / 32;

    /// lane for always in under 32
    const int lane = tid % 32;

    /// grouping into 8 [0,1,2,3,4,5,6,7]
    const int group = lane / 4;

    /// batch id for tracking into different batches
    const int batchid = blockIdx.x;

    /// head id for tracking for differnt heads in a batch
    const int headid = blockIdx.y;

    /// now we need iterator over each head of size seq_len * headdim according to what we want 
    const int tileitr = blockIdx.z;

    /// now a ptr for global index using above definitions , so wee need stride for that
    const long long base = (long long)batchid * NUM_HEADS * SEQLEN * HEAD_DIM +
                           (long long)headid * SEQLEN * HEAD_DIM;

    /// now pointing each parameter to actualy head
    const __half* Qptr = Q + base;

    const __half* Kptr = K + base;

    const __half* Vptr = V + base;

     __half* Optr = output + base;

    /// now we will add another base to store the logexp sum , so the final matrix shape is SEQLEN * 1 so same as base just no HEADDIM
    const long long stat_base = (long long)batchid * NUM_HEADS * SEQLEN +
                                (long long)headid * SEQLEN;

    /// now we will point this too
    float* Lptr = L_out + stat_base;

    /// now allocating shared mem , we will use char based memory allocation for this

    extern __shared__ char smem_raw[];

    __half* smenA = reinterpret_cast<__half*>(smem_raw);

    __half* smenB   = smenA   + (16 * 16);         // MMA Fragment A workspace: [16 x 16]

    __half* qshared = smenB   + (16 * 8);          // MMA Fragment B workspace: [16 x 8]

    __half* kshared = qshared + (Br * Br);         // Q Block: [Br x Br]
    
    __half* vshared = kshared + (Br * Br);         // K Block: [Br x Br]

    float* output  = reinterpret_cast<float*>(vshared + (Br * Br)); // Output Accumulator: [Br x Br]

    float* alpha_s = l      + (Br * Br);                  // Renormalization scaling α: [Br]

    float* beta_s  = alpha_s + Br;                 // Renormalization scaling β: [Br]

    /// now we need to zero all the m and l here so no garbage value
    /// we are initializing it in threads so they live in register till end
    float m_i = -FLT_MAX;

    float l_i = 0.f;

    /// now we know are warpping here for 128 threads means 4 warp , so we will iterate over 4 tiles at once of size Br * Br , but it doesn't change
    /// the fact that we will iterate row by row so our main one itr will be tileitr and other will be SEQLEN / BR
    /// we will use 1 row of Q and iterate over all rows of K , because its Q @ K.T so instead of cols we do rows 

}   

