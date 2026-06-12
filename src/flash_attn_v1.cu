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
#include "helpers.cuh"

/// Pad size for accuratly handling odd sizes and no bank conflicts
#define PADDED_STRIDE 8

/// this is to show how much buffering , we are going for triple
#define STAGES 3

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
    /// and we are uysing triple buffer

    extern __shared__ char smem_raw[];
    
    __half* k_base = reinterpret_cast<__half*>(smem_raw);

    const int TILE_ELEMENTS = Br * (HEAD_DIM + PADDED_STRIDE);
    
    __half* v_base = k_base + (TILE_ELEMENTS * STAGES);

    __half* k_stages[STAGES] = {
        k_base,
        k_base + TILE_ELEMENTS,
        k_base + (TILE_ELEMENTS * 2)
    };

    __half* v_stages[STAGES] = {
        v_base,
        v_base + TILE_ELEMENTS,
        v_base + (TILE_ELEMENTS * 2)
    };

    float* out = reinterpret_cast<float*>(v_base + (TILE_ELEMENTS * STAGES));

    /// now we need to zero all the m and l here so no garbage value
    /// we are initializing it in threads so they live in register till end
    float m_i = -FLT_MAX;

    float l_i = 0.f;

    float r_alpha = 1.0f;      // Local rescaling factor alpha

    float r_beta  = 1.0f;      // Local rescaling factor beta

    /// now we know are warpping here for 128 threads means 4 warp , so we will iterate over 4 tiles at once of size Br * Br , but it doesn't change
    /// the fact that we will iterate row by row so our main one itr will be tileitr and other will be SEQLEN / BR
    /// we will use 1 row of Q and iterate over all rows of K , because its Q @ K.T so instead of cols we do rows 

    /// row iterator for Q
    const int Qrowitr = tileitr;

    /// now we will do Q @ K.T so SEQLEN * HEAD_DIM @ HEAD_DIM * SEQLEN -> SEQLEN * SEQLEN
    /// 1 row from Q and all rows from K forms 1 final row , and Q next row forms next and so on.........

    /// to match our tile size we will iterate over full len in small chunks of size TILE means Br
    const int Krowitr = SEQLEN / Br;

    /// now in each row we will iterate over 4 Br * Br blocks means we have to check how much we can iterate in a row
    /// this is how much we iterate in a row for K and Q
    const int warpcolitr = (HEAD_DIM + 4 * Br - 1) / (4 * Br);

    /// but we will load Q in registers only so we have to iterate over size wille be Br * HEADDIM
    /// if HEADDIM exceeds or equal to 256 we drop our Br to 16 else 64 and max we can go is 256 as headdim 

    /// now we need to make a uint32_t register for Q to remain in registers
    /// calculations we have Br * HEADDIM elements and 2 bytes each
    /// formula is (total elements) * size_of(data_type) / total_threads * 4

    const int Q_ELEMENTS_PER_THREAD = (Br * HEAD_DIM) / blockDim.x;
    const int Q_REGISTERS_PER_THREAD = (Q_ELEMENTS_PER_THREAD * sizeof(__half)) / 4; 

    uint32_t q_reg[Q_REGISTERS_PER_THREAD];
    
    /// now we will load __half2 elements because each threads load , so we are using uint32_t as global standard so 32 bit , means 2 half element 
    __half2* global_q_half2 = reinterpret_cast<__half2*>(const_cast<__half*>(Qptr) + (Qrowitr * Br * HEAD_DIM));

    /// total half2 elements
    const int total_half2 = Br * HEAD_DIM / 2;

    /// we will use pragma unroll so it can happen wihout waiting
    /// so each threads have Q_REGISTERS_PER_THREAD this much registers and each warp first takes total threads * 2 element in rg[0] and then next in reg[1] so this way threads are collesced not registers
    #pragma unroll
    for(int regIDX = 0 ; regIDX < Q_REGISTERS_PER_THREAD ; regIDX++)
    {
        int global_idx = (regIDX * blockDim.x) + tid;

        if(global_idx < total_half2) {
            q_reg[regIDX] = *reinterpret_cast<uint32_t*>(&global_q_half2[global_idx]);
        }
        else {
            q_reg[regIDX] = 0; 
        }
    }

    __syncthreads();

    /// now we have Q in our register 
    /// thing to be considered Q has a shape of SEQLEN * HEAD_DIMso anyway i am covering whole cols of each row , so i can preload it once instead doing it in a loop
    
    /// this tells the number of row iteration in K.T
    const int totalrow = (SEQLEN + HEAD_DIM - 1) / HEAD_DIM; 

    /// this tells the total col iteration that could happen inside a row of K.T
    const int totalcol = (HEAD_DIM + Br - 1) / Br;


    

}   

