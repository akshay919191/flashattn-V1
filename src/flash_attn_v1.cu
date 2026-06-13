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
// #include <torch/extension.h>

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
    const int warp = tid / 32;

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

    constexpr int Q_REGS = ((Br * HEAD_DIM) / 128) / 2; /// 128 are number of threads
    uint32_t q_reg[Q_REGS];
    
    /// now we will load __half2 elements because each threads load , so we are using uint32_t as global standard so 32 bit , means 2 half element 
    __half2* global_q_half2 = reinterpret_cast<__half2*>(const_cast<__half*>(Qptr) + (Qrowitr * Br * HEAD_DIM));

    /// total half2 elements
    const int total_half2 = Br * HEAD_DIM / 2;
    const int q_row_base = warp * 16;


    const int Tk = HEAD_DIM / 16;
    const int Tn = Br / 8;
    const int SMEM_STRIDE = HEAD_DIM + 8;

    /// we will use pragma unroll so it can happen wihout waiting
    /// so each threads have Q_REGISTERS_PER_THREAD this much registers and each warp first takes total threads * 2 element in rg[0] and then next in reg[1] so this way threads are collesced not registers
    #pragma unroll
    for (int k = 0; k < Tk; k++) {
        int row0 = q_row_base + (lane / 4) + Qrowitr * Br;      
        int row1 = row0 + 8;                     
        int col0 = k * 16 + (lane % 4) * 2;      
        int col1 = col0 + 8;                     

        int base = k * 4;
        q_reg[base+0] = *reinterpret_cast<const uint32_t*>(&Qptr[row0 * HEAD_DIM + col0]);
        q_reg[base+1] = *reinterpret_cast<const uint32_t*>(&Qptr[row1 * HEAD_DIM + col0]);
        q_reg[base+2] = *reinterpret_cast<const uint32_t*>(&Qptr[row0 * HEAD_DIM + col1]);
        q_reg[base+3] = *reinterpret_cast<const uint32_t*>(&Qptr[row1 * HEAD_DIM + col1]);
    }
    __syncthreads();

    /// now we have Q in our register 
    /// thing to be considered Q has a shape of SEQLEN * HEAD_DIMso anyway i am covering whole cols of each row

    /// this tells the number of row iteration in K.T
    /// V shape is HEAD_DIM * BR
    const int totalrow = (SEQLEN + HEAD_DIM - 1) / HEAD_DIM; 

    /// this tells the total col iteration that could happen inside a row of K.T
    const int numtiles = (HEAD_DIM + Br - 1) / Br;

    /// now we will use float4 to load so i need to find number of float4 in V one block 
    /// size if HEAD_DIM * BR
    const int float4s_per_tile = (HEAD_DIM * Br) / 8;
    const int Matrix_row       = SEQLEN;
    const int Matrix_col       = HEAD_DIM;

    /// loading slot 0
    uint32_t smenptr0 = __cvta_generic_to_shared(k_stages[0]);
    asyncLOAD<HEAD_DIM, Br, 128>(
        Kptr,
        smenptr0,
        tid,
        HEAD_DIM,
        (HEAD_DIM + PADDED_STRIDE),
        float4s_per_tile,
        Matrix_row , Matrix_col,
        0 , 0
        );
    
    /// loading slot 1
    /// do a safety check if we have more slots to load or not
    uint32_t smenptr1 = __cvta_generic_to_shared(k_stages[1]);
    asyncLOAD<HEAD_DIM, Br, 128>(
        Kptr,
        smenptr1,
        tid,
        HEAD_DIM,
        (HEAD_DIM + PADDED_STRIDE),
        float4s_per_tile,
        Matrix_row , Matrix_col,
        0 , 1
        );

    uint32_t ptr0 = __cvta_generic_to_shared(v_stages[0]);
    asyncLOAD<HEAD_DIM, Br, 128>(
        Vptr,
        ptr0,
        tid,
        HEAD_DIM,
        (HEAD_DIM + PADDED_STRIDE),
        float4s_per_tile,
        Matrix_row , Matrix_col,
        0 , 0
        );
    
    /// loading slot 1
    /// do a safety check if we have more slots to load or not
    uint32_t ptr1 = __cvta_generic_to_shared(v_stages[1]);
    asyncLOAD<HEAD_DIM, Br, 128>(
        Vptr,
        ptr1,
        tid,
        HEAD_DIM,
        (HEAD_DIM + PADDED_STRIDE),
        float4s_per_tile,
        Matrix_row , Matrix_col,
        0 , 1
        );
    
    /// commit all the into pipeline
    asm volatile("cp.async.commit_group;\n");

    /// wait for all streams to be finished
    asm volatile("cp.async.wait_all;\n");

    /// no need ig because of wait but still a safe fallback
    __syncthreads();


    for(int rowid = 0 ; rowid < totalrow ; rowid++)
    {
        /// now we iterating over rows
        /// now we will iterate over cols in ech row
        for(int tile = 0 ; tile < numtiles ; tile++)
        {
            /// now the real thing we need triple load here 
            /// 2 loaded already 

            int cur  = (tile % STAGES); /// cur means current slot in shared mem not actual tile id
            int next = (tile + 2);
            int pre  = (tile + 2) % STAGES; /// which we need to load now

            if(next < numtiles)
            {
                uint32_t smenptr = __cvta_generic_to_shared(k_stages[pre]);
                asyncLOAD<HEAD_DIM, Br, 128>(
                    Kptr,
                    smenptr,
                    tid,
                    HEAD_DIM,
                    (HEAD_DIM + PADDED_STRIDE),
                    float4s_per_tile,
                    Matrix_row , Matrix_col,
                    rowid , next
                    );

                /// we gonna commit to our pipeline
                asm volatile("cp.async.commit_group;\n");
            }

            /// now we have to wait for each stream line to complete
            if (tile < numtiles - 2) {
                asm volatile("cp.async.wait_group 2;\n" ::: "memory");
            } 
            else if (tile == numtiles - 2) {
                asm volatile("cp.async.wait_group 1;\n" ::: "memory");
            } 
            else {
                asm volatile("cp.async.wait_group 0;\n" ::: "memory");
            }

            /// we are syncing for a safety fallback
            __syncthreads();

            /// now we have loaded all at fast speed 

            /// now we have slots we need mma PTX assembly mma.sync to multiply matrix 
            /// we have function for that too , now we need to maop the maths because our function is for m16n8k16 and our actual shapes are HEAD_DIM * Br

            /// as our shaped are large than required we will use looping
            /// here is some indexing for looping over each slot for inline asm PTX

            /// reason i am not using variable si to get saved from register spilling

            /// reason behind hardcoding to 16 is mma.sync works on 16*8*16 here and usign Br is essential for Q
            // const int Tr = Br / 16;

            /// its for col iteration in a row of Query and row iter fro a single col for Key.Transpose
            // const int Tc = HEAD_DIM / 16;

            /// this is for ech col iteration in a row of key.Transpose 
            // const int cc = Br / 8;

            float acc[4][4] = {{0.f}};

            #pragma unroll
            for (int nc = 0; nc < Tn; nc++) {
                #pragma unroll
                for (int kc = 0; kc < Tk; kc++) {
                    m16n8k16_reg(
                        acc[nc][0], acc[nc][1], acc[nc][2], acc[nc][3],
                        &q_reg[kc * 4],
                        k_stages[cur],
                        lane, group,
                        SMEM_STRIDE,
                        kc,
                        nc
                    );
                }
            }

            {
            int row0 = Qrowitr * Br + q_row_base + (lane / 4);
            int row1 = row0 + 8;
            #pragma unroll
            for (int nc = 0; nc < Tn; nc++) 
                {
                int col0 = nc * 8 + (lane % 4) * 2;
                int col1 = col0 + 1;

                out[row0 * Br + col0] = acc[nc][0];
                out[row0 * Br + col1] = acc[nc][1];
                out[row1 * Br + col0] = acc[nc][2];
                out[row1 * Br + col1] = acc[nc][3];
                }
            }
        }
        /// here 
    }
}   

