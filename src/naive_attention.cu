#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <iostream>
#include <torch/extension.h>
    


template<int HEAD_DIM, int Br, int Bc , int NUM_HEADS>
__global__ void forwardFlashAttention(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ out,
    float* __restrict__ L_global, 
    int L, int D,
    int seqlen, int headdim 
)
{
    __align__(16) extern __shared__ __half smem[]; 

    __half* smenA   = smem;                              
    __half* smenB   = smenA   + 16 * 16;                 
    __half* qshared = smenB   + 16 * 8;                  
    __half* kshared = qshared + Br * Bc;                 
    __half* vshared = kshared + Bc * Bc;                 
    float* output  = (float*)(vshared + Br * HEAD_DIM); 
    float* m = (float*)(output + Br * Bc);              
    float* l = m + Br;                                   
    float* alpha_s = l + Br;                             
    float* beta_s  = alpha_s + Br;         
    
    const float scale = 1.0f / sqrtf((float)HEAD_DIM);
                            
    const int batchid = blockIdx.x;
    const int headid  = blockIdx.y;
    const int tileid  = blockIdx.z;

    const int tid     = threadIdx.x;

    const int lane  = tid % 32;
    const int group = lane / 4;

    const long long base = (long long)batchid * NUM_HEADS * seqlen * headdim +
                           (long long)headid  * seqlen * headdim;

    const long long stats_base = (long long)batchid * NUM_HEADS * seqlen +
                                 (long long)headid  * seqlen;

    const __half* Qptr = Q + base;
    const __half* Kptr = K + base;
    const __half* Vptr = V + base;
          __half* optr = out + base;
          float* Lptr = L_global + stats_base;

    for(int i = tid ; i < Br ; i += blockDim.x) // we are doing row wise softmax for Br * Bc so Br rows
        m[i] = -FLT_MAX , l[i] = 0.f;

    __syncthreads();

    const int rowtileid = tileid;
    const int total  = seqlen / Bc;  
    const int colitr = headdim / Bc;
    
    for(int rowid = 0 ; rowid < total ; rowid++)
    {
        for(int i = tid ; i < Br * Bc ; i += blockDim.x)
            output[i] = 0.f;
        __syncthreads();

        for(int col = 0 ; col < colitr ; col++)
        {
            for(int i = tid ; i < Br * Bc / 8; i += blockDim.x)
            {
                int r = i / 4;  
                int c = i % 4;

                *reinterpret_cast<float4*>(&qshared[r * 32 + c * 8]) = 
                    *reinterpret_cast<const float4*>(&Qptr[rowtileid * headdim * Br + col * 32 + r * headdim + c * 8]);
            }

            for(int i = tid ; i < Bc * Bc ; i += blockDim.x)
            {
                int r = i / Bc;
                int c = i % Bc;

                kshared[c * Bc + r] = Kptr[rowid * Bc * headdim + col * Bc + r * headdim + c];
            }

            // MMA block
            {
                for(int tr = 0 ; tr < Br / 16 ; tr++) 
                {
                    for(int col = 0 ; col < Bc / 8 ; col++)
                    {
                        float d1 = 0.f , d2 = 0.f , d3 = 0.f , d4 = 0.f;

                        for(int tc = 0 ; tc < Bc / 16 ; tc++) 
                        {
                            for(int i = tid ; i < 16 * 16 / 8 ; i += blockDim.x)
                            {
                                int r = i / 2;
                                int c = i % 2;

                                *reinterpret_cast<float4*>(&smenA[r * 16 + c * 8]) = 
                                    *reinterpret_cast<const float4*>(&qshared[tr * 16 * 32 + tc * 16 + r * 32 + c * 8]);
                            }

                            for(int i = tid ; i < 16 * 8 / 8 ; i += blockDim.x)
                            {
                                int r = i;
                                int c = i % 1;
                                
                                *reinterpret_cast<float4*>(&smenB[r * 8 + c]) = 
                                    *reinterpret_cast<const float4*>(&kshared[tc * 16 * 32 + col * 8 + r * 32 + c]);
                            }

                            __syncthreads();

                            const int col0 = (lane % 4) * 2; 
                            const int col1 = col0 + 8;

                            uint32_t a_frag[4];
                            a_frag[0] = *reinterpret_cast<const uint32_t*>(&smenA[group       * 16 + col0]);
                            a_frag[1] = *reinterpret_cast<const uint32_t*>(&smenA[(group + 8) * 16 + col0]);
                            a_frag[2] = *reinterpret_cast<const uint32_t*>(&smenA[group       * 16 + col1]);
                            a_frag[3] = *reinterpret_cast<const uint32_t*>(&smenA[(group + 8) * 16 + col1]);

                            uint32_t b_frag[2];

                            const int r0 = (lane % 4) * 2;
                            const int r1 = r0 + 8;

                            b_frag[0] = (uint32_t(__half_as_ushort(smenB[r0 * 8 + group])) | (uint32_t(__half_as_ushort(smenB[(r0 + 1) * 8 + group])) << 16));
                            b_frag[1] = (uint32_t(__half_as_ushort(smenB[r1 * 8 + group])) | (uint32_t(__half_as_ushort(smenB[(r1 + 1) * 8 + group])) << 16));

                            __syncthreads();
                            asm volatile(
                                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                                "{%0,%1,%2,%3},"
                                "{%4,%5,%6,%7},"
                                "{%8,%9},"
                                "{%10,%11,%12,%13};"    
                                    : "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4)
                                    : "r"(a_frag[0]), "r"(a_frag[1]),
                                    "r"(a_frag[2]), "r"(a_frag[3]),
                                    "r"(b_frag[0]), "r"(b_frag[1])
                                    ,"f"(d1),"f"(d2),"f"(d3),"f"(d4)
                                );
                        }

                        const int r0 = group;
                        const int r1 = r0 + 8;
                        const int c0 = (lane % 4) * 2;
                        const int c1 = c0 + 1 ;

                        output[(tr * 16 + r0) * Bc + col * 8 + c0] += d1 * scale;
                        output[(tr * 16 + r0) * Bc + col * 8 + c1] += d2 * scale;
                        output[(tr * 16 + r1) * Bc + col * 8 + c0] += d3 * scale;
                        output[(tr * 16 + r1) * Bc + col * 8 + c1] += d4 * scale;
                    }
                }
            }
        }

        // Load V into shared memory
        for(int i = tid; i < Br * headdim / 8; i += blockDim.x)
        {
            *reinterpret_cast<float4*>(&vshared[i * 8]) =
            *reinterpret_cast<const float4*>(&Vptr[rowid * Bc * headdim + i * 8]);
        }

        __syncthreads();

        if(tid < Br)
        {
            float m_tile = -FLT_MAX;
            float l_tile = 0.f;

            for(int c = 0 ; c < Bc ; c++)
                m_tile = fmaxf(m_tile , output[tid * Bc + c]);
            
            for(int c = 0 ; c < Bc ; c++)
            {
                output[tid * Bc + c] = expf(output[tid * Bc + c] - m_tile);
                l_tile += output[tid * Bc + c]; 
            }

            float m_old  =   m[tid];
            float m_new  = fmaxf(m_old , m_tile);
            alpha_s[tid] = expf(m_old  - m_new);
            beta_s[tid]  = expf(m_tile - m_new);

            l[tid] = alpha_s[tid] * l[tid] + beta_s[tid] * l_tile;
            m[tid] = m_new;
        }

        __syncthreads();

        const int itr = headdim / 8;    
        const int rowitr = Br / 16; 

        for(int iter = 0 ; iter < rowitr ; iter++)
        {
            for(int colitr = 0 ; colitr < itr ; colitr++)
            {
                float d1 = 0.f , d2 = 0.f , d3 = 0.f , d4 = 0.f;
                
                for(int cc = 0 ; cc < 2 ; cc++)
                {
                    for(int i = tid ; i < 16 * 16 ; i += blockDim.x)
                    {
                        int r = i / 16;
                        int c = i % 16;

                        smenA[r * 16 + c] = __float2half_rn(output[iter * 16 * 32 + cc * 16 + r * 32 + c]);
                    }

                    for(int i = tid ; i < 16 * 8 ; i += blockDim.x)
                    {
                        int r = i / 8;
                        int c = i % 8;
                        smenB[r * 8 + c] = vshared[(cc * 16 + r) * headdim + colitr * 8 + c];
                    }
                    __syncthreads();

                    const int col0 = (lane % 4) * 2;
                    const int col1 = col0 + 8;

                    uint32_t a_frag[4];
                    a_frag[0] = *reinterpret_cast<const uint32_t*>(&smenA[ group      * 16 + col0]);
                    a_frag[1] = *reinterpret_cast<const uint32_t*>(&smenA[(group + 8) * 16 + col0]);
                    a_frag[2] = *reinterpret_cast<const uint32_t*>(&smenA[ group      * 16 + col1]);
                    a_frag[3] = *reinterpret_cast<const uint32_t*>(&smenA[(group + 8) * 16 + col1]);

                    const int r0 = (lane % 4) * 2;
                    const int r1 = r0 + 8;

                    uint32_t b_frag[2];
                    b_frag[0] = (uint32_t(__half_as_ushort(smenB[r0 * 8 + group])) | (uint32_t(__half_as_ushort(smenB[(r0 + 1) * 8 + group])) << 16));
                    b_frag[1] = (uint32_t(__half_as_ushort(smenB[r1 * 8 + group])) | (uint32_t(__half_as_ushort(smenB[(r1 + 1) * 8 + group])) << 16));

                    __syncthreads();
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3},"
                        "{%4,%5,%6,%7},"
                        "{%8,%9},"
                        "{%10,%11,%12,%13};"
                            : "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4)
                            : "r"(a_frag[0]), "r"(a_frag[1]),
                            "r"(a_frag[2]), "r"(a_frag[3]),
                            "r"(b_frag[0]), "r"(b_frag[1])
                            ,"f"(d1),"f"(d2),"f"(d3),"f"(d4)
                        );
                }

                const int roww = rowtileid * Br * headdim + iter * 16 * headdim;
                const int colbase = colitr * 8;

                const int rr0 = group;
                const int rr1 = rr0 + 8;
                const int cc0 = (lane % 4) * 2;
                const int cc1 = cc0 + 1;

                int actual_row0 = iter * 16 + rr0;  
                int actual_row1 = iter * 16 + rr1;  

                float aa = __half2float(optr[roww + colbase + rr0 * headdim + cc0]);
                float ab = __half2float(optr[roww + colbase + rr0 * headdim + cc1]);
                float ac = __half2float(optr[roww + colbase + rr1 * headdim + cc0]);
                float ad = __half2float(optr[roww + colbase + rr1 * headdim + cc1]);

                aa = alpha_s[actual_row0] * aa + beta_s[actual_row0] * d1;
                ab = alpha_s[actual_row0] * ab + beta_s[actual_row0] * d2;
                ac = alpha_s[actual_row1] * ac + beta_s[actual_row1] * d3;
                ad = alpha_s[actual_row1] * ad + beta_s[actual_row1] * d4;

                optr[roww + colbase + rr0 * headdim + cc0] = __float2half(aa);
                optr[roww + colbase + rr0 * headdim + cc1] = __float2half(ab);
                optr[roww + colbase + rr1 * headdim + cc0] = __float2half(ac);
                optr[roww + colbase + rr1 * headdim + cc1] = __float2half(ad);
            }
        }
    }   

    for(int idx = tid; idx < Br * headdim; idx += blockDim.x)
    {
        int r = idx / headdim;
        int c = idx % headdim;
        float linv = __fdividef(1.0f, l[r]);
        int global_idx = rowtileid * Br * headdim + r * headdim + c;
        optr[global_idx] = __float2half(__half2float(optr[global_idx]) * linv);
    }

    __syncthreads();

  
    if (tid < Br) {
        int global_row_idx = rowtileid * Br + tid;
        if (global_row_idx < seqlen) {
            Lptr[global_row_idx] = m[tid] + logf(l[tid]);
        }
    }
}


template<int Br , int Bc , int NUM_HEADS>
__global__ void backwardFlashAttention_DQ(
    const __half* __restrict__ Q,         // [seq_len, headdim]
    const __half* __restrict__ K,         // [seq_len, headdim]
    const __half* __restrict__ V,         // [seq_len, headdim]
    const __half* __restrict__ O,
    const float* __restrict__ L_,         // Log-sum-exp statistics from Forward Pass [seq_len]
    __half* __restrict__ score,           // Shared/Global intermediate buffer
    __half* __restrict__ dl_score_global,
    __half* __restrict__ dL_dout,         // Incoming gradients from output [seq_len, headdim]
    __half* __restrict__ dQ,           // Output Gradient Query [seq_len, headdim]
    float* __restrict__ _dot,
    int seq_len,
    int headdim
)

{
    extern __shared__ char smem_buffer[];  

    size_t offset = 0;

    __half* smenA    = reinterpret_cast<__half*>(smem_buffer + offset);
    offset += 16 * 16 * sizeof(__half);

    __half* smenB    = reinterpret_cast<__half*>(smem_buffer + offset);
    offset += 16 * 8 * sizeof(__half);

    __half* qshared  = reinterpret_cast<__half*>(smem_buffer + offset);
    offset += Br * Bc * sizeof(__half);

    __half* kshared  = reinterpret_cast<__half*>(smem_buffer + offset);
    offset += Bc * Bc * sizeof(__half);

    __half* vshared  = reinterpret_cast<__half*>(smem_buffer + offset);
    offset += Bc * Bc * sizeof(__half);

    __half* dl_dout  = reinterpret_cast<__half*>(smem_buffer + offset);
    offset += Br * Bc * sizeof(__half);

    __half* dl_score = reinterpret_cast<__half*>(smem_buffer + offset);
    offset += Br * Bc * sizeof(__half);

    __half* scores   = reinterpret_cast<__half*>(smem_buffer + offset);
    offset += Br * Bc * sizeof(__half);

    __half* dl_ds    = reinterpret_cast<__half*>(smem_buffer + offset);
    offset += Br * Bc * sizeof(__half);

    offset = (offset + sizeof(float) - 1) / sizeof(float) * sizeof(float);

    float* dot       = reinterpret_cast<float*>(smem_buffer + offset);
    offset += Br * sizeof(float);

    float* L         = reinterpret_cast<float*>(smem_buffer + offset);
    offset += Br * sizeof(float);

    float* dq_accum  = reinterpret_cast<float*>(smem_buffer + offset);
    offset += Br * headdim * sizeof(float);

    const int batchid = blockIdx.x;
    const int headid  = blockIdx.y;
    const int tileid  = blockIdx.z;
    const int tid     = threadIdx.x;

    const long long base = (long long)batchid * seq_len * headdim * NUM_HEADS +
                            (long long)headid * seq_len * headdim;

    const __half* Qptr  = Q + base;
    const __half* Kptr  = K + base;
    const __half* Vptr  = V + base;
    const __half* Optr  = O + base;
          __half* dL_dQ = dQ + base;

    const __half* outP = dL_dout + base;

    const int lane  = tid % 32;
    const int group = lane / 4;

    const int rowtileid = tileid;
    const int total     = seq_len / Bc;  // whether Bc can be 32 or 64 , 1024 / 32 = 32 or 1024 / 64 = 16
    const int rowINitr  = headdim / Bc;  // even though its kinda obvious we are covering a row all columns at once just a safe sanity check

    if(tid < Br)
        L[tid] = L_[(long long)batchid * NUM_HEADS * seq_len + 
             (long long)headid  * seq_len + 
             rowtileid * Br + tid];
    
    if(tid < Br) dot[tid] = 0.f;
        __syncthreads();

    for(int col = 0 ; col < rowINitr ; col++)
    {
        for(int i = tid ; i < Br * Bc / 8 ; i += blockDim.x)
        {
            int r = i / (Bc / 8);
            int c = i % (Bc / 8);

            *reinterpret_cast<float4*>(&dl_dout[r * Bc + c * 8]) = 
                *reinterpret_cast<const float4*>(&outP[rowtileid * Br * headdim + col * Bc + r * headdim + c * 8]);
        }

        for(int i = tid ; i < Br * Bc / 8 ; i += blockDim.x)
        {
            // we will use qshared for out(O)
            int r = i / (Bc / 8);
            int c = i % (Bc / 8);

            *reinterpret_cast<float4*>(&qshared[r * Bc + c * 8]) = 
                *reinterpret_cast<const float4*>(&Optr[rowtileid * Br * headdim + col * Bc + r * headdim + c * 8]);
        }

        __syncthreads();

        if(tid < Br)
        {
            float sum = 0.f;
            for(int k = 0; k < Bc; k++)
                sum += __half2float(dl_dout[tid * Bc + k]) *
                    __half2float(qshared[tid * Bc + k]);
            dot[tid] += sum;   // accumulate over col tiles
        }

        __syncthreads();
    }

    if (tid < Br) {
        long long dot_global_idx = (long long)batchid * NUM_HEADS * seq_len + 
                                (long long)headid * seq_len + 
                                (rowtileid * Br + tid);
        
        if ((rowtileid * Br + tid) < seq_len) {
            _dot[dot_global_idx] = dot[tid];
        }
    }

    for(int i = tid; i < Br * headdim; i += blockDim.x)
        dq_accum[i] = 0.f;
    __syncthreads();


    for(int rowid = 0 ; rowid < total ; rowid++)
    {

        for(int i = tid; i < Br * Bc; i += blockDim.x)
            scores[i] = __float2half(0.f);

        for(int i = tid; i < Br * Bc; i += blockDim.x)
            dl_score[i] = __float2half(0.f);
        __syncthreads();

        // we need Q @ K.T
        for(int colitr = 0 ; colitr < rowINitr ; colitr++)
        {
            // loading Q (vector float 4 load for half -- 8 elements at once)
            for(int i = tid ; i < Br * Bc / 8 ; i += blockDim.x)
            {
                int r = i / (Bc / 8);
                int c = i % (Bc / 8);

                *reinterpret_cast<float4*>(&qshared[r * Bc + c * 8]) =
                    *reinterpret_cast<const float4*>(&Qptr[rowtileid * Br * headdim + colitr * Bc + r * headdim + c * 8]);
            }

            // cannot implement vector loading cuz we are loading in transpose form
            for(int i = tid ; i < Bc * Bc ; i += blockDim.x)
            {
                int r = i / Bc;
                int c = i % Bc;

                kshared[c * Bc + r] = Kptr[rowid * Bc * headdim + colitr * Bc + r * headdim + c];
            }

            __syncthreads();

            // we got Q and K , size (Br , Bc) @ (Bc , Bc).T -> (Br , Bc)
            // we gonna use mma -> m16n8k16

            const int Tr   = Br / 16; 
            const int Tc   = Bc / 16;
            const int citr = Bc / 8 ;

            for(int rr = 0 ; rr < Tr ; rr++)
            {
                for(int cc = 0 ; cc < citr ; cc++)
                {
                    float d1 = 0.f , d2 = 0.f , d3 = 0.f , d4 = 0.f;

                    for(int col = 0 ; col < Tc ; col++)
                    {
                        for(int i = tid ; i < 16 * 16 / 8 ; i += blockDim.x)
                        {
                            int r = i / (16 / 8);
                            int c = i % (16 / 8);

                            *reinterpret_cast<float4*>(&smenA[r * 16 + c * 8]) = 
                                *reinterpret_cast<const float4*>(&qshared[rr * 16 * Bc + col * 16 + r * Bc + c * 8]);
                        }

                        for(int i = tid ; i < 16 * 8 ; i += blockDim.x)
                        {
                            int r = i / 8;
                            int c = i % 8;

                            smenB[r * 8 + c] = kshared[cc * 8 + col * 16 * Bc + r * Bc + c];
                        }

                        __syncthreads();

                        // we have smen loaded we can do mma now
                        uint32_t a_frag[4];

                        const int col0 = (lane % 4) * 2;
                        const int col1 = col0 + 8;

                        a_frag[0] = *reinterpret_cast<const uint32_t*>(&smenA[ group      * 16 + col0]);
                        a_frag[1] = *reinterpret_cast<const uint32_t*>(&smenA[(group + 8) * 16 + col0]);
                        a_frag[2] = *reinterpret_cast<const uint32_t*>(&smenA[ group      * 16 + col1]);
                        a_frag[3] = *reinterpret_cast<const uint32_t*>(&smenA[(group + 8) * 16 + col1]);

                        uint32_t b_frag[2];

                        const int r0 = (lane % 4) * 2;
                        const int r1 = r0 + 8;

                        b_frag[0] = (uint32_t(__half_as_ushort(smenB[r0 * 8 + group])) | (uint32_t(__half_as_ushort(smenB[(r0 + 1) * 8 + group])) << 16));

                        b_frag[1] = (uint32_t(__half_as_ushort(smenB[r1 * 8 + group])) | (uint32_t(__half_as_ushort(smenB[(r1 + 1) * 8 + group])) << 16));

                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                            "{%0,%1,%2,%3},"
                            "{%4,%5,%6,%7},"
                            "{%8,%9},"
                            "{%10,%11,%12,%13};"
                            : "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4)
                            : "r"(a_frag[0]), "r"(a_frag[1]),                            
                            "r"(a_frag[2]), "r"(a_frag[3]),
                            "r"(b_frag[0]), "r"(b_frag[1])
                            ,"f"(d1),"f"(d2),"f"(d3),"f"(d4)
                        ); 
                        __syncthreads();
                    }

                    // now map this
                    const int r0 = group;
                    const int r1 = r0 + 8;

                    const int c0 = (lane % 4) * 2;
                    const int c1 = c0 + 1;

                    const int idx0 = rr * 16 * Bc + cc * 8 + r0 * Bc + c0;
                    const int idx1 = rr * 16 * Bc + cc * 8 + r0 * Bc + c1;
                    const int idx2 = rr * 16 * Bc + cc * 8 + r1 * Bc + c0;
                    const int idx3 = rr * 16 * Bc + cc * 8 + r1 * Bc + c1;

                    scores[idx0] = __float2half(__half2float(scores[idx0]) + d1);
                    scores[idx1] = __float2half(__half2float(scores[idx1]) + d2);
                    scores[idx2] = __float2half(__half2float(scores[idx2]) + d3);
                    scores[idx3] = __float2half(__half2float(scores[idx3]) + d4);

                    // scores shapes are Br * Bc

                }
            }
            __syncthreads();
        
            // dl_score = dl_out (Br , Bc) and V.T (Bc , Bc) -> (Br , Bc)

            // loaded dl_out
            for(int i = tid ; i < Br * Bc / 8 ; i += blockDim.x)
            {
                int r = i / (Bc / 8);
                int c = i % (Bc / 8);

                *reinterpret_cast<float4*>(&dl_dout[r * Bc + c * 8]) =
                    *reinterpret_cast<const float4*>(&outP[rowtileid * Br * headdim + colitr * Bc + r * headdim + c * 8]);
            }

            // loaded V
            for(int i = tid ; i < Bc * Bc ; i += blockDim.x)
            {
                int r = i / Bc;
                int c = i % Bc;

                vshared[c * Bc + r] = Vptr[rowid * Bc * headdim + colitr * Bc + r * headdim + c];
            }
            __syncthreads();

            for(int rr = 0 ; rr < Tr ; rr++)
            {
                for(int cc = 0 ; cc < citr ; cc++)
                {
                    float d1 = 0.f , d2 = 0.f , d3 = 0.f , d4 = 0.f;

                    for(int col = 0 ; col < Tc ; col++)
                    {
                        for(int i = tid ; i < 16 * 16 / 8 ; i += blockDim.x)
                        {
                            int r = i / (16 / 8);
                            int c = i % (16 / 8);

                            *reinterpret_cast<float4*>(&smenA[r * 16 + c * 8]) = 
                                *reinterpret_cast<const float4*>(&dl_dout[rr * 16 * Bc + col * 16 + r * Bc + c * 8]);
                        }

                        for(int i = tid ; i < 16 * 8 ; i += blockDim.x)
                        {
                            int r = i / 8;
                            int c = i % 8;

                            smenB[r * 8 + c] = vshared[cc * 8 + col * 16 * Bc + r * Bc + c];
                        }

                        __syncthreads();

                        uint32_t a_frag[4];

                        const int col0 = (lane % 4) * 2;
                        const int col1 = col0 + 8;

                        a_frag[0] = *reinterpret_cast<const uint32_t*>(&smenA[ group      * 16 + col0]);
                        a_frag[1] = *reinterpret_cast<const uint32_t*>(&smenA[(group + 8) * 16 + col0]);
                        a_frag[2] = *reinterpret_cast<const uint32_t*>(&smenA[ group      * 16 + col1]);
                        a_frag[3] = *reinterpret_cast<const uint32_t*>(&smenA[(group + 8) * 16 + col1]);

                        uint32_t b_frag[2];

                        const int r0 = (lane % 4) * 2;
                        const int r1 = r0 + 8;

                        b_frag[0] = (uint32_t(__half_as_ushort(smenB[r0 * 8 + group])) | (uint32_t(__half_as_ushort(smenB[(r0 + 1) * 8 + group])) << 16));

                        b_frag[1] = (uint32_t(__half_as_ushort(smenB[r1 * 8 + group])) | (uint32_t(__half_as_ushort(smenB[(r1 + 1) * 8 + group])) << 16));

                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                            "{%0,%1,%2,%3},"
                            "{%4,%5,%6,%7},"
                            "{%8,%9},"
                            "{%10,%11,%12,%13};"
                            : "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4)
                            : "r"(a_frag[0]), "r"(a_frag[1]),                            
                            "r"(a_frag[2]), "r"(a_frag[3]),
                            "r"(b_frag[0]), "r"(b_frag[1])
                            ,"f"(d1),"f"(d2),"f"(d3),"f"(d4)
                        ); 
                         __syncthreads();
                    }

                    const int out_row0 = group;
                    const int out_row1 = out_row0 + 8;

                    const int out_col0 = (lane % 4) * 2;
                    const int out_col1 = out_col0 + 1;

                    const int idx_d1 = rr * 16 * Bc + cc * 8 + out_row0 * Bc + out_col0;
                    const int idx_d2 = rr * 16 * Bc + cc * 8 + out_row0 * Bc + out_col1;
                    const int idx_d3 = rr * 16 * Bc + cc * 8 + out_row1 * Bc + out_col0;
                    const int idx_d4 = rr * 16 * Bc + cc * 8 + out_row1 * Bc + out_col1;

                    dl_score[idx_d1] = __float2half(__half2float(dl_score[idx_d1]) + d1);
                    dl_score[idx_d2] = __float2half(__half2float(dl_score[idx_d2]) + d2);
                    dl_score[idx_d3] = __float2half(__half2float(dl_score[idx_d3]) + d3);
                    dl_score[idx_d4] = __float2half(__half2float(dl_score[idx_d4]) + d4);

                    // dl_score shapes are Br * Bc
                }
                __syncthreads();
            }
        } 
            

            for(int i = tid; i < Br * Bc; i += blockDim.x)
            {
                int r = i / Bc;
                int c = i % Bc;

                float val = __half2float(scores[r * Bc + c]);
                float s   = expf(val / sqrtf((float)headdim) - L[r]);
                scores[r * Bc + c] = __float2half(s);   // now scores holds S
            }
            __syncthreads();

            for(int i = tid ; i < Br * Bc ; i += blockDim.x)
            {
                int r = i / Bc;
                int c = i % Bc;
                float val = __half2float(dl_score[r * Bc + c]);
                float s = val - dot[r];  
                dl_score[r * Bc + c] = __float2half(s);
            }
            __syncthreads();

            for(int i = tid ; i < Br * Bc ; i += blockDim.x)
            {
                int r = i / Bc; 
                int c = i % Bc;

                float val1 = __half2float(scores[r * Bc + c]);
                float val2 = __half2float(dl_score[r * Bc + c]);
                
                dl_ds[r * Bc + c] = __float2half(val1 * val2);
            }

            __syncthreads();

            // now we have proper dl/ds
            for(int x = 0 ; x < headdim / Bc ; x++)
            {
                for(int i = tid ; i < Bc * Bc / 8 ; i += blockDim.x)
                {   
                    int r = i / (Bc / 8);
                    int c = i % (Bc / 8);

                    *reinterpret_cast<float4*>(&kshared[r * Bc + c * 8]) = 
                        *reinterpret_cast<const float4*>(&Kptr[rowid * Bc * headdim + x * Bc + r * headdim + c * 8]);
                }
                __syncthreads(); //

            for(int i = tid; i < Br * Bc; i += blockDim.x)
            {
                int r = i / Bc;
                int c = i % Bc;

                long long score_base = (long long)batchid * NUM_HEADS * seq_len * seq_len
                                    + (long long)headid  * seq_len * seq_len;

                int global_row = rowtileid * Br + r;
                int global_col = rowid    * Bc + c;

                dl_score_global[score_base + global_row * seq_len + global_col] = dl_ds[r * Bc + c];
            }  
            __syncthreads();
            for(int i = tid; i < Br * Bc; i += blockDim.x)
            {
                int r = i / Bc;
                int c = i % Bc;
                long long score_base = (long long)batchid * NUM_HEADS * seq_len * seq_len
                                    + (long long)headid  * seq_len * seq_len;

                int global_row = rowtileid * Br + r;
                int global_col = rowid    * Bc + c;

                score[score_base + global_row * seq_len + global_col] = scores[r * Bc + c];
            }
            __syncthreads();

            for(int Trow = 0 ; Trow < Br / 16 ; Trow++)
            {
                for(int tcc = 0 ; tcc < Bc / 8 ; tcc++)
                {
                    float d1 = 0.f , d2 = 0.f , d3 = 0.f , d4 = 0.f;

                    for(int tcr = 0 ; tcr < Bc / 16 ; tcr++)
                    {
                        for(int i = tid ; i < 16 * 16 / 8 ; i += blockDim.x)
                        {
                            int r = i / (16 / 8);
                            int c = i % (16 / 8);

                            *reinterpret_cast<float4*>(&smenA[r * 16 + c * 8]) =    
                                *reinterpret_cast<const float4*>(&dl_ds[Trow * 16 * Bc + tcr * 16 + r * Bc + c * 8]);
                        }

                        // Load smenB from row-major kshared tile
                        for(int i = tid ; i < 16 * 8 ; i += blockDim.x)
                        {
                            int r = i / 8;
                            int c = i % 8;

                            smenB[r * 8 + c] = kshared[tcc * 8 + tcr * 16 * Bc + r * Bc + c];
                        }
                        
                        __syncthreads();

                        uint32_t a_frag[4];
                        const int smenA_col0 = (lane % 4) * 2;
                        const int smenA_col1 = smenA_col0 + 8;

                        a_frag[0] = *reinterpret_cast<const uint32_t*>(&smenA[ group      * 16 + smenA_col0]);
                        a_frag[1] = *reinterpret_cast<const uint32_t*>(&smenA[(group + 8) * 16 + smenA_col0]);
                        a_frag[2] = *reinterpret_cast<const uint32_t*>(&smenA[ group      * 16 + smenA_col1]);
                        a_frag[3] = *reinterpret_cast<const uint32_t*>(&smenA[(group + 8) * 16 + smenA_col1]);

                        uint32_t b_frag[2];
                        const int smenB_row0 = (lane % 4) * 2;
                        const int smenB_row1 = smenB_row0 + 8;

                        b_frag[0] = (uint32_t(__half_as_ushort(smenB[smenB_row0 * 8 + group])) | 
                                    (uint32_t(__half_as_ushort(smenB[(smenB_row0 + 1) * 8 + group])) << 16));

                        b_frag[1] = (uint32_t(__half_as_ushort(smenB[smenB_row1 * 8 + group])) | 
                                    (uint32_t(__half_as_ushort(smenB[(smenB_row1 + 1) * 8 + group])) << 16));

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
                            "f"(d1),"f"(d2),"f"(d3),"f"(d4)
                        ); 
                        
                        __syncthreads();
                    }
                    
                    const int thread_row0 = lane / 4;
                    const int thread_row1 = thread_row0 + 8;

                    const int thread_col0 = (lane % 4) * 2;
                    const int thread_col1 = thread_col0 + 1;

                    const int idx_d1 = Trow * 16 * headdim + thread_row0 * headdim + (x * Bc + tcc * 8 + thread_col0);
                    const int idx_d2 = Trow * 16 * headdim + thread_row0 * headdim + (x * Bc + tcc * 8 + thread_col1);
                    const int idx_d3 = Trow * 16 * headdim + thread_row1 * headdim + (x * Bc + tcc * 8 + thread_col0);
                    const int idx_d4 = Trow * 16 * headdim + thread_row1 * headdim + (x * Bc + tcc * 8 + thread_col1);

                
                    const float bwd_scale = 1.0f / sqrtf((float)headdim);
                    dq_accum[idx_d1] += d1 * bwd_scale;
                    dq_accum[idx_d2] += d2 * bwd_scale;
                    dq_accum[idx_d3] += d3 * bwd_scale;
                    dq_accum[idx_d4] += d4 * bwd_scale;
                    }
                    __syncthreads();
            }
        }
    }
    const size_t global_offset = (size_t)rowtileid * Br * headdim;
    for (size_t i = tid; i < (size_t)Br * headdim; i += blockDim.x) {
        dL_dQ[global_offset + i] = __float2half(dq_accum[i]);
    }

}




template<int Br , int NUM_HEADS>
__global__ void backwardFlashAttention_DK_DV(
    const __half* __restrict__ Q,             
    const __half* __restrict__ score,               
    const __half* __restrict__ dl_score_global,     
    const __half* __restrict__ dL_dout,        
    __half* __restrict__ dL_dK,           
    __half* __restrict__ dL_dV,           
    int seq_len,
    int headdim
)
{
    extern __shared__ char smem_buffer[];  

    size_t offset = 0;

    __half* smenA = reinterpret_cast<__half*>(smem_buffer + offset);
    offset += 16 * 16 * sizeof(__half);

    __half* smenB = reinterpret_cast<__half*>(smem_buffer + offset);
    offset += 16 * 8 * sizeof(__half);

    __half* dscore = reinterpret_cast<__half*>(smem_buffer + offset);
    offset += Br * Br * sizeof(__half);

    __half* DQ = reinterpret_cast<__half*>(smem_buffer + offset);
    offset += Br * Br * sizeof(__half);

    __half* DL_ds = reinterpret_cast<__half*>(smem_buffer + offset);
    offset += Br * Br * sizeof(__half);

    __half* dl_out = reinterpret_cast<__half*>(smem_buffer + offset);
    offset += Br * Br * sizeof(__half);

    float* acc     = reinterpret_cast<float*>(smem_buffer + offset);
    offset += Br * Br * sizeof(float);

    const int batchid = blockIdx.x;
    const int headid  = blockIdx.y;
    const int tileid  = blockIdx.z;

    const int tid     = threadIdx.x;

    const int lane    = tid % 32;
    const int group   = lane / 4;

    const long long base = (long long)batchid * NUM_HEADS * seq_len * headdim + 
                            (long long)headid * seq_len * headdim;
    
    const long long score_base = (long long)batchid * NUM_HEADS * seq_len * seq_len +
                                 (long long)headid  * seq_len * seq_len;

    const __half* Qptr = Q               + base;
    const __half* Sptr = score           + score_base;
    const __half* DSpt = dl_score_global + score_base;
    const __half* Optr = dL_dout         + base;

    const int rowitr    = headdim / Br;
    const int coll      = seq_len / Br;
    const int rowtileid = tileid;

    // as we doing same for both 64 * 64 @ 64 * 64 so same vars can reduce registers 
    const int Tr = Br / 16;
    const int Tc = Br /  8;
    // we need one more but as our dim is 64 so we can reuse Tr
    
    for(int colitr = 0 ; colitr < coll ; colitr++)
    { 
        for(int rowid = 0 ; rowid < rowitr ; rowid++)
        {
            // we going for DV
            for(int i = tid ; i < Br * Br ; i += blockDim.x)
            {
                int r = i / Br;
                int c = i % Br;

                dscore[c * Br + r] = Sptr[rowtileid * Br + colitr * Br * seq_len + r * seq_len + c];
            }

            for(int i = tid ; i < Br * Br / 8 ; i += blockDim.x)
            {
                int r = i / (Br / 8);
                int c = i % (Br / 8);

                *reinterpret_cast<float4*>(&dl_out[r * Br + c * 8]) = 
                    *reinterpret_cast<const float4*>(&Optr[colitr * Br * headdim + rowid * Br + r * headdim + c * 8]);
            }

            __syncthreads(); // it is important so we do not run into race condition

            for(int i = tid ; i < Br * Br ; i += blockDim.x)
                acc[i] = 0.f; __syncthreads();
            
            for(int rr = 0 ; rr < Tr ; rr++)
            {
                for(int cc = 0 ; cc < Tc ; cc++)
                {
                    float d1 = 0.f , d2 = 0.f , d3 = 0.f , d4 = 0.f;

                    for(int col = 0 ; col < Tr ; col++)
                    {
                        for(int i = tid ; i < 16 * 16 ; i += blockDim.x)
                        {
                            int r = i / 16;
                            int c = i % 16;

                            smenA[r * 16 + c] = dscore[rr * 16 * Br + col * 16 + r * Br + c];
                        }

                        for(int i = tid ; i < 16 * 8 / 8 ; i += blockDim.x)
                        {
                            int r = i / (8 / 8);
                            int c = i % (8 / 8);

                            *reinterpret_cast<float4*>(&smenB[r * 8 + c * 8]) = 
                                *reinterpret_cast<const float4*>(&dl_out[cc * 8 + col * 16 * Br + r * Br + c * 8]);
                        }
                        __syncthreads();

                        const int r0 = (lane % 4) * 2;
                        const int r1 = r0 + 8;

                        uint32_t a_frag[4];

                        a_frag[0] = *reinterpret_cast<const uint32_t*>(&smenA[ group * 16      + r0]);
                        a_frag[1] = *reinterpret_cast<const uint32_t*>(&smenA[(group + 8) * 16 + r0]);
                        a_frag[2] = *reinterpret_cast<const uint32_t*>(&smenA[ group * 16      + r1]);
                        a_frag[3] = *reinterpret_cast<const uint32_t*>(&smenA[(group + 8) * 16 + r1]);

                        const int c0 = (lane % 4) * 2;
                        const int c1 = c0 + 8;

                        uint32_t b_frag[2];

                        b_frag[0] = (uint32_t(__half_as_ushort(smenB[c0 * 8 + group])) | (uint32_t(__half_as_ushort(smenB[(c0 + 1) * 8 + group])) << 16));
                        b_frag[1] = (uint32_t(__half_as_ushort(smenB[c1 * 8 + group])) | (uint32_t(__half_as_ushort(smenB[(c1 + 1) * 8 + group])) << 16));
                        
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
                    __syncthreads();

                    const int r0 = group;
                    const int r1 = r0 + 8;
                    const int c0 = (lane % 4) * 2;
                    const int c1 = c0 + 1;

                    const int global = rr * 16 * Br + cc * 8;
                    
                    const int idx0 = global + r0 * Br + c0;
                    const int idx1 = global + r0 * Br + c1;
                    const int idx2 = global + r1 * Br + c0;
                    const int idx3 = global + r1 * Br + c1;

                    acc[idx0] += d1;
                    acc[idx1] += d2;
                    acc[idx2] += d3;
                    acc[idx3] += d4;

                    __syncthreads();
                }
            }

            // from here we need to save it globally

            const size_t global_offset = (size_t)base 
                    + (size_t)rowtileid * Br * headdim 
                    + rowid * Br;
            for (size_t i = tid; i < (size_t)Br * Br; i += blockDim.x) {
                int r = i / Br;
                int c = i % Br;
                size_t idx = global_offset + r * headdim + c;  
                atomicAdd(reinterpret_cast<__half*>(&dL_dV[idx]), __float2half(acc[i]));
            }
            
            __syncthreads();

            // we going for DV
            for(int i = tid ; i < Br * Br ; i += blockDim.x)
            {
                int r = i / Br;
                int c = i % Br;

                DL_ds[c * Br + r] = DSpt[rowtileid * Br + colitr * Br * seq_len + r * seq_len + c];
            }
            __syncthreads();

            for(int i = tid; i < Br * Br; i += blockDim.x)
            {
                DL_ds[i] = __float2half(__half2float(DL_ds[i]) / sqrtf((float)headdim));
            }
            __syncthreads();

            for(int i = tid ; i < Br * Br / 8 ; i += blockDim.x)
            {
                int r = i / (Br / 8);
                int c = i % (Br / 8);

                *reinterpret_cast<float4*>(&DQ[r * Br + c * 8]) = 
                    *reinterpret_cast<const float4*>(&Qptr[colitr * Br * headdim + rowid * Br + r * headdim + c * 8]);
            }

            __syncthreads(); // it is important so we do not run into race condition
            for(int i = tid ; i < Br * Br ; i += blockDim.x)
                acc[i] = 0.f; __syncthreads();
            
            for(int rr = 0 ; rr < Tr ; rr++)
            {
                for(int cc = 0 ; cc < Tc ; cc++)
                {
                    float d1 = 0.f , d2 = 0.f , d3 = 0.f , d4 = 0.f;

                    for(int col = 0 ; col < Tr ; col++)
                    {
                        for(int i = tid ; i < 16 * 16 ; i += blockDim.x)
                        {
                            int r = i / 16;
                            int c = i % 16;

                            smenA[r * 16 + c] = DL_ds[rr * 16 * Br + col * 16 + r * Br + c];
                        }

                        for(int i = tid ; i < 16 * 8 / 8 ; i += blockDim.x)
                        {
                            int r = i / (8 / 8);
                            int c = i % (8 / 8);

                            *reinterpret_cast<float4*>(&smenB[r * 8 + c * 8]) = 
                                *reinterpret_cast<const float4*>(&DQ[cc * 8 + col * 16 * Br + r * Br + c * 8]);
                        }
                        __syncthreads();

                        const int r0 = (lane % 4) * 2;
                        const int r1 = r0 + 8;

                        uint32_t a_frag[4];

                        a_frag[0] = *reinterpret_cast<const uint32_t*>(&smenA[ group * 16      + r0]);
                        a_frag[1] = *reinterpret_cast<const uint32_t*>(&smenA[(group + 8) * 16 + r0]);
                        a_frag[2] = *reinterpret_cast<const uint32_t*>(&smenA[ group * 16      + r1]);
                        a_frag[3] = *reinterpret_cast<const uint32_t*>(&smenA[(group + 8) * 16 + r1]);

                        const int c0 = (lane % 4) * 2;
                        const int c1 = c0 + 8;

                        uint32_t b_frag[2];

                        b_frag[0] = (uint32_t(__half_as_ushort(smenB[c0 * 8 + group])) | (uint32_t(__half_as_ushort(smenB[(c0 + 1) * 8 + group])) << 16));
                        b_frag[1] = (uint32_t(__half_as_ushort(smenB[c1 * 8 + group])) | (uint32_t(__half_as_ushort(smenB[(c1 + 1) * 8 + group])) << 16));
                        
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

                    // from here we need to save it globally
                    const int r0 = group;
                    const int r1 = r0 + 8;
                    const int c0 = (lane % 4) * 2;
                    const int c1 = c0 + 1;

                    const int global = rr * 16 * Br + cc * 8;
                    
                    const int idx0 = global + r0 * Br + c0;
                    const int idx1 = global + r0 * Br + c1;
                    const int idx2 = global + r1 * Br + c0;
                    const int idx3 = global + r1 * Br + c1;

                    acc[idx0] += d1;
                    acc[idx1] += d2;
                    acc[idx2] += d3;
                    acc[idx3] += d4;

                    __syncthreads(); 
                }
            } // 

            const size_t globadl = (size_t)base + (size_t)rowtileid * Br * headdim + rowid * Br;
                        for (size_t i = tid; i < (size_t)Br * Br; i += blockDim.x) {
                int r = i / Br;
                int c = i % Br;
                size_t idx = globadl + r * headdim + c; 
                atomicAdd(reinterpret_cast<__half*>(&dL_dK[idx]), __float2half(acc[i]));
            }
            __syncthreads();
        }
    }
}
#include "flashattn.h"