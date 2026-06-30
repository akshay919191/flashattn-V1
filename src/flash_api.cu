#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <vector>
#include <cmath>
#include <cstdint>

#include "helper.cuh"
#include "flash_Acc_reg.cuh"

template<int Br, int Bc, int D>
std::vector<torch::Tensor> flash_fwd_impl(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
)
{
    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);

    auto O = torch::empty_like(Q);
    auto L = torch::empty(
        {B, H, N},
        torch::TensorOptions().device(Q.device()).dtype(torch::kFloat32)
    );

    constexpr int PAD = 8;
    constexpr int Q_STRIDE = D + PAD;
    constexpr int K_STRIDE = D + PAD;
    constexpr int V_STRIDE = D + PAD;
    constexpr int S_STRIDE = Bc;

    size_t smem_size =
        Br * Q_STRIDE * sizeof(__half) +
        Bc * K_STRIDE * sizeof(__half) +
        Bc * K_STRIDE * sizeof(__half) +
        Bc * V_STRIDE * sizeof(__half) +
        Bc * V_STRIDE * sizeof(__half) +
        Br * S_STRIDE * sizeof(float) +
        Br * Bc * sizeof(__half) +
        Br * sizeof(float) +
        Br * sizeof(float) +
        Br * sizeof(float) +
        1024;

    dim3 block(128);
    dim3 grid(B, H, (N + Br - 1) / Br);

    cudaFuncSetAttribute(
        flashattn_fwd_kernel<Br, Bc, D>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_size
    );
    C10_CUDA_CHECK(cudaGetLastError());

    flashattn_fwd_kernel<Br, Bc, D>
        <<<grid, block, smem_size, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(K.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(V.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(O.data_ptr<at::Half>()),
            L.data_ptr<float>(),
            N
        );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {O, L};
}

std::vector<torch::Tensor> flash_fwd(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
)
{
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA");
    TORCH_CHECK(K.is_cuda(), "K must be CUDA");
    TORCH_CHECK(V.is_cuda(), "V must be CUDA");

    TORCH_CHECK(Q.scalar_type() == torch::kFloat16, "Q must be fp16");
    TORCH_CHECK(K.scalar_type() == torch::kFloat16, "K must be fp16");
    TORCH_CHECK(V.scalar_type() == torch::kFloat16, "V must be fp16");

    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");

    TORCH_CHECK(Q.dim() == 4, "Q must be [B,H,N,D]");
    TORCH_CHECK(K.dim() == 4, "K must be [B,H,N,D]");
    TORCH_CHECK(V.dim() == 4, "V must be [B,H,N,D]");

    int B = Q.size(0);
    int H = Q.size(1);
    int N_runtime = Q.size(2);
    int D_runtime = Q.size(3);

    TORCH_CHECK(K.size(0) == B && K.size(1) == H && K.size(2) == N_runtime && K.size(3) == D_runtime,
                "K shape must match Q");
    TORCH_CHECK(V.size(0) == B && V.size(1) == H && V.size(2) == N_runtime && V.size(3) == D_runtime,
                "V shape must match Q");

    TORCH_CHECK(N_runtime > 0, "N must be > 0");

    if (D_runtime == 32) {
        return flash_fwd_impl<16, 32, 32>(Q, K, V);
    }

    if (D_runtime == 64) {
        return flash_fwd_impl<16, 32, 64>(Q, K, V);
    }

    if (D_runtime == 128) {
        return flash_fwd_impl<16, 32, 128>(Q, K, V);
    }

    if (D_runtime == 256) {
        return flash_fwd_impl<16, 16, 256>(Q, K, V);
    }

    TORCH_CHECK(false, "Unsupported D. Supported D: 32, 64, 128, 256");
}

template<int Br, int Bc, int D>
std::vector<torch::Tensor> flash_bwd_impl(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor dO,
    torch::Tensor L
)
{
    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);

    auto DQ = torch::empty_like(Q);
    auto DK = torch::empty_like(K);
    auto DV = torch::empty_like(V);

    auto Delta = torch::empty(
        {B, H, N},
        torch::TensorOptions().device(Q.device()).dtype(torch::kFloat32)
    );

    dim3 block(128);

    int total_rows = B * H * N;
    int delta_blocks = (total_rows + 3) / 4;

    calc_delta_kernel<D>
        <<<delta_blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __half*>(O.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(dO.data_ptr<at::Half>()),
            Delta.data_ptr<float>(),
            total_rows
        );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    constexpr int PAD = 8;

    constexpr int Q_STRIDE  = D + PAD;
    constexpr int K_STRIDE  = D + PAD;
    constexpr int V_STRIDE  = D + PAD;
    constexpr int DO_STRIDE = D + PAD;

    size_t bwd_smem_size =
        Br * Q_STRIDE  * sizeof(__half) +
        Bc * K_STRIDE  * sizeof(__half) +
        Bc * V_STRIDE  * sizeof(__half) +
        Br * DO_STRIDE * sizeof(__half) +

        Br * Bc * sizeof(float) +
        Br * sizeof(float) +
        Br * sizeof(float) +

        Br * Bc * sizeof(__half) +
        Br * Bc * sizeof(float) +
        Br * Bc * sizeof(__half) +

        1024;

    int Tr = (N + Br - 1) / Br;
    int Tc = (N + Bc - 1) / Bc;

    dim3 dkdv_grid(B, H, Tc);

    cudaFuncSetAttribute(
        flashattn_bwd_dkdv_kernel<Br, Bc, D>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)bwd_smem_size
    );
    C10_CUDA_CHECK(cudaGetLastError());

    flashattn_bwd_dkdv_kernel<Br, Bc, D>
        <<<dkdv_grid, block, bwd_smem_size, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(K.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(V.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(dO.data_ptr<at::Half>()),
            L.data_ptr<float>(),
            Delta.data_ptr<float>(),
            reinterpret_cast<__half*>(DK.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(DV.data_ptr<at::Half>()),
            N
        );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 dq_grid(B, H, Tr);

    cudaFuncSetAttribute(
        flashattn_bwd_dq_kernel<Br, Bc, D>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)bwd_smem_size
    );
    C10_CUDA_CHECK(cudaGetLastError());

    flashattn_bwd_dq_kernel<Br, Bc, D>
        <<<dq_grid, block, bwd_smem_size, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(K.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(V.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(dO.data_ptr<at::Half>()),
            L.data_ptr<float>(),
            Delta.data_ptr<float>(),
            reinterpret_cast<__half*>(DQ.data_ptr<at::Half>()),
            N
        );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {DQ, DK, DV};
}

std::vector<torch::Tensor> flash_bwd(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor dO,
    torch::Tensor L
)
{
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA");
    TORCH_CHECK(K.is_cuda(), "K must be CUDA");
    TORCH_CHECK(V.is_cuda(), "V must be CUDA");
    TORCH_CHECK(O.is_cuda(), "O must be CUDA");
    TORCH_CHECK(dO.is_cuda(), "dO must be CUDA");
    TORCH_CHECK(L.is_cuda(), "L must be CUDA");

    TORCH_CHECK(Q.scalar_type() == torch::kFloat16, "Q must be fp16");
    TORCH_CHECK(K.scalar_type() == torch::kFloat16, "K must be fp16");
    TORCH_CHECK(V.scalar_type() == torch::kFloat16, "V must be fp16");
    TORCH_CHECK(O.scalar_type() == torch::kFloat16, "O must be fp16");
    TORCH_CHECK(dO.scalar_type() == torch::kFloat16, "dO must be fp16");
    TORCH_CHECK(L.scalar_type() == torch::kFloat32, "L must be fp32");

    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");
    TORCH_CHECK(O.is_contiguous(), "O must be contiguous");
    TORCH_CHECK(dO.is_contiguous(), "dO must be contiguous");
    TORCH_CHECK(L.is_contiguous(), "L must be contiguous");

    TORCH_CHECK(Q.dim() == 4, "Q must be [B,H,N,D]");
    TORCH_CHECK(K.dim() == 4, "K must be [B,H,N,D]");
    TORCH_CHECK(V.dim() == 4, "V must be [B,H,N,D]");
    TORCH_CHECK(O.dim() == 4, "O must be [B,H,N,D]");
    TORCH_CHECK(dO.dim() == 4, "dO must be [B,H,N,D]");
    TORCH_CHECK(L.dim() == 3, "L must be [B,H,N]");

    int B = Q.size(0);
    int H = Q.size(1);
    int N_runtime = Q.size(2);
    int D_runtime = Q.size(3);

    TORCH_CHECK(K.size(0) == B && K.size(1) == H && K.size(2) == N_runtime && K.size(3) == D_runtime,
                "K shape must match Q");
    TORCH_CHECK(V.size(0) == B && V.size(1) == H && V.size(2) == N_runtime && V.size(3) == D_runtime,
                "V shape must match Q");
    TORCH_CHECK(O.size(0) == B && O.size(1) == H && O.size(2) == N_runtime && O.size(3) == D_runtime,
                "O shape must match Q");
    TORCH_CHECK(dO.size(0) == B && dO.size(1) == H && dO.size(2) == N_runtime && dO.size(3) == D_runtime,
                "dO shape must match Q");
    TORCH_CHECK(L.size(0) == B && L.size(1) == H && L.size(2) == N_runtime,
                "L shape must be [B,H,N]");

    TORCH_CHECK(N_runtime > 0, "N must be > 0");

    if (D_runtime == 32) {
        return flash_bwd_impl<16, 32, 32>(Q, K, V, O, dO, L);
    }

    if (D_runtime == 64) {
        return flash_bwd_impl<16, 32, 64>(Q, K, V, O, dO, L);
    }

    if (D_runtime == 128) {
        return flash_bwd_impl<16, 32, 128>(Q, K, V, O, dO, L);
    }

    if (D_runtime == 256) {
        return flash_bwd_impl<16, 16, 256>(Q, K, V, O, dO, L);
    }

    TORCH_CHECK(false, "Unsupported D. Supported D: 32, 64, 128, 256");
}