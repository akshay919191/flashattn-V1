#pragma once

#include "bwd_kernel.cuh"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <limits>
#include <vector>
 

template<int Br, int Bc, int D_PAD, bool Masked>
static std::vector<torch::Tensor> launch_bwd_impl(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    const torch::Tensor& O,
    const torch::Tensor& dO,
    const torch::Tensor& L
) {
    const int B        = static_cast<int>(Q.size(0));
    const int H        = static_cast<int>(Q.size(1));
    const int Sq       = static_cast<int>(Q.size(2));
    const int Skv      = static_cast<int>(K.size(2));
    const int actual_D = static_cast<int>(Q.size(3));
    const int kvhead   = static_cast<int>(K.size(1));

    const int64_t total_rows_64 =
        static_cast<int64_t>(B) * H * Sq;
    TORCH_CHECK(
        total_rows_64 <= std::numeric_limits<int>::max(),
        "B*H*Sq is too large");
    const int total_rows = static_cast<int>(total_rows_64);

    TORCH_CHECK((Skv + Bc - 1) / Bc <= 65535,
                "Skv requires too many backward tiles");
    TORCH_CHECK((Sq  + Br - 1) / Br <= 65535,
                "Sq requires too many backward tiles");

    auto dQ    = torch::empty_like(Q);
    auto dK    = torch::empty_like(K);
    auto dV    = torch::empty_like(V);
    auto Delta = torch::empty(
        {Q.size(0), Q.size(1), Q.size(2)},
        Q.options().dtype(torch::kFloat32));

    dim3 block(128);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(Q.get_device());

    // 1) Delta kernel: Di = rowsum(O ∘ dO)
    const int rows_per_delta_block = 4;
    const int delta_blocks =
        (total_rows + rows_per_delta_block - 1) / rows_per_delta_block;

    flashattn_bwd_delta_kernel<D_PAD>
        <<<delta_blocks, block, 0, stream>>>(
            reinterpret_cast<const __half*>(O.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(dO.data_ptr<at::Half>()),
            Delta.data_ptr<float>(),
            actual_D, total_rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // 2) dK / dV kernel
    constexpr size_t dkdv_smem = flashattn_bwd_dkdv_smem_bytes<Br, Bc, D_PAD>();

    C10_CUDA_CHECK(cudaFuncSetAttribute(
        flashattn_bwd_dkdv_kernel<Br, Bc, D_PAD, Masked>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(dkdv_smem)));

    flashattn_bwd_dkdv_kernel<Br, Bc, D_PAD, Masked>
        <<<dim3(B, H, (Skv + Bc - 1) / Bc), block, dkdv_smem, stream>>>(
            reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(K.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(V.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(dO.data_ptr<at::Half>()),
            L.data_ptr<float>(),
            Delta.data_ptr<float>(),
            reinterpret_cast<      __half*>(dK.data_ptr<at::Half>()),
            reinterpret_cast<      __half*>(dV.data_ptr<at::Half>()),
            actual_D, Skv, Sq, kvhead);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // 3) dQ kernel
    constexpr size_t dq_smem = flashattn_bwd_dq_smem_bytes<Br, Bc, D_PAD>();

    C10_CUDA_CHECK(cudaFuncSetAttribute(
        flashattn_bwd_dq_kernel<Br, Bc, D_PAD, Masked>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(dq_smem)));

    flashattn_bwd_dq_kernel<Br, Bc, D_PAD, Masked>
        <<<dim3(B, H, (Sq + Br - 1) / Br), block, dq_smem, stream>>>(
            reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(K.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(V.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(dO.data_ptr<at::Half>()),
            L.data_ptr<float>(),
            Delta.data_ptr<float>(),
            reinterpret_cast<      __half*>(dQ.data_ptr<at::Half>()),
            actual_D, Skv, Sq, kvhead);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {dQ, dK, dV};
}
 

template<bool Masked>
static std::vector<torch::Tensor> dispatch_bwd(
    int d_pad,
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    const torch::Tensor& O,
    const torch::Tensor& dO,
    const torch::Tensor& L
) {
    switch (d_pad) {
        case  32: return launch_bwd_impl< 32, 32,  32, Masked>(Q, K, V, O, dO, L);
        case  64: return launch_bwd_impl< 32, 32,  64, Masked>(Q, K, V, O, dO, L);
        case  80: return launch_bwd_impl< 32, 32,  80, Masked>(Q, K, V, O, dO, L);
        case  96: return launch_bwd_impl< 32, 32,  96, Masked>(Q, K, V, O, dO, L);
        case 128: return launch_bwd_impl< 16, 32, 128, Masked>(Q, K, V, O, dO, L);
        case 160: return launch_bwd_impl< 16, 32, 160, Masked>(Q, K, V, O, dO, L);
        case 192: return launch_bwd_impl< 16, 16, 192, Masked>(Q, K, V, O, dO, L);
        case 224: return launch_bwd_impl< 16, 16, 224, Masked>(Q, K, V, O, dO, L);
        case 256: return launch_bwd_impl< 16, 16, 256, Masked>(Q, K, V, O, dO, L);
    }
    TORCH_CHECK(false, "unsupported D_PAD");
    return {};
}
