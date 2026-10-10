#pragma once

#include "fwd_kernel.cuh"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <limits>
#include <vector>

template<int D_PAD, int Bc, bool Masked,
         bool FULL_TILES = false>
static std::vector<torch::Tensor> launch_fwd_impl(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V
) {
    constexpr int Br      = 64;
    constexpr int PAD     = 8;
    constexpr int D_STRIDE = D_PAD + PAD;
    constexpr size_t smem_bytes =
        (Br * D_STRIDE + 4 * Bc * D_STRIDE) * sizeof(__half) + 256;

    const int B       = static_cast<int>(Q.size(0));
    const int H       = static_cast<int>(Q.size(1));
    const int Sq      = static_cast<int>(Q.size(2));
    const int Skv     = static_cast<int>(K.size(2));
    const int actual_D = static_cast<int>(Q.size(3));
    const int kvhead  = static_cast<int>(K.size(1));

    auto O = torch::empty_like(Q);
    auto L = torch::empty(
        {Q.size(0), Q.size(1), Q.size(2)},
        Q.options().dtype(torch::kFloat32));

    dim3 block(128);
    dim3 grid((Sq + Br - 1) / Br, H, B);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(Q.get_device());

    C10_CUDA_CHECK(cudaFuncSetAttribute(
        flashattn_fwd<Br, Bc, D_PAD, Masked, FULL_TILES>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));

    flashattn_fwd<Br, Bc, D_PAD, Masked, FULL_TILES>
        <<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(K.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(V.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(O.data_ptr<at::Half>()),
            L.data_ptr<float>(),
            actual_D, Skv, Sq, kvhead);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {O, L};
}


template<bool Masked>
static std::vector<torch::Tensor> dispatch_fwd(
    int d_pad,
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V
) {
    switch (d_pad) {
        case  32: return launch_fwd_impl< 32, 64, Masked>(Q, K, V);
        case  64: return launch_fwd_impl< 64, 64, Masked>(Q, K, V);
        case  80: return launch_fwd_impl< 80, 64, Masked>(Q, K, V);
        case  96: return launch_fwd_impl< 96, 64, Masked>(Q, K, V);
        case 128:
        if (Q.size(2) % 64 == 0 &&
            K.size(2) % 64 == 0 &&
            Q.size(3) == 128) {
            return launch_fwd_impl<128, 64, Masked, true>(Q, K, V);
        }

        return launch_fwd_impl<128, 64, Masked>(Q, K, V);
        case 160: return launch_fwd_impl<160, 32, Masked>(Q, K, V);
        case 192: return launch_fwd_impl<192, 32, Masked>(Q, K, V);
        case 224: return launch_fwd_impl<224, 32, Masked>(Q, K, V);
        case 256: return launch_fwd_impl<256, 16, Masked>(Q, K, V);
    }
    TORCH_CHECK(false, "unsupported D_PAD");
    return {};
}
