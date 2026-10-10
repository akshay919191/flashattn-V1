#pragma once

#include "fwd_kernel.cuh"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <limits>
#include <vector>


// smem now has three regions: Q(Br) + K(Bc) + V(Bc), all with the same stride.
template<int D_PAD, int Bc>
static constexpr size_t smem_fwd() {
    constexpr int STRIDE = D_PAD + 8;
    constexpr int Br = 64;
    return static_cast<size_t>(Br + Bc + Bc) * STRIDE * sizeof(__half);
}


template<int D_PAD, int Bc, bool Masked,
         bool FULL_TILES = false>
static std::vector<torch::Tensor> launch_fwd_impl(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V
) {
    constexpr int Br = 64;

    constexpr size_t smem_bytes = smem_fwd<D_PAD, Bc>();

    const int B        = static_cast<int>(Q.size(0));
    const int H        = static_cast<int>(Q.size(1));
    const int Sq       = static_cast<int>(Q.size(2));
    const int Skv      = static_cast<int>(K.size(2));
    const int actual_D = static_cast<int>(Q.size(3));
    const int kvhead   = static_cast<int>(K.size(1));

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


// Bc selection rationale (Q decoupled, smem = (Br=64 + Bc + Bc) * (D_PAD+8) * 2):
//   Target: <= 50688B (49.5KB) so 2 blocks fit in 102.4KB L1/shared on Ampere.
//   Bc must also divide all power-of-2 sequence lengths — so Bc ∈ {16, 32, 64}.
//   D<=96:  Bc=64 -> <=39936B (39KB)  ✓
//   D=128:  Bc=32 -> 34816B  (34KB)   ✓  [Bc=48 was 42.5KB but 4096%48≠0 → silent zero output]
//   D=160:  Bc=32 -> 43008B  (42KB)   ✓
//   D=192:  Bc=32 -> 51200B  (50KB)   ✓  (exactly at limit)
//   D=224:  Bc=16 -> 44544B  (43.5KB) ✓  [Bc=32 -> 59392B too large]
//   D=256:  Bc=16 -> 50688B  (49.5KB) ✓
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
                K.size(2) % 32 == 0 &&
                Q.size(3) == 128) {
                return launch_fwd_impl<128, 32, Masked, true>(Q, K, V);
            }
            return launch_fwd_impl<128, 32, Masked, false>(Q, K, V);
        case 160: return launch_fwd_impl<160, 32, Masked>(Q, K, V);
        case 192: return launch_fwd_impl<192, 32, Masked>(Q, K, V);
        case 224: return launch_fwd_impl<224, 16, Masked>(Q, K, V);
        case 256: return launch_fwd_impl<256, 16, Masked>(Q, K, V);
    }
    TORCH_CHECK(false, "unsupported D_PAD");
    return {};
}
