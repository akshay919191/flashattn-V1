#pragma once

#include "fwd_kernel.cuh"   // = flashattn_fwd_fixed.cuh

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <limits>
#include <vector>


template<int D_PAD, int Bc>
static constexpr size_t smem_fwd() {
    return flashattn_fwd_smem_bytes<64, Bc, D_PAD>();
}


template<int D_PAD, int Bc, bool Masked, bool FULL_TILES>
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

    // The kernel returns silently on these, which would leave O/L uninitialized.
    TORCH_CHECK(actual_D == D_PAD, "head dim ", actual_D, " != D_PAD ", D_PAD);
    TORCH_CHECK(kvhead > 0 && H % kvhead == 0, "num_heads must be a multiple of num_kv_heads");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(),
                "Q, K, V must be contiguous [B, H, S, D]");

    auto O = torch::empty_like(Q);
    auto L = torch::empty(
        {Q.size(0), Q.size(1), Q.size(2)},
        Q.options().dtype(torch::kFloat32));

    dim3 block(128);
    dim3 grid((Sq + Br - 1) / Br, H, B);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(Q.get_device());

    auto kernel = flashattn_fwd<Br, Bc, D_PAD, Masked, FULL_TILES>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));
    // Ask for the full shared-memory carveout so two blocks/SM can actually be resident.
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared));

    kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(O.data_ptr<at::Half>()),
        L.data_ptr<float>(),
        actual_D, Skv, Sq, kvhead);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {O, L};
}


// Picks the full-tile kernel whenever the shapes allow it; the ragged kernel
// (guarded loads + bounds masks) is only used when it is actually needed.
template<int D_PAD, int Bc, bool Masked>
static std::vector<torch::Tensor> launch_fwd_auto(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V
) {
    const bool full = (Q.size(2) % 64 == 0) && (K.size(2) % Bc == 0);
    return full ? launch_fwd_impl<D_PAD, Bc, Masked, true >(Q, K, V)
                : launch_fwd_impl<D_PAD, Bc, Masked, false>(Q, K, V);
}


// smem = max(Br, 2*Bc) * (D_PAD+8) * 2, because Q is staged in the K/V region.
// Budget for 2 blocks/SM on sm_86: (102400 / 2) - 1024 reserved = 50176 B per block.
//   D<=96:  Bc=64 -> 128 rows   D=96: 26624B
//   D=128:  Bc=32 ->  64 rows   17408B
//   D=160:  Bc=32 ->  64 rows   21504B
//   D=192:  Bc=32 ->  64 rows   25600B   (was 51200B: only 1 block/SM)
//   D=224:  Bc=32 ->  64 rows   29696B
//   D=256:  Bc=32 ->  64 rows   33792B   (was 50688B: only 1 block/SM)
// Bc=16 costs the same smem as Bc=32 now; check -Xptxas -v for spills at D=224/256
// and drop back to 16 if Bc=32 spills.
template<bool Masked>
static std::vector<torch::Tensor> dispatch_fwd(
    int d_pad,
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V
) {
    switch (d_pad) {
        case  32: return launch_fwd_auto< 32, 64, Masked>(Q, K, V);
        case  64: return launch_fwd_auto< 64, 64, Masked>(Q, K, V);
        case  80: return launch_fwd_auto< 80, 64, Masked>(Q, K, V);
        case  96: return launch_fwd_auto< 96, 64, Masked>(Q, K, V);
        case 128: return launch_fwd_auto<128, 32, Masked>(Q, K, V);
        case 160: return launch_fwd_auto<160, 32, Masked>(Q, K, V);
        case 192: return launch_fwd_auto<192, 32, Masked>(Q, K, V);
        case 224: return launch_fwd_auto<224, 32, Masked>(Q, K, V);
        case 256: return launch_fwd_auto<256, 32, Masked>(Q, K, V);
    }
    TORCH_CHECK(false, "unsupported D_PAD");
    return {};
}
