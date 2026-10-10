#include "fwd_launch.cuh"
#include "bwd_launch.cuh"

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include <vector>


static int select_d_pad(int d) {
    if (d <=  32) return  32;
    if (d <=  64) return  64;
    if (d <=  80) return  80;
    if (d <=  96) return  96;
    if (d <= 128) return 128;
    if (d <= 160) return 160;
    if (d <= 192) return 192;
    if (d <= 224) return 224;
    if (d <= 256) return 256;
    TORCH_CHECK(false, "head dimension must be <= 256");
    return 0;
}

static void check_qkv(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V
) {
    TORCH_CHECK(Q.is_cuda(),   "Q must be CUDA");
    TORCH_CHECK(K.is_cuda(),   "K must be CUDA");
    TORCH_CHECK(V.is_cuda(),   "V must be CUDA");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat16, "Q must be float16");
    TORCH_CHECK(K.scalar_type() == torch::kFloat16, "K must be float16");
    TORCH_CHECK(V.scalar_type() == torch::kFloat16, "V must be float16");
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");
    TORCH_CHECK(Q.dim() == 4, "Q must have shape [B,H,Sq,D]");
    TORCH_CHECK(K.dim() == 4, "K must have shape [B,H,Skv,D]");
    TORCH_CHECK(V.dim() == 4, "V must have shape [B,H,Skv,D]");
    TORCH_CHECK(Q.device() == K.device(), "Q and K must be on the same CUDA device");
    TORCH_CHECK(Q.device() == V.device(), "Q and V must be on the same CUDA device");
    TORCH_CHECK(Q.size(0) == K.size(0),   "Q and K batch dimensions must match");
    TORCH_CHECK(Q.size(0) == V.size(0),   "Q and V batch dimensions must match");
    TORCH_CHECK(K.size(2) == V.size(2),   "K and V sequence lengths must match");
    TORCH_CHECK(Q.size(3) == K.size(3),   "Q and K head dimensions must match");
    TORCH_CHECK(Q.size(3) == V.size(3),   "Q and V head dimensions must match");
    TORCH_CHECK(Q.size(0) > 0, "B must be positive");
    TORCH_CHECK(Q.size(1) > 0, "H must be positive");
    TORCH_CHECK(K.size(1) > 0, "H_kv must be positive");
    TORCH_CHECK(K.size(1) <= Q.size(1), "H_kv must be <= H");
    TORCH_CHECK(Q.size(1) % K.size(1) == 0,
                "H must be divisible by H_kv (GQA)");
    TORCH_CHECK(Q.size(2) > 0, "Sq must be positive");
    TORCH_CHECK(K.size(2) > 0, "Skv must be positive");
    TORCH_CHECK(Q.size(3) > 0, "D must be positive");
    TORCH_CHECK(Q.size(3) <= 256, "D must be <= 256");
    TORCH_CHECK(Q.size(0) <= std::numeric_limits<int>::max(), "B is too large");
    TORCH_CHECK(Q.size(1) <= 65535, "H is too large");
    TORCH_CHECK(Q.size(2) <= std::numeric_limits<int>::max(), "Sq is too large");
    TORCH_CHECK(K.size(2) <= std::numeric_limits<int>::max(), "Skv is too large");
    TORCH_CHECK((Q.size(2) + 63) / 64 <= 65535,
                "Sq requires too many forward tiles");
}

static void check_backward_inputs(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    const torch::Tensor& O,
    const torch::Tensor& dO,
    const torch::Tensor& L
) {
    check_qkv(Q, K, V);
    TORCH_CHECK(O.is_cuda(),   "O must be CUDA");
    TORCH_CHECK(dO.is_cuda(),  "dO must be CUDA");
    TORCH_CHECK(L.is_cuda(),   "L must be CUDA");
    TORCH_CHECK(O.device()  == Q.device(), "O must be on the same CUDA device as Q");
    TORCH_CHECK(dO.device() == Q.device(), "dO must be on the same CUDA device as Q");
    TORCH_CHECK(L.device()  == Q.device(), "L must be on the same CUDA device as Q");
    TORCH_CHECK(O.scalar_type()  == torch::kFloat16, "O must be float16");
    TORCH_CHECK(dO.scalar_type() == torch::kFloat16, "dO must be float16");
    TORCH_CHECK(L.scalar_type()  == torch::kFloat32, "L must be float32");
    TORCH_CHECK(O.is_contiguous(),  "O must be contiguous");
    TORCH_CHECK(dO.is_contiguous(), "dO must be contiguous");
    TORCH_CHECK(L.is_contiguous(),  "L must be contiguous");
    TORCH_CHECK(O.sizes()  == Q.sizes(), "O shape must match Q");
    TORCH_CHECK(dO.sizes() == Q.sizes(), "dO shape must match Q");
    TORCH_CHECK(L.dim() == 3, "L must have shape [B,H,Sq]");
    TORCH_CHECK(L.size(0) == Q.size(0), "L batch dimension must match Q");
    TORCH_CHECK(L.size(1) == Q.size(1), "L head count must match Q");
    TORCH_CHECK(L.size(2) == Q.size(2), "L sequence length must match Q");
}


std::vector<torch::Tensor> flash_fwd_cuda(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    bool causal
) {
    check_qkv(Q, K, V);
    c10::cuda::CUDAGuard device_guard(Q.device());
    const int d_pad = select_d_pad(static_cast<int>(Q.size(3)));
    if (causal) return dispatch_fwd<true> (d_pad, Q, K, V);
    return             dispatch_fwd<false>(d_pad, Q, K, V);
}

std::vector<torch::Tensor> flash_bwd_cuda(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    const torch::Tensor& O,
    const torch::Tensor& dO,
    const torch::Tensor& L,
    bool causal
) {
    check_backward_inputs(Q, K, V, O, dO, L);
    c10::cuda::CUDAGuard device_guard(Q.device());
    const int d_pad = select_d_pad(static_cast<int>(Q.size(3)));
    if (causal) return dispatch_bwd<true> (d_pad, Q, K, V, O, dO, L);
    return             dispatch_bwd<false>(d_pad, Q, K, V, O, dO, L);
}
