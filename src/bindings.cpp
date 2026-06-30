#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> flash_fwd(torch::Tensor Q,
                                     torch::Tensor K,
                                     torch::Tensor V);

std::vector<torch::Tensor> flash_bwd(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor dO,
    torch::Tensor L
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_fwd", &flash_fwd, "FlashAttention forward kernel");
    m.def("flash_bwd", &flash_bwd, "FlashAttention backward kernel");
}