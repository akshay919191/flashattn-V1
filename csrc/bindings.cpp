#include <torch/extension.h>

#include <vector>

namespace py = pybind11;

std::vector<torch::Tensor> flash_fwd_cuda(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    bool causal
);

std::vector<torch::Tensor> flash_bwd_cuda(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    const torch::Tensor& O,
    const torch::Tensor& dO,
    const torch::Tensor& L,
    bool causal
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "flash_fwd",
        &flash_fwd_cuda,
        py::arg("Q"),
        py::arg("K"),
        py::arg("V"),
        py::arg("causal") = false
    );
    m.def(
        "flash_bwd",
        &flash_bwd_cuda,
        py::arg("Q"),
        py::arg("K"),
        py::arg("V"),
        py::arg("O"),
        py::arg("dO"),
        py::arg("L"),
        py::arg("causal") = false
    );
}