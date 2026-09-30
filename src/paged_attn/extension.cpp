// extension.cpp -- Torch C++ bindings for the paged-attention CUDA kernels.
//
// Kernel implementations live in paged_kernel.cu.  helper.cuh is included
// there and must sit next to that file.

#include <torch/extension.h>
#include <vector>


// Forward self-attention.  Returns {O, Lse}.
std::vector<torch::Tensor> paged_attn_fwd_cuda(
    const torch::Tensor& Q,
    const torch::Tensor& Kpool,
    const torch::Tensor& Vpool,
    const torch::Tensor& blocktable,
    const torch::Tensor& seq_lens,
    const torch::Tensor& queryloc,
    int64_t max_q_len,
    double  scale,
    bool    causal,
    bool    return_lse);

// Backward.  Returns {dQ, dKpool, dVpool}.
// Requires the forward output O (needed for D_i = rowsum(O * dO)).
std::vector<torch::Tensor> paged_attn_bwd_cuda(
    const torch::Tensor& Q,
    const torch::Tensor& Kpool,
    const torch::Tensor& Vpool,
    const torch::Tensor& blocktable,
    const torch::Tensor& seq_lens,
    const torch::Tensor& queryloc,
    const torch::Tensor& dO,
    const torch::Tensor& Lse,
    const torch::Tensor& O,
    int64_t max_q_len,
    double  scale,
    bool    causal);

// Cross-attention forward (never causal).  Returns O.
torch::Tensor paged_cross_attn_fwd_cuda(
    const torch::Tensor& Q,
    const torch::Tensor& Kpool,
    const torch::Tensor& Vpool,
    const torch::Tensor& blocktable,
    const torch::Tensor& enc_lens,
    const torch::Tensor& queryloc,
    int64_t max_q_len,
    double  scale);

// ---------------------------------------------------------------------------
// Python bindings
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Paged attention forward/backward CUDA kernels (sm_80+)";

    m.def(
        "paged_attn_fwd",
        &paged_attn_fwd_cuda,
        "Paged self-attention forward. Returns (O, Lse).",
        py::arg("Q"),
        py::arg("Kpool"),
        py::arg("Vpool"),
        py::arg("blocktable"),
        py::arg("seq_lens"),
        py::arg("queryloc"),
        py::arg("max_q_len"),
        py::arg("scale")       = 0.0,
        py::arg("causal")      = true,
        py::arg("return_lse")  = true);

    m.def(
        "paged_attn_bwd",
        &paged_attn_bwd_cuda,
        "Paged self-attention backward. Returns (dQ, dKpool, dVpool).",
        py::arg("Q"),
        py::arg("Kpool"),
        py::arg("Vpool"),
        py::arg("blocktable"),
        py::arg("seq_lens"),
        py::arg("queryloc"),
        py::arg("dO"),
        py::arg("Lse"),
        py::arg("O"),
        py::arg("max_q_len"),
        py::arg("scale")  = 0.0,
        py::arg("causal") = true);

    m.def(
        "paged_cross_attn_fwd",
        &paged_cross_attn_fwd_cuda,
        "Paged cross-attention forward (never causal). Returns O.",
        py::arg("Q"),
        py::arg("Kpool"),
        py::arg("Vpool"),
        py::arg("blocktable"),
        py::arg("enc_lens"),
        py::arg("queryloc"),
        py::arg("max_q_len"),
        py::arg("scale") = 0.0);
}