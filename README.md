# flashattn-scratch

A from-scratch CUDA implementation of the FlashAttention forward pass written
in CUDA C++ with inline PTX. Uses `mma.sync` tensor-core instructions
(`m16n8k16`), `cp.async` tile loads, and online softmax entirely in registers.

This is a learning and profiling project, not a production replacement for
PyTorch SDPA or the official FlashAttention libraries.

## Quick start

### Requirements

- Linux, NVIDIA GPU with compute capability 8.0 or newer.
- CUDA Toolkit with `nvcc` on `PATH`, matching the CUDA version used by your
  PyTorch install.
- Python 3.9+ and a CUDA-enabled PyTorch install.
- Tested: Python 3.10, PyTorch 2.7.1+cu118, CUDA 11.8, RTX 3050 6GB Laptop.

### Install

```bash
git clone https://github.com/akshay919191/flashattn-V1.git
cd flashattn-V1
pip install -e . --no-build-isolation
```

### Smoke test

```python
import torch
from flash_acc_reg import flash_attn

q = torch.randn(2, 16, 1024, 64, device="cuda", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

out = flash_attn(q, k, v, causal=True)
print(out.shape)   # torch.Size([2, 16, 1024, 64])
```

## API

### High-level autograd wrapper

```python
from flash_acc_reg import flash_attn
O = flash_attn(Q, K, V, causal=False)
```

### Low-level kernel interface

```python
from flash_acc_reg import _C
O, L = _C.flash_fwd(Q, K, V, causal)
```

`L` is the per-row log-sum-exp (`m + log(l)`, FP32) needed for gradient
computation.

### Tensor shapes

```
Q, O:  [B, H,    Sq,  D]
K, V:  [B, H_kv, Skv, D]
L:     [B, H,    Sq]
```

`H_kv` must divide `H`. `H_kv == H` is standard MHA; `H_kv == 1` is MQA.

## Supported inputs

| Property | Status |
|---|---|
| Device | CUDA, compute capability 8.0+ |
| Dtype | FP16 |
| Accumulation | FP32 (MMA accumulators and softmax statistics) |
| Masking | None or top-left causal |
| GQA / MQA | Supported |
| Cross-attention | Supported (`Sq ≠ Skv` when tile constraints are met) |
| BF16, dropout, variable-length sequences | Not implemented |

**Shape constraints:** `Sq` and `Skv` must both be multiples of 64. `D` must
be one of `{32, 64, 80, 96, 128, 160, 192, 224, 256}`. Inputs outside these
constraints trigger an early return and produce unwritten output; validate
before calling.

## Implementation

Four warps per block (128 threads), `m16n8k16` MMA with FP32 accumulators.

**Shared-memory layout:** three separate regions — `Q` (Br rows), `K` (Bc
rows), `V` (Bc rows) — each with a padded stride of `D + 8` elements to avoid
bank conflicts. Decoupling Q into its own buffer removes the `Br ≤ Bc`
constraint from the original design and allows Bc to be tuned independently
per head dimension.

**Pipeline:** Q is loaded once into its dedicated buffer and read into
registers before the kv-loop starts. Inside the loop, the next K tile is
prefetched with `cp.async` while the current V tile's PV MMA is in flight,
hiding most of the global-memory load latency.

**Bc dispatch:** Bc is chosen separately for causal and non-causal to
maximize either occupancy (causal) or tile size (non-causal):

| `D` | Causal `Bc` | Shared mem | Non-causal `Bc` | Shared mem |
|---:|---:|---:|---:|---:|
|  32 | 64 | 15 KiB | 64 | 15 KiB |
|  64 | 64 | 27 KiB | 64 | 27 KiB |
|  96 | 64 | 39 KiB | 64 | 39 KiB |
| 128 | 32 | 34 KiB | 64 | 51 KiB |
| 192 | 16 | 38 KiB | 64 | 75 KiB |
| 256 | 16 | 50 KiB | 64 | 99 KiB |

For causal attention, small Bc lets two blocks reside on the same SM at
`D ≤ 224` (GA107 limit: 96 KiB shared per SM). Causal tiles naturally skip
roughly half the kv-loop iterations, so the extra iterations from small Bc
cost less than the occupancy gain pays back in latency hiding. For non-causal
attention, all tiles are processed; fewer iterations from large Bc reduces
`__syncthreads()` overhead more than the second resident block would help.

## Benchmark

**GPU:** NVIDIA GeForce RTX 3050 6GB Laptop (GA107, sm\_86), CUDA 11.8,
PyTorch 2.7.1+cu118, FP16. SDPA backend: `auto`. 32 heads. Workload holds
`B × N = 16384` constant (`B = 128/64/32/16/8/4` for `N = 128/256/512/1024/2048/4096`).

Metric: `speedup = t_SDPA / t_ext`. Values above 1.00 mean the custom kernel
was faster.

```
python -m benchmarks.bench_fwd
```

### Causal — speedup vs SDPA

| `D \ N` | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---:|---:|---:|---:|---:|---:|---:|
|  32 | 1.01 | 1.05 | **1.20** | 1.11 | 1.04 | 1.01 |
|  64 | 1.19 | **1.39** | 1.24 | 1.05 | 1.00 | 1.00 |
|  96 | 1.23 | 1.29 | 0.90 | 0.93 | 0.95 | 0.98 |
| 128 | 1.23 | **1.55** | 1.22 | 0.95 | 0.97 | 0.99 |
| 192 | 1.01 | 1.05 | 0.93 | 0.86 | 0.86 | 0.83 |
| 256 | 1.18 | 1.22 | 1.04 | 0.96 | 0.96 | 0.91 |

Causal is the stronger case. The kernel beats SDPA across all of D=32 and
D=64 for every measured sequence length, and wins at short-to-mid sequences
for D=96, D=128, and even D=256. The peak cell is D=128, N=256 at **1.55×**.

### Non-causal — speedup vs SDPA

| `D \ N` | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---:|---:|---:|---:|---:|---:|---:|
|  32 | 1.02 | 1.05 | 1.03 | 1.01 | 1.00 | 0.98 |
|  64 | **1.27** | 1.10 | 0.96 | 0.96 | 0.96 | 0.95 |
|  96 | 1.04 | 0.84 | 0.87 | 0.88 | 0.89 | 0.89 |
| 128 | 0.99 | 0.87 | 0.92 | 0.94 | 0.94 | 0.95 |
| 192 | 0.92 | 0.75 | 0.79 | 0.81 | 0.81 | 0.79 |
| 256 | 0.82 | 0.74 | 0.81 | 0.83 | 0.83 | 0.85 |

Non-causal is competitive at D=32 across all lengths and wins at small N for
D=64 and D=96. At larger D, non-causal processes every kv-tile and SDPA pulls
ahead, mainly because PyTorch Flash uses wider tile sizes enabled by its
larger shared-memory budget on the production build.

### TFLOPs at N=4096

| `D` | Custom (causal) | SDPA (causal) | Custom (non-causal) | SDPA (non-causal) |
|---:|---:|---:|---:|---:|
|  32 | 16.2 | 16.0 | 16.6 | 16.9 |
|  64 | 16.6 | 16.7 | 17.0 | 17.8 |
|  96 | 15.9 | 16.2 | 16.2 | 18.3 |
| 128 | 16.0 | 16.2 | 16.9 | 17.9 |
| 192 | 13.6 | 16.3 | 14.2 | 17.9 |
| 256 | 12.9 | 14.3 | 13.8 | 16.3 |

At D≤128 the kernel sustains 15.9–16.6 TFLOPs causal, within ~2% of SDPA.
The gap at D=192/256 is a shared-memory and occupancy constraint: the
GA107's 96 KiB per SM and the three-buffer layout prevent fitting more than
one resident block at those sizes.

## Why it falls behind at large D

The RTX 3050 has 96 KiB of shared memory per SM. With three buffers (Q, K, V)
at D=256, the causal path uses 50 KiB — leaving just enough room for one
block at that size and none of the inter-block latency hiding that benefits
smaller dimensions. The non-causal path uses 99 KiB at D=256 (the full SM
budget), which is one block in its entirety.

The gap is a hardware constraint, not an algorithmic one: a wider SM with more
shared memory (A100, H100) would push the crossover to larger D without any
kernel changes.

## Project layout

```
flashattn-V1/
├── csrc/
│   ├── fwd_kernel.cuh     # forward kernel (MMA, online softmax, masking)
│   ├── fwd_launch.cuh     # Bc dispatch, smem sizing, launch config
│   ├── bwd_kernel.cuh     # backward kernels (baseline)
│   ├── bwd_launch.cuh     # backward dispatch
│   ├── mma_helpers.cuh    # ldmatrix, tile loaders, pack helpers
│   ├── flashattn.cu       # entry points
│   └── bindings.cpp       # pybind11 module
├── flash_acc_reg/
│   ├── __init__.py        # compiled extension or JIT fallback
│   ├── _jit.py
│   └── ops.py             # autograd wrapper and input validation
├── tests/
└── benchmarks/
    └── bench_fwd.py
```

## Build options

| Variable | Default | Effect |
|---|---|---|
| `TORCH_CUDA_ARCH_LIST` | Detected from GPU | Target arch(s), e.g. `"8.0;8.6"` |
| `FLASH_DEBUG` | `0` | `1` → `-O0 -G` for `compute-sanitizer` / `cuda-gdb` |
| `FLASH_FAST_MATH` | `1` | `0` → disable `--use_fast_math` |

## Clean rebuild

```bash
pip uninstall -y flash-acc-reg
rm -rf build *.egg-info
find . -name '*.so' -delete
rm -rf ~/.cache/torch_extensions/*/flash_acc_reg_jit
pip install -e . --no-build-isolation
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `nvcc not found` | Add CUDA Toolkit `bin/` to `PATH` |
| PyTorch/CUDA version mismatch | Compare `nvcc --version` with `torch.version.cuda` |
| Zero output / `L` all zeros | Input shape hits an unsupported size — check `Sq % 64 == 0`, `Skv % 64 == 0`, and `D` is in the dispatch table |
| Old behavior after editing CUDA | Clean build and JIT cache, then reinstall |

## References

- Dao et al., *FlashAttention*, NeurIPS 2022.
- Dao, *FlashAttention-2*, ICLR 2024.
- NVIDIA, [Ampere Tuning Guide — Occupancy](https://docs.nvidia.com/cuda/archive/12.5.1/ampere-tuning-guide/index.html#occupancy).