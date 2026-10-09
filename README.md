# FlashAttention CUDA Scratch

A from-scratch CUDA implementation of FlashAttention forward and backward. It uses inline PTX tensor-core MMA (`m16n8k16`), `cp.async` tile loads, online softmax, and a manually launched three-kernel backward pass.

This is a learning and profiling project, **not a production replacement** for PyTorch SDPA or the official FlashAttention implementations.

On the tested NVIDIA RTX 3050 6GB Laptop GPU (`sm_86`), the latest forward benchmark is close to SDPA for many dimensions up to `D=128`, but falls behind more clearly at `D=192` and `D=256`. The recorded backward results are substantially slower than SDPA, especially at longer sequence lengths. The tables below show the measured results and the hardware and implementation constraints that explain the gap.

## Quick start

### Requirements

- Linux and an NVIDIA GPU with compute capability 8.0 or newer.
- CUDA Toolkit with `nvcc` available on `PATH`, matching the CUDA version used by your PyTorch installation.
- Python 3.9+ and a CUDA-enabled PyTorch installation.
- Tested environment: Python 3.10, PyTorch 2.7.1+cu118, CUDA 11.8, RTX 3050 6GB Laptop GPU.

### Install

```bash
git clone https://github.com/akshay919191/flashattn-V1.git
cd flashattn-V1
pip install -e . --no-build-isolation
```

`--no-build-isolation` uses the PyTorch already installed in the active environment instead of building against an isolated environment that may use a different CUDA-enabled PyTorch package.

### Smoke test

```python
import torch
from flash_acc_reg import flash_attn

q = torch.randn(2, 16, 1024, 64, device="cuda", dtype=torch.float16, requires_grad=True)
k = torch.randn(2, 16, 1024, 64, device="cuda", dtype=torch.float16, requires_grad=True)
v = torch.randn(2, 16, 1024, 64, device="cuda", dtype=torch.float16, requires_grad=True)

out = flash_attn(q, k, v, causal=True)
out.sum().backward()

print(out.shape, q.grad.shape)
```

When running from the repository root, `import flash_acc_reg` can use the JIT fallback if the compiled extension is not available. The first JIT import compiles the CUDA extension and caches it under `~/.cache/torch_extensions`.

## API

### High-level autograd interface

```python
from flash_acc_reg import flash_attn

O = flash_attn(Q, K, V, causal=False)
O.backward(dO)
```

The wrapper validates the basic tensor properties and registers the custom forward and backward kernels with PyTorch autograd.

### Low-level kernel interface

```python
from flash_acc_reg import _C

O, L = _C.flash_fwd(Q, K, V, causal)
dQ, dK, dV = _C.flash_bwd(Q, K, V, O, dO, L, causal)
```

The argument order for `flash_bwd` is `Q, K, V, O, dO, L, causal`.

### Tensor shapes

```text
Q, O, dO, dQ: [B, H,    Sq,  D]
K, V:         [B, H_kv, Skv, D]
dK, dV:       [B, H_kv, Skv, D]
L:            [B, H,    Sq]
```

`H_kv` must divide `H`. `H_kv == H` is standard multi-head attention; `H_kv == 1` is multi-query attention. The backward returns `dK` and `dV` with `H_kv` heads. `L` is the per-query-row log-sum-exp (`m + log(l)`, FP32) consumed by the backward pass.

Pass `causal=True` or `causal=False` explicitly in tests and benchmarks.

## Current input constraints

| Property | Current behavior |
|---|---|
| Device | CUDA; kernels target compute capability 8.0+; benchmarked on `sm_86` |
| Input dtype | FP16 only |
| Accumulation | FP32 MMA accumulators and FP32 softmax statistics |
| Attention type | Self-attention and cross-attention; `Sq` may differ from `Skv` when the tile constraints below are met |
| GQA / MQA | Supported by the head mapping `kv_head = head / (H / H_kv)` |
| Masking | Unmasked or top-left causal (`key_index <= query_index`) |
| Dropout | Not supported |
| BF16 | Not supported |
| Variable-length packed sequences | Not supported |
| Sliding-window or bottom-right causal masking | Not supported |

**Important implementation constraints:** the current forward kernel assumes both `Sq` and `Skv` are multiples of 64. It also currently requires `D` to equal one of the dispatched tile dimensions: `{32, 64, 80, 96, 128, 160, 192, 224, 256}`. Although the dispatcher selects a padded dimension for other `D` values, the current forward kernel returns early when `actual_D != D_PAD`; those arbitrary dimensions are therefore not safely supported yet. The same early-return behavior applies when the forward sequence lengths are not multiples of 64. The high-level API does not currently turn all of these cases into explicit validation errors, so do not rely on an output from an unsupported shape.

The benchmark shapes in this README use supported tile sizes and sequence lengths divisible by 64.

## Project layout

```text
flashattn-V1/
├── pyproject.toml
├── setup.py
├── README.md
├── csrc/
│   ├── bindings.cpp       # pybind11 module
│   ├── flashattn.cu       # entry points and dispatch
│   ├── mma_helpers.cuh    # MMA, tile-load and layout helpers
│   ├── fwd_kernel.cuh    # forward kernel
│   ├── fwd_launch.cuh    # forward launch configuration
│   ├── bwd_kernel.cuh    # backward kernels
│   └── bwd_launch.cuh    # backward launch configuration
├── flash_acc_reg/
│   ├── __init__.py        # compiled extension or JIT fallback
│   ├── _jit.py            # torch.utils.cpp_extension.load
│   └── ops.py             # autograd wrapper and input validation
├── src/baseline/
├── tests/
└── benchmarks/
    ├── bench_fwd.py
    └── bench_bwd.py
```

## Implementation

### Forward

The forward kernel uses a `64 x 64` query/key tile configuration and 128 threads per CTA (four warps). It stages the query through the K shared-memory buffer, loads the query fragments into registers, and then reuses the K buffer for key tiles. The K and V tiles are **single-buffered**, not two full alternating K/V buffer sets. `cp.async` loads are overlapped with portions of the MMA and softmax work, with explicit `wait_group` calls and barriers where shared-memory buffers are reused.

The main steps are:

1. Load Q, K, and V tiles.
2. Compute attention scores with tensor-core MMA (`m16n8k16`) and FP32 accumulators.
3. Apply scaling and the optional top-left causal mask.
4. Update the online softmax maximum and normalization sum.
5. Compute the probability-weighted V contribution with MMA and accumulate the output.
6. Store the output and per-row log-sum-exp for backward.

The shared-memory footprint for the current forward layout is:

`2 * Bc * (D_PAD + 8) * sizeof(fp16)`, with `Bc = 64`.

| `D_PAD` | Shared memory per CTA |
|---:|---:|
| 32 | 10 KiB |
| 64 | 18 KiB |
| 80 | 22 KiB |
| 96 | 26 KiB |
| 128 | 34 KiB |
| 160 | 42 KiB |
| 192 | 50 KiB |
| 224 | 58 KiB |
| 256 | 66 KiB |

The exact values are rounded here to the nearest KiB; the source formula includes the padded shared-memory stride.

### Backward

Backward is split into three kernels:

1. **Delta:** `Delta = rowsum(O * dO)`.
2. **dK/dV:** one block owns each KV tile and writes its gradient tile.
3. **dQ:** one block owns each query tile and writes its gradient tile.

This ownership strategy avoids atomics and the need to pre-zero `dQ`, `dK`, and `dV`. Causally dead KV tiles are explicitly written as zeros by their owning block. The trade-off is that the separate dQ and dK/dV kernels reread operands rather than computing all gradients in one fused pass.

The backward math follows the usual attention derivatives:

```text
S   = Q K^T
P   = exp(S * scale - L)
dV  = P^T @ dO
dP  = dO @ V^T
dS  = P * (dP - Delta) * scale
dQ  = dS @ K
dK  = dS^T @ Q
```

## Correctness and validation

An earlier recorded test sweep reported 18/18 modes passing across nine shapes and self-attention/cross-attention, causal/unmasked cases. The recorded maximum absolute errors against a float32 PyTorch reference were:

| Tensor | Max absolute error |
|---|---:|
| `O` | 0.0012186 |
| `L` | 0.0000014 |
| `dQ` | 0.0014327 |
| `dK` | 0.0014586 |
| `dV` | 0.0017262 |

These are historical figures from the previously recorded sweep. The latest source changes to the forward tile loader and shared-memory sizing have been exercised by the forward benchmark, but a full correctness sweep after those changes has not been established by the benchmark log alone. Re-run the tests on the exact checkout before treating these error figures as validation of that revision:

```bash
pytest tests/ -v
```

## Benchmark environment

```text
GPU:        NVIDIA GeForce RTX 3050 6GB Laptop GPU (GA107, sm_86)
CUDA:       11.8
PyTorch:    2.7.1+cu118
Input dtype: FP16
SDPA:       PyTorch backend auto
Heads:      H = 16
Workload:   B * N = 16384; B = 128/64/32/16/8/4 for N = 128/256/512/1024/2048/4096
```

Run the benchmarks with:

```bash
python -m benchmarks.bench_fwd
python -m benchmarks.bench_bwd
```

The metric in the tables is `speedup = t_SDPA / t_extension`; values above 1 mean the extension was faster in that cell. Causal TFLOPs use effective masked FLOPs. For small sequence lengths, timings are strongly affected by launch and wrapper overhead; isolated speedups above 1 should not be treated as evidence of a sustained compute-throughput advantage.

**Data provenance:** the forward tables and forward TFLOPs below use the latest supplied `bench_fwd` run. The backward tables and backward TFLOPs are retained from the earlier recorded sweep and were not refreshed by that forward-only run.

## Results

### Forward speedup — unmasked (SDPA / extension)

| `D \ N` | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---:|---:|---:|---:|---:|---:|---:|
| 32  | 1.02 | 1.05 | 1.06 | 1.01 | 1.00 | 0.99 |
| 64  | 1.27 | 1.10 | 0.96 | 0.96 | 0.95 | 0.95 |
| 96  | 1.04 | 0.83 | 0.87 | 0.87 | 0.89 | 0.89 |
| 128 | 1.00 | 0.85 | 0.90 | 0.93 | 0.94 | 0.94 |
| 192 | 0.89 | 0.70 | 0.73 | 0.74 | 0.74 | 0.74 |
| 256 | 0.50 | 0.46 | 0.50 | 0.52 | 0.53 | 0.55 |

### Forward speedup — causal (SDPA / extension)

| `D \ N` | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---:|---:|---:|---:|---:|---:|---:|
| 32  | 1.01 | 1.04 | 1.15 | 1.07 | 1.02 | 0.99 |
| 64  | 1.19 | 1.38 | 1.22 | 1.03 | 0.99 | 0.98 |
| 96  | 1.23 | 1.29 | 0.90 | 0.94 | 0.95 | 0.98 |
| 128 | 1.23 | 1.56 | 1.21 | 0.94 | 0.98 | 1.00 |
| 192 | 0.95 | 0.95 | 0.85 | 0.79 | 0.79 | 0.77 |
| 256 | 0.72 | 0.72 | 0.60 | 0.54 | 0.51 | 0.48 |

### Backward speedup — unmasked (SDPA / extension)

| `D \ N` | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---:|---:|---:|---:|---:|---:|---:|
| 32  | 0.82 | 0.52 | 0.37 | 0.31 | 0.28 | 0.26 |
| 64  | 0.67 | 0.44 | 0.34 | 0.29 | 0.27 | 0.25 |
| 96  | 0.78 | 0.50 | 0.40 | 0.34 | 0.32 | 0.30 |
| 128 | 0.58 | 0.37 | 0.30 | 0.25 | 0.23 | 0.22 |
| 192 | 0.68 | 0.45 | 0.35 | 0.29 | 0.26 | 0.25 |
| 256 | 0.49 | 0.32 | 0.25 | 0.26 | 0.32 | 0.17 |

The recorded causal backward sweep is incomplete, so no full causal-backward speedup table is presented.

### Forward TFLOPs at `N = 4096` — unmasked

| `D` | Extension forward | SDPA forward |
|---:|---:|---:|
| 32  | 16.8 | 16.9 |
| 64  | 16.9 | 17.8 |
| 96  | 16.3 | 18.3 |
| 128 | 16.8 | 17.9 |
| 192 | 13.2 | 18.0 |
| 256 | 9.0 | 16.5 |

### Backward TFLOPs at `N = 4096` — unmasked, historical sweep

| `D` | Extension backward | SDPA backward |
|---:|---:|---:|
| 32  | 4.1 | 15.5 |
| 64  | 4.2 | 16.5 |
| 96  | 4.9 | 16.2 |
| 128 | 3.6 | 15.9 |
| 192 | 3.2 | 12.7 |
| 256 | 2.4 | 13.8 |

## Why it is slower than SDPA

The results reflect both the limits of the target GPU and the current kernel design. The hardware is not the only explanation: the performance gap at large head dimensions shows that this implementation does not keep the available hardware equally busy.

### Hardware and shared-memory limits

The benchmark runs on a GA107 laptop GPU with compute capability 8.6. NVIDIA documents a maximum of 100 KiB shared memory per SM and 99 KiB per thread block for compute capability 8.6; the A100's compute-capability-8.0 configuration permits a larger shared-memory budget. See the [NVIDIA Ampere tuning guide](https://docs.nvidia.com/cuda/archive/12.5.1/ampere-tuning-guide/index.html#occupancy).

The current forward CTA uses a 64-row tile, 128 threads, one K tile and one V tile in shared memory. At `D_PAD=192`, those tiles use about 50 KiB; at `D_PAD=256`, about 66 KiB. That makes it difficult to keep multiple CTAs resident on an SM, especially at the larger dimensions. Fewer resident warps means fewer independent warps available to cover shared-memory, instruction and tensor-core latency. Register use and other block resources also affect the actual occupancy, so the shared-memory calculation is a constraint, not a complete occupancy prediction.

The RTX 3050's 6 GB of VRAM is mainly a capacity limit: it constrains how large a batch or workload can fit. It is not, by itself, the explanation for low measured TFLOPs in these benchmark cases. The relevant performance limits are compute throughput, memory traffic, occupancy, and synchronization.

### Forward: near parity through `D=128`, larger gap at `D=192/256`

At `N=4096`, the extension reaches 16.3–16.9 TFLOPs for `D=32–128`, close to SDPA's 16.9–18.3 TFLOPs in the same run. The gap is more pronounced at `D=192` (13.2 vs 18.0 TFLOPs) and `D=256` (9.0 vs 16.5 TFLOPs). This pattern is consistent with larger MMA fragments, higher shared-memory use and register pressure reducing how efficiently the custom kernel uses the GPU.

The kernel has one K and one V shared-memory tile rather than multiple complete buffer sets. It overlaps some `cp.async` loads with compute, but waits and barriers are still needed before a shared-memory tile can be consumed or reused. The current pipeline does not hide all of that latency. For short sequences, launch and framework overhead is a large fraction of total time; this is why small-N speedup ratios fluctuate and should not be extrapolated to long sequences.

### Backward: data movement and synchronization dominate

In backward, the implementation performs several matrix operations for every Q/K tile pair. Intermediate probability and gradient-score values move through shared memory to bridge accumulator and MMA operand layouts, with barriers around those conversions. This adds work beyond the tensor-core operations themselves.

The three-kernel split gives each output tile a single owner and avoids atomics, but it also causes the dQ kernel and dK/dV kernel to reread operands. That increases memory traffic compared with a well-fused implementation. The historical benchmark reflects this: at `N=4096`, the extension records 2.4–4.9 TFLOPs while SDPA records 12.7–16.5 TFLOPs. The gap is therefore not explained solely by a lower theoretical compute ceiling; the current backward algorithm makes less effective use of the available compute and memory system.

### What the numbers do and do not show

- Forward is close to SDPA for many `D <= 128` long-sequence cases, but is slower for most `D=192` and `D=256` cases.
- At `N=4096`, the unmasked forward speedups are `0.99x`, `0.95x`, `0.89x`, `0.94x`, `0.74x`, and `0.55x` for `D=32,64,96,128,192,256` respectively.
- At `N=4096`, causal forward speedups are `0.99x`, `0.98x`, `0.98x`, `1.00x`, `0.77x`, and `0.48x` for those same dimensions.
- Some short-sequence cells show the extension faster than SDPA, but they are more sensitive to launch and measurement overhead and do not establish a general advantage.
- The backward results are from an earlier sweep and should be rerun before being compared with a new SDPA or driver/PyTorch environment.
- Results apply to the GPU, shapes, precision, masking, software versions and backend listed above. They are not a claim about all NVIDIA GPUs or all attention workloads.

## Build options

| Variable | Default | Effect |
|---|---|---|
| `TORCH_CUDA_ARCH_LIST` | Detected from the visible GPU, otherwise project default | Target architecture(s), for example `"8.0;8.6;8.9"` |
| `FLASH_DEBUG` | `0` | `1` enables a debug build with `-O0 -G` for tools such as `compute-sanitizer` and `cuda-gdb` |
| `FLASH_FAST_MATH` | `1` | `0` disables `--use_fast_math` |
| `MAX_JOBS` | Build-system default | Limits parallel compilation jobs |

Kernels are intended for `sm_80+`. The active forward implementation has the shape restrictions documented above; do not assume unsupported inputs are rejected safely.

## Clean rebuild

```bash
pip uninstall -y flash-acc-reg
rm -rf build *.egg-info
find . -name '*.so' -delete
rm -rf ~/.cache/torch_extensions/*/flash_acc_reg_jit
pip install -e . --no-build-isolation
```

## Troubleshooting

| Symptom | Likely cause or action |
|---|---|
| `nvcc not found` | Install the CUDA Toolkit and put its `bin` directory on `PATH`. |
| PyTorch/CUDA version mismatch | Compare `nvcc --version` with `python -c "import torch; print(torch.version.cuda)"`. |
| `dynamic module does not define module export function` | Check that the module name in `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` matches the built extension and remove stale `.so` files. |
| Old behavior after editing CUDA sources | Clean the build and JIT cache, then rebuild. |
| Import failure after a build error | Fix the earlier compiler error first; the import error is often a consequence of the missing extension. |
| Unsupported `D`, `Sq`, or `Skv` | Use one of the supported forward tile dimensions and sequence lengths divisible by 64; current early returns may otherwise leave output unwritten. |

## Profiling

With Nsight Compute installed, a basic profiling command is:

```bash
sudo env PYTHONPATH=. PATH="$PATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  /usr/local/cuda-11.8/bin/ncu \
  --section SpeedOfLight \
  --section Occupancy \
  --section SchedulerStats \
  --section WarpStateStats \
  --section MemoryWorkloadAnalysis \
  --launch-skip 20 \
  --launch-count 1 \
  --target-processes all \
  python -m benchmarks.bench_bwd
```

Counters worth inspecting include tensor-pipeline activity, achieved occupancy, shared-memory bank conflicts, and warp-stall reasons such as `long_scoreboard`, `barrier` and `short_scoreboard`. The build keeps `-lineinfo` so Nsight Compute can associate instructions with source lines.

## Repository hygiene

Do not commit generated build artifacts:

```text
build/
dist/
*.egg-info/
*.so
*.o
*.ncu-rep
*.nsys-rep
__pycache__/
.pytest_cache/
.torch_extensions/
```

## References

- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, 2022.
- Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*, 2023.
- [NVIDIA Ampere GPU Architecture Tuning Guide](https://docs.nvidia.com/cuda/archive/12.5.1/ampere-tuning-guide/index.html#occupancy).