# FlashAttention CUDA Scratch

A from-scratch CUDA implementation of FlashAttention forward and backward: tensor-core MMA (`m16n8k16`), `cp.async` tiled loads, double-buffered shared memory, online softmax in registers, and a manually launched three-kernel backward. Built for learning and profiling, not as a production replacement for PyTorch SDPA or FlashAttention.

Measured position vs SDPA (RTX 3050 6GB Laptop, sm_86): forward ~0.5-0.6x, backward ~0.2-0.4x. SDPA operates at the practical tensor-core ceiling of this GPU; the gap is quantified and explained in [Why it is slower than SDPA](#why-it-is-slower-than-sdpa).

## Quick start

Requirements:

- Linux, NVIDIA GPU with compute capability 8.0 or newer
- CUDA toolkit with `nvcc` on `PATH`, matching the CUDA version your PyTorch was built with
- Python 3.9+ and PyTorch 2.1+ with CUDA support (tested: Python 3.10, PyTorch 2.7.1+cu118, CUDA 11.8)

```bash
git clone https://github.com/<you>/flash-attention-scratch.git
cd flash-attention-scratch
pip install -e . --no-build-isolation
```

Smoke test:

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

No install? Run Python from the repository root and `import flash_acc_reg`. If the compiled module is not found, the kernels are compiled on first import (about a minute) and cached in `~/.cache/torch_extensions`.

## Supported inputs

| Property | Support |
|---|---|
| Device | CUDA, sm_80+ required (`cp.async`, `m16n8k16`) |
| Input dtype | FP16 (FP32 accumulation in softmax and MMA) |
| B, H, Sq, Skv | Runtime dynamic; cross-attention (`Sq != Skv`) supported |
| Masking | Unmasked; causal (top-left aligned, key <= query) |
| GQA / MQA | Yes. K/V may carry `H_kv` heads with `H % H_kv == 0` (`H_kv = H`: MHA, `H_kv = 1`: MQA) |
| Head dim D | Any integer 1..256 (no divisibility requirement); rounded up to `D_PAD` in {32, 64, 80, 96, 128, 160, 192, 224, 256} |
| Dropout | No |
| Variable-length packed sequences | No |
| Sliding window / bottom-right causal | No |

## Python API

### High level (autograd)

```python
from flash_acc_reg import flash_attn

O = flash_attn(Q, K, V, causal=False)
O.backward(dO)          # gradients flow through the custom backward kernels
```

`flash_attn` validates its inputs (CUDA tensors, 4D shape) and registers the forward and backward kernels with PyTorch autograd.

### Low level (raw kernels)

```python
from flash_acc_reg import _C

O, L = _C.flash_fwd(Q, K, V, causal)
dQ, dK, dV = _C.flash_bwd(Q, K, V, O, dO, L, causal)
```

Note the argument order of `flash_bwd`: `O`, then `dO`, then `L`.

### Shapes

```
Q, O, dO, dQ: [B, H,    Sq,  D]
K, V:         [B, H_kv, Skv, D]
dK, dV:       [B, H_kv, Skv, D]   (returned with H_kv heads)
L:            [B, H,    Sq]
```

`H_kv` must divide `H`; violations are rejected with explicit errors. `L` is the per-row log-sum-exp (`m + log(l)`, FP32) consumed by the backward.

Pass `causal=True/False` explicitly in tests and benchmarks.

## Project layout

```
flash-attention-scratch/
├── pyproject.toml
├── setup.py                 build script (AOT build, builds flash_acc_reg._C)
├── README.md
├── .gitignore
├── csrc/
│   ├── bindings.cpp         pybind11 module
│   └── flashattn.cu         forward and backward kernels + launchers
├── flash_acc_reg/           importable Python package
│   ├── __init__.py          loads _C, or falls back to the JIT build
│   ├── _jit.py              JIT build via torch.utils.cpp_extension.load
│   └── ops.py               autograd wrapper + input validation
├── src/baseline/            baseline kernels used for comparison
├── tests/                   correctness tests (pytest)
└── benchmarks/              bench_fwd.py, bench_bwd.py
```

## Current status

- forward: working (tested sweep below)
- backward: working (tested sweep below)
- causal: working (top-left mask)
- cross-attention: working for `Sq < Skv` and `Sq > Skv`
- dtype: FP16 only

Backward uses three kernels:

1. Delta kernel: `Delta = rowsum(O * dO)`
2. dK/dV kernel: one block per (batch, head, KV tile)
3. dQ kernel: one block per (batch, head, Q tile)

The split gives every output tile a single owning block: no atomics, plain stores, and causally-dead KV tiles store zeros themselves. dK/dV/dQ require no pre-zeroing.

## Correctness

The combined test compares O, L, dQ, dK, dV against a float32 PyTorch reference. Latest sweep: 9 shapes x 18 modes (self- and cross-attention, causal and unmasked) across all nine `D_PAD` values.

**Summary: 18/18 modes passed**

| Tensor | Max abs error |
|---|---|
| O | 0.0012186 |
| L | 0.0000014 |
| dQ | 0.0014327 |
| dK | 0.0014586 |
| dV | 0.0017262 |

These validate the tested inputs and tolerances; they are not a proof for every shape or distribution.

Run the tests (a GPU is required):

```bash
pytest tests/ -v
```

Re-run the sweep after any kernel change. The any-D loader, padded strides, and exclusive-store rework are covered by the same harness.

## Benchmark environment

```
GPU:        NVIDIA GeForce RTX 3050 6GB Laptop GPU (GA107, sm_86)
CUDA:       11.8
PyTorch:    2.7.1+cu118
dtype:      FP16
SDPA:       backend auto
Harness:    H = 16; token count held constant at B*N = 16384
            (B = 128/64/32/16/8/4 for N = 128/256/512/1024/2048/4096)
            causal TFLOPs counted with effective (masked) FLOPs
```

```bash
python -m benchmarks.bench_fwd
python -m benchmarks.bench_bwd
```

Metric: `speedup = t_SDPA / t_extension` (> 1 means the extension is faster). SDPA's forward+backward comparison includes its autograd graph; the extension's includes the forward kernel plus the three backward kernels.

## Results

### Forward speedup (SDPA / extension), unmasked

| D \ N | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|
| 32 | 0.64 | 0.45 | 0.51 | 0.54 | 0.57 | 0.59 |
| 64 | 0.81 | 0.52 | 0.49 | 0.53 | 0.55 | 0.57 |
| 96 | 0.64 | 0.43 | 0.48 | 0.51 | 0.52 | 0.53 |
| 128 | 0.65 | 0.46 | 0.50 | 0.54 | 0.56 | 0.57 |
| 192 | 0.64 | 0.51 | 0.53 | 0.54 | 0.55 | 0.55 |
| 256 | 0.65 | 0.55 | 0.56 | 0.56 | 0.56 | 0.57 |

### Forward speedup, causal

| D \ N | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|
| 32 | 0.76 | 0.57 | 0.55 | 0.56 | 0.58 | 0.60 |
| 64 | 0.89 | 0.75 | 0.61 | 0.56 | 0.57 | 0.59 |
| 96 | 0.90 | 0.70 | 0.48 | 0.52 | 0.54 | 0.57 |
| 128 | 0.95 | 0.87 | 0.66 | 0.54 | 0.58 | 0.61 |
| 192 | 0.75 | 0.68 | 0.61 | 0.57 | 0.57 | 0.57 |
| 256 | 0.93 | 0.84 | 0.70 | 0.65 | 0.64 | 0.64 |

### Backward speedup (SDPA / extension), unmasked

| D \ N | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|
| 32 | 0.82 | 0.52 | 0.37 | 0.31 | 0.28 | 0.26 |
| 64 | 0.67 | 0.44 | 0.34 | 0.29 | 0.27 | 0.25 |
| 96 | 0.78 | 0.50 | 0.40 | 0.34 | 0.32 | 0.30 |
| 128 | 0.58 | 0.37 | 0.30 | 0.25 | 0.23 | 0.22 |
| 192 | 0.68 | 0.45 | 0.35 | 0.29 | 0.26 | 0.25 |
| 256 | 0.49 | 0.32 | 0.25 | 0.26 | 0.32 | 0.17 |

### Backward speedup, causal (run interrupted; partial)

| D \ N | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|
| 32 | 1.09 | 0.70 | 0.50 | 0.38 | 0.32 | 0.29 |
| 64 | 0.91 | - | - | - | - | - |

### TFLOPs at N=4096 (compute-bound end of the sweep, unmasked)

| D | ext fwd | SDPA fwd | ext bwd | SDPA bwd |
|---|---|---|---|---|
| 32 | 10.0 | 17.0 | 4.1 | 15.5 |
| 64 | 10.1 | 17.8 | 4.2 | 16.5 |
| 96 | 9.7 | 18.3 | 4.9 | 16.2 |
| 128 | 10.2 | 17.8 | 3.6 | 15.9 |
| 192 | 9.9 | 18.0 | 3.2 | 12.7 |
| 256 | 9.4 | 16.6 | 2.4 | 13.8 |

### Reading of the data

- Forward plateaus at ~10 TFLOPs regardless of D; SDPA plateaus at 16.5-18.4.
- Backward plateaus at 2.4-4.9 TFLOPs; SDPA at 13-16.5 for D <= 128.
- At N <= 512 both sides are launch/overhead-floored; those cells measure overhead floors, not kernels.
- The single > 1x cell (backward causal, D=32, N=128) is that floor, not a kernel advantage.
- SDPA's backward drops at D = 192/256 (different backend route for large head dims) but still leads by 2-6x.

## Why it is slower than SDPA

### Hardware frame

On GA10x, dense FP16 tensor ops with FP32 accumulate run at 2x the FP32 rate; at this part's boost clocks that is about 18 TFLOPs.

SDPA's best cells sit essentially at that ceiling, so the reference here is hardware-bound and the entire gap is implementation headroom.

Consumer Ampere also caps dynamic shared memory at about 100 KB per SM (vs 164 KB on A100). That cap is the binding constraint throughout.

### Forward: ~0.5-0.6x, plateau ~10 TFLOPs

**Shared-memory capacity limits occupancy.**

One CTA is 128 threads (4 warps) holding Q + 2xK + 2xV tiles: about 25 KB/CTA at `D_PAD`=32, about 45 KB at 64, and 56-87 KB at 80-256. Against the ~100 KB/SM cap that yields 12 / 8 / 4 resident warps per SM.

Four warps cannot overlap `ldmatrix`, `mma`, and `cp.async` latencies; the tensor pipe stalls on shared-memory dependencies. This is the dominant forward gap and the reason the ratio is flat at ~0.5-0.6x for every D >= 80.

The two-stage K/V pipeline with `wait_group 0` drains the pipeline each iteration; deeper staging needs shared memory the budget does not have at large D.

Overhead floor at N <= 512 (launches, binding, Python) caps both implementations.

### Backward: ~0.2-0.4x, plateau ~2.4-4.9 TFLOPs

The causes are structural, not a matter of tuning.

**P and dS cross shared memory between every pair of GEMMs.** The MMA accumulator layout (C fragment) is not a valid A-operand layout, so each step does:

```
registers -> smem -> softmax/dS math -> smem -> ldmatrix -> mma
```

That is four extra full-tile shared-memory round trips and three `__syncthreads()` per (Q, K) tile pair in each kernel. FA2-class kernels keep P/dS register-resident with per-warp row ownership and shuffle-based layout conversion. This is the largest single backward cost.

**The ~100 KB cap forces tiny tiles.** Br=16-32, Bc=16-32 at `D_PAD` >= 128 with 2xQ + 2xdO staged for the pipeline. Per-step fixed costs (syncs, L/Delta staging, the four smem passes) amortize against Br x Bc x D mma FLOPs; at Br=16 that ratio is poor. (After the stride/bank rework, Br=Bc=64 at D=128 needs about 130 KB. That fits an A100, not GA10x.)

**The two-kernel split re-streams operands.** The dQ kernel reads all of K, V per Q tile; the dK/dV kernel reads all of Q, dO per KV tile. That is roughly 2x the HBM traffic of a fused backward. It is a deliberate trade to avoid atomics; SDPA's fused kernel reads each operand about once.

**Low occupancy.** About 8 warps/SM (2 CTAs x 4 warps): the forward's latency-hiding problem, compounded by more barriers per step.

### Not the cause

- FP16 P/dS storage (SDPA does the same)
- MMA shape (same `m16n8k16`)
- Accumulation precision (FP32 in both)
- Python binding at large N (noise)
- The loader (`cp.async` with hardware zero-fill; the scalar path only serves ragged `D % 8 != 0` tails)

### Already fixed and reflected in these numbers

- p/dS shared-memory bank conflicts (padded strides; the old layout serialized whole warps on single banks)
- FP16 atomics on dK/dV/dQ (replaced by exclusive-ownership stores)
- dK/dV zero-init memsets
- predicated `cp.async` fills
- unrolling
- `launch_bounds`

## Roadmap (ranked by expected payoff on this GPU)

1. Register-resident P/dS backward: per-warp row ownership, shuffle conversions, no smem round trips. Targets the 3-6x backward gap directly.
2. Occupancy-first tiling for the ~100 KB budget: shapes that keep >= 2 CTAs/SM at every D (smaller staging, split-D across warps) rather than 1 CTA x 4 warps.
3. Fused backward (dQ folded into the dK/dV kernel via FP32 workspace + reduction) to halve operand re-streaming.
4. Deeper `cp.async` pipeline where it fits (`D_PAD` <= 64 forward: 3-4 stages, `wait_group 1`).
5. Small-N floor: CUDA Graph capture of the 3-4 launches; `exp2f` with folded scale; `__half2` elementwise passes.

Any speed claim should be based on repeated measurements with the same shape, mask, device, PyTorch build, and SDPA backend.

## Build

### Option 1: install (recommended)

```bash
pip install -e . --no-build-isolation
```

`--no-build-isolation` makes the build use the PyTorch and CUDA already in your environment instead of downloading a second copy of PyTorch that may not match your CUDA version.

The extension compiles into the package as `flash_acc_reg._C`.

### Option 2: JIT (no install)

Run Python from the repository root and `import flash_acc_reg`. The first import compiles the kernels with `torch.utils.cpp_extension.load` and caches them in `~/.cache/torch_extensions`; later imports load instantly. This path needs `nvcc` at import time.

### Build options (environment variables)

| Variable | Default | Effect |
|---|---|---|
| `TORCH_CUDA_ARCH_LIST` | detected from the visible GPU, else `8.6` | Target architecture(s), e.g. `"8.0;8.6;8.9"` for a multi-arch build |
| `FLASH_DEBUG` | `0` | `1` builds with `-O0 -G` for `compute-sanitizer` and `cuda-gdb` |
| `FLASH_FAST_MATH` | `1` | `0` drops `--use_fast_math`; useful to see how much error fast math contributes |
| `MAX_JOBS` | all cores | Limit parallel compile jobs if the build runs out of RAM |

Kernels require sm_80+. On older architectures the kernels exit without raising an error and the outputs are left untouched, so check your GPU before trusting results.

### Clean rebuild

```bash
pip uninstall -y flash-acc-reg
rm -rf build *.egg-info
find . -name '*.so' -delete
rm -rf ~/.cache/torch_extensions/*/flash_acc_reg_jit
pip install -e . --no-build-isolation
```

### Troubleshooting

| Symptom | Likely cause |
|---|---|
| `nvcc not found` | CUDA toolkit missing or not on `PATH` |
| Version mismatch error from PyTorch | `nvcc` major version differs from the CUDA version PyTorch was built with (`python -c "import torch; print(torch.version.cuda)"`) |
| `dynamic module does not define module export function` | `bindings.cpp` does not use `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)`, or an old `.so` with a different name is being imported |
| Old behavior after editing a `.cu` file | A stale `.so` or JIT cache; do the clean rebuild above |
| `ImportError` for `flash_acc_reg_ext` | Old tests or benchmarks still import the previous module name; use `from flash_acc_reg import flash_attn` or `from flash_acc_reg import _C` |

## Implementation notes

### Forward

- tiled Q/K/V loads via `cp.async` (hardware zero-fill of out-of-range tiles)
- double-buffered K/V shared memory, 2-stage pipeline
- tensor-core `m16n8k16` score computation (FP32 accumulators)
- online softmax in registers (per-warp max/sum, quad reduction)
- tensor-core P @ V
- logsumexp written for backward

### Backward

```
S   = Q K^T
P   = exp(S * scale - L)
dV  = P^T @ dO
dP  = dO V^T
dS  = P * (dP - Delta) * scale
dQ  = dS @ K               (Q-tile-owned kernel)
dK  = dS^T @ Q             (KV-tile-owned kernel)
```

### Layout details that matter

- shared-memory strides padded (+8) to avoid bank conflicts, including the p/dS tile
  - stored `[k][q]` in dK/dV
  - stored `[q][k]` in dQ
- exclusive tile ownership gives plain stores, no atomics, no pre-zeroing
- GQA mapping: `kv_head = head / (H / H_kv)`
- host-side constexpr smem calculators mirror the kernel layouts and opt in via `cudaFuncSetAttribute` (> 48 KB dynamic smem requires it)

## Current limitations

- FP16 only
- CUDA only (sm_80+)
- top-left causal mask only
- no dropout
- no BF16
- no varlen
- no sliding window
- no bottom-right rectangular causal alignment
- forward occupancy limited to 1 CTA/SM (4 warps) for `D_PAD` >= 80 on ~100 KB parts
- performance behind PyTorch SDPA (forward ~0.5-0.6x, backward ~0.2-0.4x here)
- the causal backward benchmark sweep is incomplete

## Upcoming work

### Dropout

Forward probability dropout, matching backward mask regeneration, seed and offset handling, deterministic testing, causal and unmasked coverage.

**Not started.**

### Kernel architecture and speed

See the ranked roadmap above. The tracking metric is backward TFLOPs at D=128, N=4096 (currently 3.6 vs SDPA 15.9 on the test GPU), plus forward warps/SM at D=128 (currently 4).

## Profiling

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

Counters that decide the roadmap order:

- `sm__pipe_tensor_cycles_active` (tensor utilization)
- `launch__occupancy_*` (warps/SM from the shared-memory cap)
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared` (should be near zero after the stride rework)
- warp-stall reasons (`long_scoreboard`, `barrier`, `short_scoreboard`)

The build keeps `-lineinfo`, so Nsight Compute can map SASS back to source lines.

## Repo hygiene

Never commit:

```
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