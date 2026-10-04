
````
# FlashAttention CUDA Scratch

A from-scratch CUDA implementation of FlashAttention forward and backward: tensor-core MMA (`m16n8k16`), `cp.async` tiled loads, double-buffered shared memory, online softmax in registers, and a manually launched three-kernel backward. Built for learning and profiling — not a production replacement for PyTorch SDPA or FlashAttention.

Measured position vs SDPA (RTX 3050 6GB Laptop, sm_86): forward ~0.5–0.6×, backward ~0.2–0.4×. SDPA operates at the practical tensor-core ceiling of this GPU; the gap is quantified and explained in [Why it is slower than SDPA](#why-it-is-slower-than-sdpa).

## Supported inputs

| Property | Support |
|---|---|
| Device | CUDA, sm_80+ required (`cp.async`, `m16n8k16`) |
| Input dtype | FP16 (FP32 accumulation in softmax and MMA) |
| B, H, Sq, Skv | Runtime dynamic; cross-attention (`Sq ≠ Skv`) supported |
| Masking | Unmasked; causal (top-left aligned, key <= query) |
| GQA / MQA | Yes — K/V may carry H_kv heads with `H % H_kv == 0` (`H_kv = H`: MHA, `H_kv = 1`: MQA) |
| Head dim D | Any integer 1..256 (no divisibility requirement); rounded up to `D_PAD ∈ {32, 64, 80, 96, 128, 160, 192, 224, 256}` |
| Dropout | No |
| Variable-length packed sequences | No |
| Sliding window / bottom-right causal | No |

## Python API

```python
O, L = flash_acc_reg_ext.flash_fwd(Q, K, V, causal)
dQ, dK, dV = flash_acc_reg_ext.flash_bwd(Q, K, V, O, dO, L, causal)
````

 Shapes:

```
Q, O, dO, dQ: [B, H,    Sq,  D]
K, V:         [B, H_kv, Skv, D]
dK, dV:       [B, H_kv, Skv, D]   (returned with H_kv heads)
L:            [B, H,    Sq]
```

 H\_kv must divide H; violations are rejected with explicit errors. L is the per-row log-sum-exp (`m + log(l)`, FP32) consumed by the backward.

 Pass `causal=True/False` explicitly in tests and benchmarks.

 ## Current status

 - forward: working (tested sweep below)
- backward: working (tested sweep below)
- causal: working (top-left mask)
- cross-attention: working for `Sq < Skv` and `Sq > Skv`
- dtype: FP16 only

 Backward uses three kernels:

 1. Delta kernel — `Delta = rowsum(O * dO)`
2. dK/dV kernel — one block per (batch, head, KV tile)
3. dQ kernel — one block per (batch, head, Q tile)

 The split gives every output tile a single owning block: no atomics, plain stores, and causally-dead KV tiles store zeros themselves — dK/dV/dQ require no pre-zeroing.

 ## Correctness

 The combined test compares O, L, dQ, dK, dV against a float32 PyTorch reference. Latest sweep: 9 shapes × 18 modes (self- and cross-attention, causal and unmasked) across all nine D\_PAD values.

 **Summary: 18/18 modes passed**

 | Tensor | Max abs error |
| --- | --- |
| O | 0.0012186 |
| L | 0.0000014 |
| dQ | 0.0014327 |
| dK | 0.0014586 |
| dV | 0.0017262 |

These validate the tested inputs and tolerances; they are not a proof for every shape or distribution.

 Re-run the sweep after any kernel change:

```
python correctness_fwd_bwd_multi.py
```

 The any-D loader, padded strides, and exclusive-store rework are covered by the same harness.

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

```
python -m benchmarks.bench_fwd
python -m benchmarks.bench_bwd
```

 Metric: `speedup = t_SDPA / t_extension` (\> 1 means the extension is faster). SDPA's forward+backward comparison includes its autograd graph; the extension's includes all four kernels.

 ## Results

 ### Forward speedup (SDPA / extension)

 | D \\ N | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.64 | 0.45 | 0.51 | 0.54 | 0.57 | 0.59 |
| 64 | 0.81 | 0.52 | 0.49 | 0.53 | 0.55 | 0.57 |
| 96 | 0.64 | 0.43 | 0.48 | 0.51 | 0.52 | 0.53 |
| 128 | 0.65 | 0.46 | 0.50 | 0.54 | 0.56 | 0.57 |
| 192 | 0.64 | 0.51 | 0.53 | 0.54 | 0.55 | 0.55 |
| 256 | 0.65 | 0.55 | 0.56 | 0.56 | 0.56 | 0.57 |

### Forward speedup, causal

 | D \\ N | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.76 | 0.57 | 0.55 | 0.56 | 0.58 | 0.60 |
| 64 | 0.89 | 0.75 | 0.61 | 0.56 | 0.57 | 0.59 |
| 96 | 0.90 | 0.70 | 0.48 | 0.52 | 0.54 | 0.57 |
| 128 | 0.95 | 0.87 | 0.66 | 0.54 | 0.58 | 0.61 |
| 192 | 0.75 | 0.68 | 0.61 | 0.57 | 0.57 | 0.57 |
| 256 | 0.93 | 0.84 | 0.70 | 0.65 | 0.64 | 0.64 |

### Backward speedup (SDPA / extension), unmasked

 | D \\ N | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.82 | 0.52 | 0.37 | 0.31 | 0.28 | 0.26 |
| 64 | 0.67 | 0.44 | 0.34 | 0.29 | 0.27 | 0.25 |
| 96 | 0.78 | 0.50 | 0.40 | 0.34 | 0.32 | 0.30 |
| 128 | 0.58 | 0.37 | 0.30 | 0.25 | 0.23 | 0.22 |
| 192 | 0.68 | 0.45 | 0.35 | 0.29 | 0.26 | 0.25 |
| 256 | 0.49 | 0.32 | 0.25 | 0.26 | 0.32 | 0.17 |

### Backward speedup, causal (run interrupted; partial)

 | D \\ N | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 1.09 | 0.70 | 0.50 | 0.38 | 0.32 | 0.29 |
| 64 | 0.91 | — | — | — | — | — |

### TFLOPs at N=4096 (compute-bound end of the sweep, unmasked)

 | D | ext fwd | SDPA fwd | ext bwd | SDPA bwd |
| --- | --- | --- | --- | --- |
| 32 | 10.0 | 17.0 | 4.1 | 15.5 |
| 64 | 10.1 | 17.8 | 4.2 | 16.5 |
| 96 | 9.7 | 18.3 | 4.9 | 16.2 |
| 128 | 10.2 | 17.8 | 3.6 | 15.9 |
| 192 | 9.9 | 18.0 | 3.2 | 12.7 |
| 256 | 9.4 | 16.6 | 2.4 | 13.8 |

### Reading of the data

 - Forward plateaus at \~10 TFLOPs regardless of D; SDPA plateaus at 16.5–18.4.
- Backward plateaus at 2.4–4.9 TFLOPs; SDPA at 13–16.5 for D \<= 128.
- At N \<= 512 both sides are launch/overhead-floored; those cells measure overhead floors, not kernels.
- The single \> 1x cell (backward causal, D=32, N=128) is that floor, not a kernel advantage.
- SDPA's backward drops at D = 192/256 (different backend route for large head dims) but still leads by 2–6×.

 ## Why it is slower than SDPA

 ### Hardware frame

 On GA10x, dense FP16 tensor ops with FP32 accumulate run at 2× the FP32 rate; at this part's boost clocks that is ≈18 TFLOPs.

 SDPA's best cells sit essentially at that ceiling, so the reference here is hardware-bound and the entire gap is implementation headroom.

 Consumer Ampere also caps dynamic shared memory at \~100 KB per SM (vs 164 KB on A100) — that cap is the binding constraint throughout.

 ### Forward: \~0.5–0.6×, plateau ≈10 TFLOPs

 Shared-memory capacity → occupancy.

 One CTA is 128 threads (4 warps) holding Q + 2×K + 2×V tiles: \~25 KB/CTA at D\_PAD=32, \~45 KB at 64, 56–87 KB at 80–256.

 Against the \~100 KB/SM cap that yields 12 / 8 / 4 resident warps per SM.

 Four warps cannot overlap ldmatrix, mma, and cp.async latencies; the tensor pipe stalls on shared-memory dependencies.

 This is the dominant forward gap and the reason the ratio is flat \~0.5–0.6× for every D \>= 80.

 Two-stage K/V pipeline with wait\_group 0 drains the pipeline each iteration; deeper staging needs shared memory the budget does not have at large D.

 Overhead floor at N \<= 512 (launches, binding, Python) caps both implementations.

 ### Backward: \~0.2–0.4×, plateau ≈2.4–4.9 TFLOPs

 Structural, not tuning:

 P and dS cross shared memory between every pair of GEMMs.

 MMA accumulator layout (C fragment) is not a valid A-operand layout, so each step does:

```
registers → smem → softmax/dS math → smem → ldmatrix → mma
```

 Four extra full-tile shared-memory round trips and three `__syncthreads()` per (Q, K) tile pair in each kernel.

 FA2-class kernels keep P/dS register-resident with per-warp row ownership and shuffle-based layout conversion.

 This is the largest single backward cost.

 The \~100 KB cap forces tiny tiles — Br=16–32, Bc=16–32 at D\_PAD \>= 128 with 2×Q + 2×dO staged for the pipeline.

 Per-step fixed costs (syncs, L/Delta staging, the four smem passes) amortize against Br × Bc × D mma FLOPs; at Br=16 that ratio is poor.

 (After the stride/bank rework, Br=Bc=64 at D=128 needs \~130 KB — fits an A100, not GA10x.)

 Two-kernel split re-streams operands.

 The dQ kernel reads all of K, V per Q tile; the dK/dV kernel reads all of Q, dO per KV tile — roughly 2× the HBM traffic of a fused backward.

 A deliberate trade to avoid atomics; SDPA's fused kernel reads each operand \~once.

 \~8 warps/SM (2 CTAs × 4 warps) — the forward's latency-hiding problem, compounded by more barriers per step.

 ### Not the cause

 - FP16 P/dS storage (SDPA does the same)
- MMA shape (same m16n8k16)
- Accumulation precision (FP32 in both)
- Python binding at large N (noise)
- The loader (cp.async with hardware zero-fill; the scalar path only serves ragged D % 8 != 0 tails)

 ### Already fixed and reflected in these numbers

 - p/dS shared-memory bank conflicts (padded strides — the old layout serialized whole warps on single banks)
- FP16 atomics on dK/dV/dQ (replaced by exclusive-ownership stores)
- dK/dV zero-init memsets
- predicated cp.async fills
- unrolling
- **launch\_bounds**

 ## Roadmap (ranked by expected payoff on this GPU)

 1. Register-resident P/dS backward — per-warp row ownership, shuffle conversions, no smem round trips. Targets the 3–6× backward gap directly.
2. Occupancy-first tiling for the \~100 KB budget — shapes that keep ≥ 2 CTAs/SM at every D (smaller staging, split-D across warps) rather than 1 CTA × 4 warps.
3. Fused backward (dQ folded into the dK/dV kernel via FP32 workspace + reduction) to halve operand re-streaming.
4. Deeper cp.async pipeline where it fits (D\_PAD \<= 64 forward: 3–4 stages, wait\_group 1).
5. Small-N floor — CUDA Graph capture of the 3–4 launches; exp2f with folded scale; \_\_half2 elementwise passes.

 Any speed claim should be based on repeated measurements with the same shape, mask, device, PyTorch build, and SDPA backend.

 ## Build

```
flash_attn_cuda.cu   (includes the kernel header)
binding.cpp
setup.py
```

```
python setup.py build_ext --inplace
```

 Clean rebuild:

```
python setup.py clean --all
rm -rf build
find . -maxdepth 1 -type f -name 'flash_acc_reg_ext*.so' -delete
python setup.py build_ext --inplace
```

 The supplied setup defaults to sm\_86 (RTX 3050 test system). Set TORCH\_CUDA\_ARCH\_LIST for another GPU.

 Kernels require sm\_80+; on older architectures they exit without error and outputs are left untouched.

 ## Implementation notes

 ### Forward

 - tiled Q/K/V loads via cp.async (hardware zero-fill of out-of-range tiles)
- double-buffered K/V shared memory, 2-stage pipeline
- tensor-core m16n8k16 score computation (FP32 accumulators)
- online softmax in registers (per-warp max/sum, quad reduction)
- tensor-core P @ V
- logsumexp written for backward

 ### Backward

```
P   = exp(S * scale - L)
S   = Q K^T
dV  = P^T @ dO
dP  = dO V^T
dS  = P * (dP - Delta) * scaled
dQ  = dS @ K               (Q-tile-owned kernel)
dK  = dS^T @ Q              (KV-tile-owned kernel)
```

 ### Layout details that matter

 - shared-memory strides padded (+8) to avoid bank conflicts, including the p/dS tile
  - stored \[k\]\[q\] in dK/dV
  - stored \[q\]\[k\] in dQ
- exclusive tile ownership -\> plain stores, no atomics, no pre-zeroing
- GQA mapping: `kv_head = head / (H / H_kv)`
- host-side constexpr smem calculators mirror the kernel layouts and opt in via `cudaFuncSetAttribute` (\>48 KB dynamic smem requires it)

 ## Current limitations

 - FP16 only
- CUDA only (sm\_80+)
- top-left causal mask only
- no dropout
- no BF16
- no varlen
- no sliding window
- no bottom-right rectangular causal alignment
- manual forward/backward API, not an autograd module
- forward occupancy limited to 1 CTA/SM (4 warps) for D\_PAD \>= 80 on \~100 KB parts
- performance behind PyTorch SDPA (forward \~0.5–0.6×, backward \~0.2–0.4× here)

 ## Upcoming work

 ### Dropout

 Forward probability dropout, matching backward mask regeneration, seed and offset handling, deterministic testing, causal and unmasked coverage.

 **Not started.**

 ## Kernel architecture and speed

 See the ranked roadmap above; the tracking metric is backward TFLOPs at D=128, N=4096 (currently 3.6 vs SDPA 15.9 on the test GPU), plus forward warps/SM at D=128 (currently 4).

 ## Profiling

```
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

 ## Repo hygiene

 Never commit:

```
build/
*.so
*.o
*.ncu-rep
*.nsys-rep
__pycache__/


```