# FlashAttention CUDA Scratch

A from-scratch CUDA implementation of FlashAttention forward and backward for learning, profiling, and kernel experimentation.

This project uses Tensor Core MMA instructions, asynchronous shared-memory loads, online softmax, and manually launched backward kernels. It is not a production replacement for PyTorch SDPA or FlashAttention. On the benchmark suite below, PyTorch SDPA is faster in nearly every case, especially for backward.

## Supported inputs

| Property | Current support |
|---|---|
| Device | CUDA |
| Input dtype | FP16 |
| Batch size `B` | Runtime dynamic |
| Head count `H` | Runtime dynamic |
| Query length `Sq` | Runtime dynamic |
| Key/value length `Skv` | Runtime dynamic |
| Self-attention | Yes |
| Cross-attention | Yes, with independent `Sq` and `Skv` |
| Unmasked attention | Yes |
| Causal attention | Yes, top-left causal |
| Dropout | No |

Runtime head dimension `D` must be divisible by 8 and no larger than 256. It is rounded up to one of these compiled padded dimensions:

```text
D_PAD = 32, 64, 80, 96, 128, 160, 192, 224, 256
```

The correctness sweep has tested each of those nine dimensions directly.

For rectangular causal attention, the implemented mask is top-left aligned:

```text
key_index <= query_index
```

It is not bottom-right aligned decoding attention.

## Python API

```python
O, L = flash_acc_reg_ext.flash_fwd(Q, K, V, causal)

dQ, dK, dV = flash_acc_reg_ext.flash_bwd(
    Q,
    K,
    V,
    O,
    dO,
    L,
    causal,
)
```

Tensor shapes:

```text
Q, O, dO, dQ: [B, H, Sq,  D]
K, V, dK, dV: [B, H, Skv, D]
L:             [B, H, Sq]
```

Pass `causal=False` or `causal=True` explicitly in tests and benchmarks so results do not depend on the binding default.

## Current status

```text
forward:  working for the tested configurations
backward: working for the tested configurations
causal:   working with the top-left mask
cross-attention: working for Sq < Skv and Sq > Skv
dtype: FP16 only
```

Backward uses three kernels:

```text
1. Delta kernel
2. dK/dV kernel owned by KV tiles
3. dQ kernel owned by Q tiles
```

The split avoids atomic additions. Delta is computed as:

```text
Delta = sum(O * dO, dim=-1)
```

## Correctness

The combined test compares `O`, `L`, `dQ`, `dK`, and `dV` against a float32 PyTorch reference.

The latest sweep covered:

```text
9 shapes
18 modes
self-attention and cross-attention
unmasked and causal attention
D = 32, 64, 80, 96, 128, 160, 192, 224, 256
```

Result:

```text
summary: 18/18 modes passed
```

Largest errors observed in that run:

| Tensor | Maximum absolute error |
|---|---:|
| `O` | 0.0012186 |
| `L` | 0.0000014 |
| `dQ` | 0.0014327 |
| `dK` | 0.0014586 |
| `dV` | 0.0017262 |

These results validate the tested random inputs and tolerances; they are not a proof for every possible shape or input distribution.

Run the full correctness sweep with:

```bash
python correctness_fwd_bwd_multi.py
```

## Benchmark environment

```text
GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU
CUDA architecture: sm_86
CUDA: 11.8
PyTorch: 2.7.1+cu118
dtype: FP16
warmup iterations: 2
timed iterations: 5
SDPA backend: automatic PyTorch selection
```

The custom forward call produces both `O` and `L`; the SDPA forward call produces only `O`. The forward-plus-backward comparison includes the custom forward, all three custom backward kernels, the SDPA forward, and SDPA autograd backward.

The ratio in the tables is:

```text
custom latency / SDPA latency
```

A ratio below `1.0x` means custom has lower latency. A ratio above `1.0x` means SDPA has lower latency.

## Forward benchmark

All times are milliseconds.

| Workload | Mode | `Sq` | `Skv` | `D` | Custom | SDPA | Ratio | Result |
|---|---|---:|---:|---:|---:|---:|---:|---|
| self 1024 | Unmasked | 1024 | 1024 | 64 | 0.2150 | 0.1784 | 1.205x | Custom 20.5% slower |
| self 1024 | Causal | 1024 | 1024 | 64 | 0.1396 | 0.1303 | 1.072x | Custom 7.2% slower |
| self 2048 | Unmasked | 2048 | 2048 | 64 | 0.7342 | 0.5915 | 1.241x | Custom 24.1% slower |
| self 2048 | Causal | 2048 | 2048 | 64 | 0.4582 | 0.3604 | 1.271x | Custom 27.1% slower |
| self 4096 | Unmasked | 4096 | 4096 | 64 | 2.8289 | 1.9957 | 1.417x | Custom 41.7% slower |
| self 4096 | Causal | 4096 | 4096 | 64 | 1.7297 | 1.1327 | 1.527x | Custom 52.7% slower |
| cross short Q | Unmasked | 512 | 2048 | 64 | 0.2134 | 0.1630 | 1.309x | Custom 30.9% slower |
| cross short Q | Causal | 512 | 2048 | 64 | 0.0444 | 0.0475 | 0.935x | Custom 6.5% lower latency |
| cross long Q | Unmasked | 2048 | 512 | 64 | 0.1879 | 0.1427 | 1.317x | Custom 31.7% slower |
| cross long Q | Causal | 2048 | 512 | 64 | 0.1726 | 0.1437 | 1.201x | Custom 20.1% slower |
| self 1024 | Unmasked | 1024 | 1024 | 128 | 0.4544 | 0.3137 | 1.448x | Custom 44.8% slower |
| self 1024 | Causal | 1024 | 1024 | 128 | 0.2703 | 0.1791 | 1.509x | Custom 50.9% slower |
| self 512 | Unmasked | 512 | 512 | 256 | 0.2877 | 0.1794 | 1.603x | Custom 60.3% slower |
| self 512 | Causal | 512 | 512 | 256 | 0.1774 | 0.1282 | 1.384x | Custom 38.4% slower |

Custom forward has lower latency in one of the 14 measured cases: causal cross-attention with `Sq=512`, `Skv=2048`, and `D=64`. SDPA has lower latency in the other 13 cases.

## Forward and backward benchmark

All times are milliseconds and include both forward and backward work.

| Workload | Mode | `Sq` | `Skv` | `D` | Custom | SDPA | Ratio | Result |
|---|---|---:|---:|---:|---:|---:|---:|---|
| self 1024 | Unmasked | 1024 | 1024 | 64 | 1.9783 | 0.6971 | 2.838x | Custom 183.8% slower |
| self 1024 | Causal | 1024 | 1024 | 64 | 1.0594 | 0.4622 | 2.292x | Custom 129.2% slower |
| self 2048 | Unmasked | 2048 | 2048 | 64 | 7.1913 | 2.2091 | 3.255x | Custom 225.5% slower |
| self 2048 | Causal | 2048 | 2048 | 64 | 3.8406 | 1.3007 | 2.953x | Custom 195.3% slower |
| self 4096 | Unmasked | 4096 | 4096 | 64 | 25.9143 | 7.1088 | 3.645x | Custom 264.5% slower |
| self 4096 | Causal | 4096 | 4096 | 64 | 13.7843 | 3.9665 | 3.475x | Custom 247.5% slower |
| cross short Q | Unmasked | 512 | 2048 | 64 | 1.8407 | 0.5564 | 3.308x | Custom 230.8% slower |
| cross short Q | Causal | 512 | 2048 | 64 | 0.3198 | 0.2646 | 1.209x | Custom 20.9% slower |
| cross long Q | Unmasked | 2048 | 512 | 64 | 1.8354 | 0.6801 | 2.699x | Custom 169.9% slower |
| cross long Q | Causal | 2048 | 512 | 64 | 1.6421 | 0.9384 | 1.750x | Custom 75.0% slower |
| self 1024 | Unmasked | 1024 | 1024 | 128 | 2.7505 | 1.1444 | 2.403x | Custom 140.3% slower |
| self 1024 | Causal | 1024 | 1024 | 128 | 1.6046 | 0.7119 | 2.254x | Custom 125.4% slower |
| self 512 | Unmasked | 512 | 512 | 256 | 2.5825 | 0.7773 | 3.323x | Custom 232.3% slower |
| self 512 | Causal | 512 | 512 | 256 | 1.4334 | 0.5994 | 2.391x | Custom 139.1% slower |

SDPA has lower forward-plus-backward latency in all 14 measured cases. The gap ranges from `1.209x` to `3.645x`.

## Higher head-dimension comparison

This table isolates the higher-dimensional cases present in the benchmark run. The `D=128` and `D=256` rows use different sequence lengths, so they should not be interpreted as a controlled head-dimension scaling experiment.

| Shape | Mode | Forward ratio | Forward result | Forward + backward ratio | Forward + backward result |
|---|---|---:|---|---:|---|
| `Sq=Skv=1024, D=128` | Unmasked | 1.448x | Custom slower | 2.403x | Custom slower |
| `Sq=Skv=1024, D=128` | Causal | 1.509x | Custom slower | 2.254x | Custom slower |
| `Sq=Skv=512, D=256` | Unmasked | 1.603x | Custom slower | 3.323x | Custom slower |
| `Sq=Skv=512, D=256` | Causal | 1.384x | Custom slower | 2.391x | Custom slower |

Run the benchmark with:

```bash
python benchmark_fwd_bwd_sdpa.py --warmup 5 --iters 20
```

Useful options:

```bash
python benchmark_fwd_bwd_sdpa.py --quick
python benchmark_fwd_bwd_sdpa.py --mode unmasked
python benchmark_fwd_bwd_sdpa.py --mode causal
python benchmark_fwd_bwd_sdpa.py --forward-only
```

## Build

The extension directory contains:

```text
flash_attn_cuda.cu
binding.cpp
setup.py
```

Build from that directory:

```bash
python setup.py build_ext --inplace
```

Clean and rebuild:

```bash
python setup.py clean --all
rm -rf build
find . -maxdepth 1 -type f -name 'flash_acc_reg_ext*.so' -delete
python setup.py build_ext --inplace
```

The supplied setup defaults to CUDA architecture `8.6`, matching the RTX 3050 test system. Set `TORCH_CUDA_ARCH_LIST` appropriately before building for another GPU.

## Implementation notes

Forward currently uses:

```text
tiled Q/K/V loading
double-buffered K/V shared memory
cp.async loads
Tensor Core MMA score computation
online softmax in registers
Tensor Core MMA for P @ V
logsumexp output for backward
```

Backward computes:

```text
P     = exp(QK^T / sqrt(D) - L)
Delta = sum(O * dO, dim=-1)
dV    = P^T @ dO
dP    = dO @ V^T
dS    = P * (dP - Delta) / sqrt(D)
dQ    = dS @ K
dK    = dS^T @ Q
```

The backward split simplifies output ownership and avoids atomics, but it also adds launch overhead and repeats some memory movement.

## Current limitations

```text
FP16 only
CUDA only
top-left causal mask only
no dropout
no BF16 path
no GQA or MQA
no variable-length packed sequences
no sliding-window attention
no bottom-right rectangular causal alignment
manual forward/backward API instead of a complete autograd wrapper
current setup targets sm_86 by default
performance is generally behind PyTorch SDPA
```

## Upcoming work

### Dropout

Planned dropout work includes:

```text
forward probability dropout
matching backward mask regeneration
seed and offset handling
deterministic testing
causal and unmasked coverage
```

Dropout is not implemented yet.

### Kernel architecture and speed

The primary optimization target is backward. Planned investigations include:

```text
reduce or fuse backward kernel launches
reduce shared-memory traffic
reduce register pressure and spills
improve cp.async overlap
tune Br and Bc by D_PAD and sequence shape
specialize self-attention and cross-attention launch choices
specialize causal tile pruning
improve Tensor Core utilization
measure occupancy and memory throughput per kernel
add architecture-specific tuning for sm_86 and newer GPUs
```

Any speed claim should be based on repeated measurements with the same shape, mask, device, PyTorch build, and SDPA backend.

## Profiling

Example Nsight Compute command:

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
  python benchmark_fwd_bwd_sdpa.py --forward-only
```

## Git notes

Generated files should not be committed:

```text
build/
*.so
*.o
*.ncu-rep
*.nsys-rep
__pycache__/
```

Track the CUDA source, binding, setup script, correctness tests, benchmarks, and this README.