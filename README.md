# FlashAttention CUDA Scratch

A from-scratch CUDA implementation of FlashAttention forward and backward.

This repo is mainly for learning CUDA kernels, MMA, shared memory, PyTorch extensions, and attention backward math. It is not a production FlashAttention replacement and it is not faster than PyTorch SDPA yet.

## Supported Shapes

Current support:

```text
dtype: fp16
B: dynamic
H: dynamic
N: dynamic
D: 32, 64, 128, 256
attention: non-causal
dropout: no
```

`N` is runtime dynamic.

`D` is not fully runtime dynamic. It is handled by dispatching to compiled kernel specializations:

```text
D = 32
D = 64
D = 128
D = 256
```

So this supports many practical shapes, but not arbitrary head dimensions yet.

Tested mainly on:

```text
GPU: RTX 3050 Laptop GPU
CUDA arch: sm_86
CUDA: 11.8
```

---

## Status

Forward pass works.

Backward pass works.

Backward is split into 3 kernels:

```text
1. Delta kernel
2. DK/DV kernel
3. DQ kernel
```

Delta is:

```text
Delta = sum(O * dO, dim=-1)
```

The split keeps output ownership simple:

```text
DK/DV kernel owns KV tiles
DQ kernel owns Q tiles
```

No atomic adds are used.

---

## Benchmark Shape

The benchmark numbers below are from this shape:

```text
B = 1
H = 8
N = 512
D = 128
dtype = fp16
```

Results will change for other shapes and GPUs.

---

## Results

### Forward Correctness

Compared against PyTorch scaled dot product attention.

```text
O max err: ~0.00024
L max err: ~0.000001
bad count > 0.05: 0
```

### Forward Benchmark

```text
custom forward: ~0.6487 ms
torch sdpa:      ~0.0865 ms
ratio:           ~7.5x slower
```

### Backward Correctness

Compared against PyTorch SDPA backward.

```text
DQ max err: 0.00048828125
DK max err: 0.00048828125
DV max err: 0.000244140625

bad > 0.01: 0 for DQ, DK, DV
NaN: False for DQ, DK, DV
```

### Backward Benchmark

```text
custom backward: 1.9782 ms
torch backward:  0.3689 ms
ratio custom/torch: 5.36x
```

The custom implementation is slower than PyTorch right now. The goal of this repo is correctness, kernel structure, profiling, and learning.

---

## Project Structure

```text
.
├── benchmarks/
│   ├── bench_fwd.py
│   ├── bench_bwd.py
│   ├── correctness_fwd.py
│   ├── correctness_bwd.py
│   └── profile_custom_fwd.py
│
├── src/
│   ├── bindings.cpp
│   ├── flash_api.cu
│   ├── flash_Acc_reg.cuh
│   ├── flash_attn_v1.cuh
│   └── helper.cuh
│
├── setup.py
└── README.md
```

Main files:

```text
src/flash_Acc_reg.cuh   CUDA kernels
src/helper.cuh          MMA/helper functions
src/flash_api.cu        PyTorch extension launch code
src/bindings.cpp        Python bindings
```

---

## Build

From the repo root:

```bash
rm -rf build flash_acc_reg_ext*.so
MAX_JOBS=4 TORCH_CUDA_ARCH_LIST="8.6" python setup.py build_ext --inplace
```

This creates a local `.so` extension file in the repo root.

Because the extension is built locally, run scripts with `PYTHONPATH=.`:

```bash
PYTHONPATH=. python benchmarks/correctness_fwd.py
```

---

## Correctness Tests

Forward:

```bash
PYTHONPATH=. python benchmarks/correctness_fwd.py
```

Backward:

```bash
PYTHONPATH=. python benchmarks/correctness_bwd.py
```

Expected backward output should look roughly like:

```text
DQ
  has nan: False
  max err: 0.00048828125
  bad > 0.01: 0

DK
  has nan: False
  max err: 0.00048828125
  bad > 0.01: 0

DV
  has nan: False
  max err: 0.000244140625
  bad > 0.01: 0
```

---

## Benchmarks

Forward benchmark:

```bash
PYTHONPATH=. python benchmarks/bench_fwd.py
```

Backward benchmark:

```bash
PYTHONPATH=. python benchmarks/bench_bwd.py
```

Current backward benchmark on RTX 3050 Laptop GPU:

```text
custom backward: 1.9782 ms
torch backward:  0.3689 ms
ratio custom/torch: 5.36x
```

---

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
  python benchmarks/profile_custom_fwd.py
```

Profiling reports are ignored by git.

---

## Implementation Notes

Forward:

```text
- tiled Q/K/V loading
- shared memory staging
- MMA-based score computation
- online softmax update
- logsumexp L saved for backward
- fp16 probability storage for the P @ V step
```

Backward:

```text
- separate Delta kernel
- separate DK/DV kernel
- separate DQ kernel
- DK/DV kernel is KV-tile owned
- DQ kernel is Q-tile owned
- no atomic adds
```

Backward math:

```text
P     = exp(QK^T / sqrt(D) - L)
Delta = sum(O * dO, dim=-1)

dV = P^T @ dO
dP = dO @ V^T
dS = P * (dP - Delta) / sqrt(D)

dQ = dS @ K
dK = dS^T @ Q
```

---

## Limitations

Current limitations:

```text
- fp16 only
- D supports only 32, 64, 128, 256
- non-causal only
- no dropout
- no variable-length packed sequences
- not faster than PyTorch SDPA
- not packaged as a general library
```

The current code is a scratch implementation for learning and profiling, not a production-ready attention library.

---

## Next Steps

Possible improvements:

```text
- add causal masking
- add more D specializations
- benchmark more N/D configs
- profile backward kernels
- reduce shared memory usage
- reduce kernel launch overhead
- improve occupancy
- clean up API
```

---

## Git Notes

Generated files should not be committed:

```text
build/
*.so
*.o
*.ncu-rep
*.nsys-rep
__pycache__/
```

Only source files, benchmarks, setup file, and README should be tracked.