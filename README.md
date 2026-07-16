# FlashAttention CUDA Scratch

A from-scratch CUDA implementation of FlashAttention forward and backward.

This repo is for learning and experimenting with CUDA kernels, MMA, shared memory, PyTorch extensions, online softmax, and attention backward math.

It is not a production FlashAttention replacement. The current forward pass is competitive with PyTorch SDPA on one tested long-sequence shape, but backward is still slower.

---

## Supported Shapes

Current support:

```text
dtype: fp16
B: runtime dynamic
H: runtime dynamic
N: runtime dynamic
D: 32, 64, 128, 256
attention: non-causal
dropout: no
```

`N` is runtime dynamic.

`D` is handled through compiled specializations:

```text
D = 32
D = 64
D = 128
D = 256
```

So the implementation supports common head dimensions, but not arbitrary head dimensions yet.

Tested mainly on:

```text
GPU: NVIDIA RTX 3050 6GB Laptop GPU
CUDA arch: sm_86
CUDA: 11.8
PyTorch: 2.7.1+cu118
```

---

## Current Status

```text
forward: working
backward: working
dtype: fp16
attention type: non-causal
```

Backward is split into three kernels:

```text
1. Delta kernel
2. DK/DV kernel
3. DQ kernel
```

Delta is:

```text
Delta = sum(O * dO, dim=-1)
```

The split keeps ownership simple:

```text
DK/DV kernel owns KV tiles
DQ kernel owns Q tiles
```

No atomic adds are used in the current backward path.

---

## Latest Benchmark Shape

Latest benchmark shape:

```text
B = 1
H = 8
N = 4096
D = 64
dtype = fp16
attention = non-causal
```

Results will change for other shapes, GPUs, CUDA versions, and PyTorch SDPA backend choices.

---

## Latest Forward Benchmark

Compared against PyTorch SDPA.

Run 1:

```text
custom:     2.8547 ms
torch sdpa: 3.0663 ms
ratio custom/torch: 0.931x
```

Run 2:

```text
custom:     2.7367 ms
torch sdpa: 3.5133 ms
ratio custom/torch: 0.779x
```

Run 3:

```text
custom:     2.8311 ms
torch sdpa: 3.0599 ms
ratio custom/torch: 0.925x
```

For this tested shape, the custom forward is faster than PyTorch SDPA in these runs.

Approximate range:

```text
custom forward: 2.74 ms - 2.85 ms
torch SDPA:     3.06 ms - 3.51 ms
ratio:          0.78x - 0.93x custom/torch
```

A ratio below `1.0x` means the custom kernel is faster for that run.

---

## Latest Backward Benchmark

Compared against PyTorch SDPA backward.

Run 1:

```text
custom backward: 2.2210 ms
torch backward:  0.8881 ms
ratio custom/torch: 2.50x
```

Run 2:

```text
custom backward: 2.2279 ms
torch backward:  0.8934 ms
ratio custom/torch: 2.49x
```

The backward pass is still slower than PyTorch SDPA backward.

Current backward status:

```text
custom backward: about 2.22 ms
torch backward:  about 0.89 ms
ratio:           about 2.5x slower
```

---

## Correctness

Forward is compared against PyTorch scaled dot product attention.

Backward is compared against PyTorch SDPA backward.

Typical backward correctness result:

```text
DQ max err: 0.00048828125
DK max err: 0.00048828125
DV max err: 0.000244140625

bad > 0.01: 0 for DQ, DK, DV
NaN: False for DQ, DK, DV
```

The current implementation passes the tested fp16 tolerances.

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

Because the extension is built locally, run scripts with:

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
  max err: around 0.0005
  bad > 0.01: 0

DK
  has nan: False
  max err: around 0.0005
  bad > 0.01: 0

DV
  has nan: False
  max err: around 0.0003
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

Latest forward benchmark on RTX 3050 Laptop GPU:

```text
B=1, H=8, N=4096, D=64

custom forward: 2.74 ms - 2.85 ms
torch SDPA:     3.06 ms - 3.51 ms
ratio:          0.78x - 0.93x custom/torch
```

Latest backward benchmark on RTX 3050 Laptop GPU:

```text
custom backward: about 2.22 ms
torch backward:  about 0.89 ms
ratio:           about 2.5x custom/torch
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

## Current Limitations

```text
- fp16 only
- D supports only 32, 64, 128, 256
- non-causal only
- no dropout
- no variable-length packed sequences
- no bf16 path
- no GQA/MQA support
- not packaged as a general library
- backward is still slower than PyTorch SDPA
```

The current code is a scratch implementation for learning, profiling, and optimization work.

---

## Optimization Notes

Current forward is already competitive on the tested long-sequence shape:

```text
B=1, H=8, N=4096, D=64
```

The main remaining target is backward.

Likely backward bottlenecks:

```text
extra kernel launches
Delta/DKDV/DQ split overhead
shared memory usage
register pressure
recomputing attention probabilities
MMA utilization
global memory movement
```

The current backward avoids atomic adds by splitting ownership, but this also means multiple kernels and more scheduling overhead.

---

## Next Steps

Possible improvements:

```text
- add causal masking
- benchmark more N/D configs
- profile backward kernels separately
- reduce backward kernel launch overhead
- improve occupancy
- reduce shared memory usage
- improve register pressure
- tune D=64 path more aggressively
- tune D=128 and D=256 separately
- add cleaner benchmark summary
- add PyTorch SDPA backend notes
- clean up API
```

Longer-term possible work:

```text
- bf16 support
- dropout
- GQA/MQA
- variable-length packed sequences
- sliding-window attention
- better autograd wrapper
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