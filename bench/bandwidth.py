#!/usr/bin/env python3
"""Measured device bandwidth and BF16 / FP8 GEMM throughput, run inside an engine image:
  docker run --rm --gpus all -v $PWD/bench:/b --entrypoint python3 IMAGE /b/bandwidth.py
"""
import torch

torch.backends.cuda.matmul.allow_tf32 = False
dev = torch.device("cuda")
print(torch.cuda.get_device_name(), "sm", torch.cuda.get_device_capability(), "torch", torch.__version__)


def timed(fn, reps):
    fn(); torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    best = 1e9
    for _ in range(5):
        start.record()
        for _ in range(reps):
            fn()
        end.record(); torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / 1e3 / reps)
    return best


n = 2 * 2**30  # 2 GiB per buffer
a = torch.empty(n, dtype=torch.uint8, device=dev); b = torch.empty_like(a)
t = timed(lambda: b.copy_(a), 10)
print(f"copy bandwidth (read + write): {2 * n / t / 1e9:.0f} GB/s")
x = torch.empty(n // 2, dtype=torch.bfloat16, device=dev).uniform_()
t = timed(lambda: x.sum(), 10)
print(f"read bandwidth (sum): {n / t / 1e9:.0f} GB/s")
del a, b, x
for s in (4096, 8192):
    m = torch.randn(s, s, dtype=torch.bfloat16, device=dev)
    t = timed(lambda: m @ m, 10)
    print(f"bf16 GEMM {s}^3: {2 * s**3 / t / 1e12:.1f} TFLOPS")
