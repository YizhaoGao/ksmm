#!/usr/bin/env python3
"""Worker subprocess for run_group1_resumable.py. Do not run directly."""
import json
import math
import os
import sys
import time

# Add src/ to path relative to this script's location
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, os.path.join(_project_root, 'src'))

import torch


def benchmark(func, args, warmup=100, iters=1000):
    for _ in range(warmup):
        func(*args)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        func(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def main():
    cfg = json.loads(sys.argv[1])
    a, b, c, d = cfg["a"], cfg["b"], cfg["c"], cfg["d"]
    batch_size = cfg["batch_size"]
    warmup, iters = cfg["warmup"], cfg["iters"]
    dtype = torch.float16 if cfg["dtype"] == "float16" else torch.float32
    do_cuda = cfg.get("do_cuda", True)
    do_triton = cfg.get("do_triton", True)
    device = "cuda"

    pattern = (a, b, c, d)
    dim_in = a * c * d

    result = {"cuda_ms": "", "triton_ms": "", "cuda_error": ""}

    x = torch.randn(batch_size, dim_in, dtype=dtype, device=device)

    # CUDA kernel
    if do_cuda:
        try:
            from ksmm_py.layer.kronecker_sparse.interface import KSLinear
            ksl = KSLinear(
                patterns=[pattern], weights=None, algo="kernel",
                dtype=dtype, bs_last=False, bias=False, device=device,
            )
            _ = ksl(x)
            torch.cuda.synchronize()
            cuda_ms = benchmark(lambda *a: ksl(a[0]), (x,), warmup=warmup, iters=iters)
            result["cuda_ms"] = f"{cuda_ms:.6f}"
        except Exception as e:
            result["cuda_error"] = str(e).split("\n")[0][:200]

    # Triton kernel
    if do_triton:
        try:
            from ksmm_triton.ksmm_triton_tc import ks_triton_forward_impl
            scaling = 1.0 / math.sqrt(c)
            K_bmm = torch.empty(a * d, c, b, dtype=dtype, device=device).uniform_(
                -scaling, scaling
            )
            _ = ks_triton_forward_impl(x, K_bmm, pattern, layout="BSF")
            torch.cuda.synchronize()
            triton_ms = benchmark(
                ks_triton_forward_impl, (x, K_bmm, pattern, "BSF"),
                warmup=warmup, iters=iters,
            )
            result["triton_ms"] = f"{triton_ms:.6f}"
        except Exception as e:
            result["triton_ms"] = ""

    print(json.dumps(result))


if __name__ == "__main__":
    main()
