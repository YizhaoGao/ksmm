"""
Profiling script for fly butterfly kernels.

Group1: Single Factor – block_butterfly_factor_multiply (one factor at a time)
Group2: Block Butterfly – block_butterfly_multiply (full decomposition)
Group3: FlatBlock Butterfly – BlockdiagButterflyMultiply (collapsed two-stage bmm)

Sweeps:
  b (block size): 2, 4, 8, 16
  N (transform size): 512, 1024, 2048, 4096
  batch_size: 1, 2, 4, ..., 4096  (powers of 2)

Each config: 10 warmup iters + 100 timed iters (average).
"""

import argparse
import csv
import itertools
import math
import sys
import time

import torch

# ── kernels ──────────────────────────────────────────────────────────────────
from src.models.layers.block_butterfly_multiply import (
    block_butterfly_factor_multiply,
    block_butterfly_multiply,
)
from src.models.layers.blockdiag_butterfly_multiply import (
    BlockdiagButterflyMultiply,
)

blockdiag_butterfly_multiply = BlockdiagButterflyMultiply.apply

# ── helpers ──────────────────────────────────────────────────────────────────
WARMUP_ITERS = 10
BENCH_ITERS = 100

BLOCK_SIZES = [2, 4, 8, 16]
TRANSFORM_SIZES = [512, 1024, 2048, 4096]
BATCH_SIZES = [2**i for i in range(0, 13)]  # 1 .. 4096


def bench(fn, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Return average wall-clock time (ms) for *fn* on the current CUDA device."""
    # warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters  # ms


# ── Group 1: Single Factor ──────────────────────────────────────────────────
def profile_single_factor(b, N, batch, device="cuda"):
    """
    block_butterfly_factor_multiply:
      twiddle: (nstacks, log_n, n // 2, 2, 2, block_size, block_size)
      input:   (batch_size, nstacks, block_size * n)
    where n = N // b, block_size = b.
    We evaluate each factor_idx independently and return dict per factor.
    """
    block_size = b
    n = N // b
    log_n = int(math.log2(n))
    assert n == 2 ** log_n, f"N/b must be power of 2, got {n}"
    nstacks = 1

    twiddle = torch.randn(nstacks, log_n, n // 2, 2, 2, block_size, block_size,
                           device=device)
    x = torch.randn(batch, nstacks, block_size * n, device=device)

    results = []
    for factor_idx in range(log_n):
        ms = bench(lambda fi=factor_idx: block_butterfly_factor_multiply(
            twiddle, x, fi, increasing_stride=True))
        results.append({
            "group": "single_factor",
            "b": b, "N": N, "batch": batch,
            "factor_idx": factor_idx, "log_n": log_n,
            "time_ms": ms,
        })
    return results


# ── Group 2: Block Butterfly ────────────────────────────────────────────────
def profile_block_butterfly(b, N, batch, device="cuda"):
    """
    block_butterfly_multiply:
      twiddle: (nstacks, nblocks, log_n, n // 2, 2, 2, block_size, block_size)
      input:   (batch_size, nstacks, block_size * n)
    """
    block_size = b
    n = N // b
    log_n = int(math.log2(n))
    assert n == 2 ** log_n
    nstacks = 1
    nblocks = 1

    twiddle = torch.randn(nstacks, nblocks, log_n, n // 2, 2, 2,
                           block_size, block_size, device=device)
    x = torch.randn(batch, nstacks, block_size * n, device=device)

    ms = bench(lambda: block_butterfly_multiply(twiddle, x, increasing_stride=True))
    return [{
        "group": "block_butterfly",
        "b": b, "N": N, "batch": batch,
        "log_n": log_n,
        "time_ms": ms,
    }]


# ── Group 3: FlatBlock Butterfly (collapsed → two bmm stages) ───────────────
def profile_flatblock_butterfly(b, N, batch, device="cuda"):
    """
    BlockdiagButterflyMultiply.apply(x, w1_bfly, w2_bfly)
      x:       (batch, N)
      w1_bfly: (k, q, p)   k = N/p,  p = q = b  (square block)
      w2_bfly: (l, s, r)   l = k, s = r = b
    For a square (N→N) transform with block size b:
      p = q = s = r = b,  k = l = N // b
    """
    p = q = s = r = b
    k = N // p
    l = k  # since l * r == k * q  →  l = k*q/r = k (when q==r==b)

    w1 = torch.randn(k, q, p, device=device)
    w2 = torch.randn(l, s, r, device=device)
    x = torch.randn(batch, N, device=device)

    ms = bench(lambda: blockdiag_butterfly_multiply(x, w1, w2))
    return [{
        "group": "flatblock_butterfly",
        "b": b, "N": N, "batch": batch,
        "time_ms": ms,
    }]


# ── main driver ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Profile fly butterfly kernels")
    parser.add_argument("--groups", nargs="+",
                        default=["single_factor", "block_butterfly", "flatblock_butterfly"],
                        choices=["single_factor", "block_butterfly", "flatblock_butterfly"],
                        help="Which groups to profile")
    parser.add_argument("--out", type=str, default="profile_results.csv",
                        help="Output CSV file")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    assert torch.cuda.is_available(), "CUDA required"

    # print GPU info
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Warmup iters: {WARMUP_ITERS}, Bench iters: {BENCH_ITERS}")
    print(f"Block sizes (b): {BLOCK_SIZES}")
    print(f"Transform sizes (N): {TRANSFORM_SIZES}")
    print(f"Batch sizes: {BATCH_SIZES}")
    print(f"Groups: {args.groups}")
    print("-" * 80)

    all_results = []

    dispatch = {
        "single_factor": profile_single_factor,
        "block_butterfly": profile_block_butterfly,
        "flatblock_butterfly": profile_flatblock_butterfly,
    }

    for group in args.groups:
        fn = dispatch[group]
        configs = list(itertools.product(BLOCK_SIZES, TRANSFORM_SIZES, BATCH_SIZES))
        total = len(configs)
        for i, (b, N, batch) in enumerate(configs, 1):
            tag = f"[{group}] b={b} N={N} batch={batch}"
            try:
                results = fn(b, N, batch, device=device)
                for r in results:
                    all_results.append(r)
                    t = r["time_ms"]
                    extra = f" factor_idx={r['factor_idx']}" if "factor_idx" in r else ""
                    print(f"  {tag}{extra}  →  {t:.4f} ms")
            except Exception as e:
                print(f"  {tag}  SKIP ({e})")
            # free GPU memory between configs
            torch.cuda.empty_cache()

        print()

    # ── write CSV ────────────────────────────────────────────────────────────
    if not all_results:
        print("No results collected.")
        return

    fieldnames = ["group", "b", "N", "batch", "factor_idx", "log_n", "time_ms"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)

    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
