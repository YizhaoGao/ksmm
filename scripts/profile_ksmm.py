#!/usr/bin/env python3
"""
Consolidated profiling script for KSMM kernels.

Tests all combinations of (a, b, c, d) patterns with given batch sizes,
benchmarking both CUDA and Triton implementations.
Each config runs with 100 warmup iterations and 1000 timed iterations.
Unsupported CUDA configs are left empty in the output CSV.

Usage:
    python scripts/profile_ksmm.py \
        --a_list 1 2 4 6 \
        --b_list 16 32 64 \
        --c_list 16 32 64 \
        --d_list 1 2 4 \
        --batch_sizes 1024 4096 \
        --dtype float16 \
        --output results/profile.csv
"""

import argparse
import csv
import itertools
import sys
import time
from typing import List, Tuple

import torch

# ── Triton forward (standalone, no autograd overhead) ────────────────────────
def triton_forward(X, K_bmm, pattern, layout='BSF'):
    """
    Wrapper around the Triton fused kernel forward pass.
    K_bmm: (a*d, c, b)  — same convention as ksmm_triton_tc.
    """
    from ksmm_triton.ksmm_triton_tc import ks_triton_forward_impl
    return ks_triton_forward_impl(X, K_bmm, pattern, layout=layout)


# ── CUDA kernel forward (via KSLinear) ───────────────────────────────────────
def cuda_forward(ksl, x):
    """Forward through the compiled CUDA KSLinear layer."""
    return ksl(x)


# ── Timing helper ────────────────────────────────────────────────────────────
def benchmark(func, args, warmup=100, iters=1000):
    """
    Returns average time in milliseconds over *iters* calls
    after *warmup* warm-up calls.  Uses cuda synchronize for accuracy.
    """
    # Warmup
    for _ in range(warmup):
        func(*args)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        func(*args)
    torch.cuda.synchronize()
    end = time.perf_counter()

    total_ms = (end - start) * 1000.0
    avg_ms = total_ms / iters
    return avg_ms


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Profile KSMM CUDA & Triton kernels across (a,b,c,d) configs."
    )
    parser.add_argument("--a_list", type=int, nargs="+", required=True)
    parser.add_argument("--b_list", type=int, nargs="+", required=True)
    parser.add_argument("--c_list", type=int, nargs="+", required=True)
    parser.add_argument("--d_list", type=int, nargs="+", required=True)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[4096])
    parser.add_argument(
        "--dtype", type=str, default="float16",
        choices=["float16", "float32"],
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--output", type=str, default="results/profile.csv")
    parser.add_argument(
        "--bs_last", action="store_true",
        help="Use batch-size-last layout (default: batch-size-first)",
    )
    parser.add_argument(
        "--triton_only", action="store_true",
        help="Only profile the Triton kernel, skip CUDA",
    )
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    device = "cuda"

    configs = list(itertools.product(
        args.a_list, args.b_list, args.c_list, args.d_list, args.batch_sizes,
    ))
    total = len(configs)
    print(f"Total configs to test: {total}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iters}")
    print(f"dtype: {args.dtype}, bs_last: {args.bs_last}")
    print()

    # CSV header
    fieldnames = [
        "a", "b", "c", "d", "batch_size",
        "dim_in", "dim_out",
        "dtype", "bs_last",
        "warmup", "iters",
        "cuda_ms", "triton_ms",
        "cuda_error",
    ]

    rows = []

    for idx, (a, b, c, d, batch_size) in enumerate(configs, 1):
        pattern = (a, b, c, d)
        dim_in = a * c * d
        dim_out = a * b * d

        tag = f"[{idx}/{total}] (a={a}, b={b}, c={c}, d={d}) BS={batch_size}"
        print(f"{tag} dim_in={dim_in} dim_out={dim_out}")

        row = dict(
            a=a, b=b, c=c, d=d, batch_size=batch_size,
            dim_in=dim_in, dim_out=dim_out,
            dtype=args.dtype, bs_last=args.bs_last,
            warmup=args.warmup, iters=args.iters,
            cuda_ms="", triton_ms="",
            cuda_error="",
        )

        # ── Create random input ──────────────────────────────────────────
        if args.bs_last:
            x = torch.randn(dim_in, batch_size, dtype=dtype, device=device)
        else:
            x = torch.randn(batch_size, dim_in, dtype=dtype, device=device)

        # ── 1. CUDA kernel ───────────────────────────────────────────────
        if not args.triton_only:
            try:
                from ksmm_py.layer.kronecker_sparse.interface import KSLinear

                ksl = KSLinear(
                    patterns=[pattern],
                    weights=None,
                    algo="kernel",
                    dtype=dtype,
                    bs_last=args.bs_last,
                    bias=False,
                    device=device,
                )
                # dry-run to trigger compilation
                _ = ksl(x)
                torch.cuda.synchronize()

                cuda_ms = benchmark(
                    cuda_forward, (ksl, x),
                    warmup=args.warmup, iters=args.iters,
                )
                row["cuda_ms"] = f"{cuda_ms:.6f}"
                print(f"  CUDA:   {cuda_ms:.4f} ms")
            except Exception as e:
                short = str(e).split("\n")[0][:120]
                row["cuda_error"] = short
                print(f"  CUDA:   SKIP ({short})")

        # ── 2. Triton kernel ─────────────────────────────────────────────
        try:
            import math
            scaling = 1.0 / math.sqrt(c)
            K_bmm = torch.empty(a * d, c, b, dtype=dtype, device=device).uniform_(
                -scaling, scaling
            )

            layout = "BSL" if args.bs_last else "BSF"
            # dry-run (also triggers triton autotune)
            _ = triton_forward(x, K_bmm, pattern, layout=layout)
            torch.cuda.synchronize()

            triton_ms = benchmark(
                triton_forward, (x, K_bmm, pattern, layout),
                warmup=args.warmup, iters=args.iters,
            )
            row["triton_ms"] = f"{triton_ms:.6f}"
            print(f"  Triton: {triton_ms:.4f} ms")
        except Exception as e:
            short = str(e).split("\n")[0][:120]
            row["triton_ms"] = ""
            print(f"  Triton: SKIP ({short})")

        rows.append(row)
        print()

    # ── Write CSV ─────────────────────────────────────────────────────────
    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Results saved to {args.output}")

    # ── Print summary table ───────────────────────────────────────────────
    print()
    hdr = f"{'a':>3} {'b':>4} {'c':>4} {'d':>3} {'BS':>6}  {'CUDA (ms)':>11}  {'Triton (ms)':>12}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        cuda_str = r["cuda_ms"] if r["cuda_ms"] else "—"
        triton_str = r["triton_ms"] if r["triton_ms"] else "—"
        print(
            f"{r['a']:>3} {r['b']:>4} {r['c']:>4} {r['d']:>3} {r['batch_size']:>6}"
            f"  {cuda_str:>11}  {triton_str:>12}"
        )


if __name__ == "__main__":
    main()
