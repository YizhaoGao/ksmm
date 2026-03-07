#!/usr/bin/env python3
"""
Consolidated profiling script for KSMM kernels.

Subcommands:
  sweep   – Original parameter sweep over (a, b, c, d) lists.
  group1  – Single-factor profiling: sweep (a, d) for fixed (M, N, b, c).
  group2  – Butterfly-chain profiling: decompose N into stages, profile chain.

Usage:
    # Original sweep (default)
    python scripts/profile_ksmm.py sweep \
        --a_list 1 2 4 6 --b_list 16 32 --c_list 16 32 --d_list 1 2 \
        --batch_sizes 1024 4096 --output results/profile.csv

    # Group 1 – single factor
    python scripts/profile_ksmm.py group1 --output results/group1.csv --triton_only

    # Group 2 – butterfly chain
    python scripts/profile_ksmm.py group2 --output results/group2.csv --triton_only
"""

import argparse
import csv
import itertools
import math
import os
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


# ── Helpers ──────────────────────────────────────────────────────────────────
def powers_of_2_range(lo, hi):
    """Return all powers of 2 in [lo, hi]."""
    result = []
    v = 1
    while v <= hi:
        if v >= lo:
            result.append(v)
        v *= 2
    return result


def is_power_of_2(n):
    return n > 0 and (n & (n - 1)) == 0


# ── Sweep (original main) ───────────────────────────────────────────────────
def run_sweep(args):
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
            scaling = 1.0 / math.sqrt(c)
            K_bmm = torch.empty(a * d, c, b, dtype=dtype, device=device).uniform_(
                -scaling, scaling
            )

            layout = "BSL" if args.bs_last else "BSF"
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


# ── Group 1: Single Factor ──────────────────────────────────────────────────
def run_group1(args):
    """
    Sweep (a, d) for each valid (b, c, M, N) combination.
    Fixed BSF layout to avoid measuring data transpose overhead.
    """
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    device = "cuda"
    layout = "BSF"

    b_values = powers_of_2_range(2, 64)
    c_values = powers_of_2_range(2, 64)
    MN_values = [512, 1024, 2048, 4096]
    batch_sizes = powers_of_2_range(2, 4096)

    # Enumerate all valid configs
    configs = []
    for b, c, M, N in itertools.product(b_values, c_values, MN_values, MN_values):
        # Require M/b == N/c  (i.e. M*c == N*b)
        if M * c != N * b:
            continue
        ad = M // b  # = N // c
        if ad < 1:
            continue
        # Enumerate (a, d) power-of-2 factor pairs of ad
        for a in powers_of_2_range(1, ad):
            if ad % a != 0:
                continue
            d = ad // a
            if not is_power_of_2(d):
                continue
            configs.append((b, c, M, N, a, d))

    total_configs = len(configs)
    total_runs = total_configs * len(batch_sizes)
    print(f"Group 1: {total_configs} (b,c,M,N,a,d) configs × {len(batch_sizes)} batch sizes = {total_runs} runs")
    print(f"Warmup: {args.warmup}, Iterations: {args.iters}, dtype: {args.dtype}")
    print()

    fieldnames = [
        "group", "b", "c", "M", "N", "a", "d", "batch_size",
        "triton_ms", "cuda_ms", "cuda_error",
    ]
    rows = []
    run_idx = 0

    for b, c, M, N, a, d in configs:
        pattern = (a, b, c, d)
        dim_in = a * c * d   # = N
        dim_out = a * b * d  # = M

        for batch_size in batch_sizes:
            run_idx += 1
            tag = f"[{run_idx}/{total_runs}] (a={a},b={b},c={c},d={d}) M={M} N={N} BS={batch_size}"
            print(tag)

            row = dict(
                group="group1", b=b, c=c, M=M, N=N, a=a, d=d,
                batch_size=batch_size,
                triton_ms="", cuda_ms="", cuda_error="",
            )

            x = torch.randn(batch_size, dim_in, dtype=dtype, device=device)

            # CUDA
            if not args.triton_only:
                try:
                    from ksmm_py.layer.kronecker_sparse.interface import KSLinear
                    ksl = KSLinear(
                        patterns=[pattern], weights=None, algo="kernel",
                        dtype=dtype, bs_last=False, bias=False, device=device,
                    )
                    _ = ksl(x)
                    torch.cuda.synchronize()
                    cuda_ms = benchmark(cuda_forward, (ksl, x),
                                        warmup=args.warmup, iters=args.iters)
                    row["cuda_ms"] = f"{cuda_ms:.6f}"
                    print(f"  CUDA:   {cuda_ms:.4f} ms")
                except Exception as e:
                    short = str(e).split("\n")[0][:120]
                    row["cuda_error"] = short
                    print(f"  CUDA:   SKIP ({short})")

            # Triton
            try:
                scaling = 1.0 / math.sqrt(c)
                K_bmm = torch.empty(a * d, c, b, dtype=dtype, device=device).uniform_(
                    -scaling, scaling
                )
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
                print(f"  Triton: SKIP ({short})")

            rows.append(row)

    # Write CSV
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {args.output}  ({len(rows)} rows)")


# ── Group 2: Butterfly Chain ────────────────────────────────────────────────
def run_group2(args):
    """
    For each N=2^n, decompose into n butterfly stages and profile
    each stage individually plus the full chain end-to-end.
    """
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    device = "cuda"
    layout = "BSF"

    N_values = [512, 1024, 2048, 4096]
    batch_sizes = powers_of_2_range(2, 4096)

    fieldnames = [
        "group", "N", "n", "batch_size", "stage",
        "a", "b", "c", "d",
        "stage_triton_ms", "total_chain_triton_ms",
    ]
    rows = []

    for N in N_values:
        n = int(math.log2(N))
        assert 2 ** n == N

        # Build stages: stage l (1..n) has a=2^(l-1), b=2, c=2, d=2^(n-l)
        # Chain order: stage n → n-1 → ... → 1
        stages = []
        for l in range(n, 0, -1):
            a = 2 ** (l - 1)
            d = 2 ** (n - l)
            stages.append((a, 2, 2, d))  # (a, b, c, d)

        print(f"N={N}, n={n}, stages={len(stages)}")
        for i, (a, b, c, d) in enumerate(stages):
            print(f"  chain[{i}]: a={a}, b=2, c=2, d={d}  (stage l={n-i})")

        for batch_size in batch_sizes:
            print(f"  BS={batch_size}")

            # Prepare kernels and do dry-runs for all stages
            stage_kernels = []
            for a, b, c, d in stages:
                pattern = (a, b, c, d)
                scaling = 1.0 / math.sqrt(c)
                K_bmm = torch.empty(a * d, c, b, dtype=dtype, device=device).uniform_(
                    -scaling, scaling
                )
                stage_kernels.append((pattern, K_bmm))

            # Profile each stage individually
            stage_times = []
            x_stage = torch.randn(batch_size, N, dtype=dtype, device=device)

            for si, ((a, b, c, d), (pattern, K_bmm)) in enumerate(zip(stages, stage_kernels)):
                try:
                    # Create appropriate input for this stage
                    dim_in = a * c * d  # = N for all butterfly stages
                    x_single = torch.randn(batch_size, dim_in, dtype=dtype, device=device)
                    _ = triton_forward(x_single, K_bmm, pattern, layout=layout)
                    torch.cuda.synchronize()

                    ms = benchmark(
                        triton_forward, (x_single, K_bmm, pattern, layout),
                        warmup=args.warmup, iters=args.iters,
                    )
                    stage_times.append(ms)
                    print(f"    stage {si} (l={n-si}): {ms:.4f} ms")
                except Exception as e:
                    stage_times.append(None)
                    short = str(e).split("\n")[0][:120]
                    print(f"    stage {si} (l={n-si}): SKIP ({short})")

            # Profile the full chain end-to-end
            def chain_forward():
                x = torch.randn(batch_size, N, dtype=dtype, device=device)
                for pattern, K_bmm in stage_kernels:
                    x = triton_forward(x, K_bmm, pattern, layout=layout)
                return x

            try:
                _ = chain_forward()
                torch.cuda.synchronize()
                chain_ms = benchmark(
                    lambda: chain_forward(), (),
                    warmup=args.warmup, iters=args.iters,
                )
                print(f"    chain total: {chain_ms:.4f} ms")
            except Exception as e:
                chain_ms = None
                short = str(e).split("\n")[0][:120]
                print(f"    chain total: SKIP ({short})")

            # Write one row per stage
            for si, ((a, b, c, d), st) in enumerate(zip(stages, stage_times)):
                row = dict(
                    group="group2", N=N, n=n, batch_size=batch_size,
                    stage=si, a=a, b=b, c=c, d=d,
                    stage_triton_ms=f"{st:.6f}" if st is not None else "",
                    total_chain_triton_ms=f"{chain_ms:.6f}" if (chain_ms is not None and si == 0) else "",
                )
                rows.append(row)

    # Write CSV
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {args.output}  ({len(rows)} rows)")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Profile KSMM CUDA & Triton kernels.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Common arguments shared across subcommands
    def add_common_args(p):
        p.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "float32"])
        p.add_argument("--warmup", type=int, default=100)
        p.add_argument("--iters", type=int, default=1000)
        p.add_argument("--output", type=str, default="results/profile.csv")
        p.add_argument("--triton_only", action="store_true",
                        help="Only profile Triton, skip CUDA")

    # ── sweep (original) ─────────────────────────────────────────────────
    sweep_parser = subparsers.add_parser("sweep", help="Original parameter sweep")
    sweep_parser.add_argument("--a_list", type=int, nargs="+", required=True)
    sweep_parser.add_argument("--b_list", type=int, nargs="+", required=True)
    sweep_parser.add_argument("--c_list", type=int, nargs="+", required=True)
    sweep_parser.add_argument("--d_list", type=int, nargs="+", required=True)
    sweep_parser.add_argument("--batch_sizes", type=int, nargs="+", default=[4096])
    sweep_parser.add_argument("--bs_last", action="store_true",
                              help="Use batch-size-last layout")
    add_common_args(sweep_parser)

    # ── group1 ───────────────────────────────────────────────────────────
    group1_parser = subparsers.add_parser("group1",
        help="Single-factor profiling: sweep (a,d) for fixed (M,N,b,c)")
    add_common_args(group1_parser)

    # ── group2 ───────────────────────────────────────────────────────────
    group2_parser = subparsers.add_parser("group2",
        help="Butterfly-chain profiling: decompose N into stages")
    add_common_args(group2_parser)

    args = parser.parse_args()

    if args.command == "sweep":
        run_sweep(args)
    elif args.command == "group1":
        run_group1(args)
    elif args.command == "group2":
        run_group2(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
