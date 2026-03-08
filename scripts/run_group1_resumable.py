#!/usr/bin/env python3
"""
Resumable Group 1 profiling with subprocess isolation.

Each (b,c,M,N,a,d) × batch_size config runs in its own subprocess so that
a CUDA crash (illegal memory access, etc.) cannot poison later configs.

Results are appended incrementally to the output CSV so you can
Ctrl-C at any time and resume later.

Usage:
    python scripts/run_group1_resumable.py                       # default
    python scripts/run_group1_resumable.py --output results/group1.csv
    python scripts/run_group1_resumable.py --triton_only         # skip CUDA
    python scripts/run_group1_resumable.py --cuda_only           # skip Triton (fill in missing CUDA)
    python scripts/run_group1_resumable.py --start_from 4287     # skip first N-1 configs
"""

import argparse
import csv
import itertools
import json
import math
import os
import subprocess
import sys
import time


# ── Helpers ──────────────────────────────────────────────────────────────────
def powers_of_2_range(lo, hi):
    result = []
    v = 1
    while v <= hi:
        if v >= lo:
            result.append(v)
        v *= 2
    return result


def is_power_of_2(n):
    return n > 0 and (n & (n - 1)) == 0


def generate_group1_configs():
    """Generate the exact same config list as run_group1 in profile_ksmm.py."""
    b_values = powers_of_2_range(2, 64)
    c_values = powers_of_2_range(2, 64)
    MN_values = [512, 1024, 2048, 4096]
    batch_sizes = powers_of_2_range(1, 4096)

    configs = []
    for b, c, M, N in itertools.product(b_values, c_values, MN_values, MN_values):
        if M * c != N * b:
            continue
        ad = M // b
        if ad < 1:
            continue
        for a in powers_of_2_range(1, ad):
            if ad % a != 0:
                continue
            d = ad // a
            if not is_power_of_2(d):
                continue
            configs.append((b, c, M, N, a, d))

    # Flatten: one entry per (config, batch_size)
    all_runs = []
    for b, c, M, N, a, d in configs:
        for bs in batch_sizes:
            all_runs.append((a, b, c, d, M, N, bs))
    return all_runs


def load_completed(csv_path):
    """Load already-completed (a,b,c,d,bs) keys from the CSV.
    A row is 'complete' if it has at least a triton_ms or cuda_ms value."""
    completed = set()
    if not os.path.exists(csv_path):
        return completed
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            has_triton = row.get("triton_ms", "").strip() != ""
            has_cuda = row.get("cuda_ms", "").strip() != ""
            if has_triton or has_cuda:
                key = (
                    int(row["a"]), int(row["b"]), int(row["c"]), int(row["d"]),
                    int(row["batch_size"]),
                )
                completed.add(key)
    return completed


# ── Path to the worker script ────────────────────────────────────────────────
WORKER_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_worker_group1.py")


def run_single_config(a, b, c, d, batch_size, dtype, warmup, iters,
                      do_cuda=True, do_triton=True, timeout=600):
    """Run a single config in a subprocess. Returns dict with results."""
    cfg = json.dumps({
        "a": a, "b": b, "c": c, "d": d, "batch_size": batch_size,
        "dtype": dtype, "warmup": warmup, "iters": iters,
        "do_cuda": do_cuda, "do_triton": do_triton,
    })

    try:
        proc = subprocess.run(
            [sys.executable, WORKER_SCRIPT_PATH, cfg],
            capture_output=True, text=True, timeout=timeout,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        if proc.returncode != 0:
            # Process crashed (CUDA error, segfault, etc.)
            stderr_short = proc.stderr.strip().split("\n")[-1][:200] if proc.stderr else "unknown"
            return {"cuda_ms": "", "triton_ms": "", "cuda_error": f"subprocess crashed: {stderr_short}"}

        # Parse the last line of stdout as JSON
        lines = [l.strip() for l in proc.stdout.strip().split("\n") if l.strip()]
        if not lines:
            return {"cuda_ms": "", "triton_ms": "", "cuda_error": "no output from subprocess"}
        return json.loads(lines[-1])

    except subprocess.TimeoutExpired:
        return {"cuda_ms": "", "triton_ms": "", "cuda_error": "timeout"}
    except Exception as e:
        return {"cuda_ms": "", "triton_ms": "", "cuda_error": str(e)[:200]}


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Resumable Group 1 profiling")
    parser.add_argument("--output", type=str, default="results/group1.csv")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--triton_only", action="store_true", help="Skip CUDA")
    parser.add_argument("--cuda_only", action="store_true", help="Skip Triton")
    parser.add_argument("--start_from", type=int, default=1,
                        help="1-based index to start from (skip earlier configs)")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Timeout per config in seconds")
    parser.add_argument("--no_resume", action="store_true",
                        help="Ignore existing CSV, start fresh")
    args = parser.parse_args()

    all_runs = generate_group1_configs()
    total = len(all_runs)

    # Load already-completed configs
    if args.no_resume:
        completed = set()
    else:
        completed = load_completed(args.output)
    print(f"Total configs: {total}, already completed: {len(completed)}")

    fieldnames = [
        "group", "b", "c", "M", "N", "a", "d", "batch_size",
        "triton_ms", "cuda_ms", "cuda_error",
    ]

    # Create output file with header if it doesn't exist
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_header = not os.path.exists(args.output) or args.no_resume
    if write_header:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    do_cuda = not args.triton_only
    do_triton = not args.cuda_only

    skipped = 0
    done = 0
    t0 = time.time()

    for idx, (a, b, c, d, M, N, bs) in enumerate(all_runs, 1):
        if idx < args.start_from:
            continue

        key = (a, b, c, d, bs)
        if key in completed:
            skipped += 1
            continue

        tag = f"[{idx}/{total}] (a={a},b={b},c={c},d={d}) M={M} N={N} BS={bs}"
        print(tag, end="  ", flush=True)

        result = run_single_config(
            a, b, c, d, bs, args.dtype, args.warmup, args.iters,
            do_cuda=do_cuda, do_triton=do_triton, timeout=args.timeout,
        )

        cuda_str = result.get("cuda_ms", "")
        triton_str = result.get("triton_ms", "")
        error_str = result.get("cuda_error", "")

        parts = []
        if cuda_str:
            parts.append(f"CUDA={float(cuda_str):.4f}ms")
        elif error_str:
            parts.append(f"CUDA=SKIP")
        if triton_str:
            parts.append(f"Triton={float(triton_str):.4f}ms")
        print("  ".join(parts) if parts else "SKIP")

        row = dict(
            group="group1", b=b, c=c, M=M, N=N, a=a, d=d,
            batch_size=bs,
            triton_ms=triton_str,
            cuda_ms=cuda_str,
            cuda_error=error_str,
        )

        # Append incrementally
        with open(args.output, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

        done += 1

    elapsed = time.time() - t0
    print(f"\nDone. {done} new configs profiled, {skipped} skipped (already in CSV).")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Results in {args.output}")


if __name__ == "__main__":
    main()
