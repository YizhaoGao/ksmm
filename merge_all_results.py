#!/usr/bin/env python3
"""Merge triton/bmm results (group1_multigpu_nocuda.csv) with cuda results
(group1_cuda_merged.csv) into a single final CSV.

For each config, takes triton_ms and bmm_ms from the nocuda file, and
cuda_ms/cuda_error from the merged cuda file.
"""
import csv

NOCUDA_FILE = "results/group1_multigpu_nocuda.csv"
CUDA_FILE = "results/group1_cuda_merged.csv"
OUTPUT_FILE = "results/group1_final.csv"

CONFIG_KEYS = ("group", "b", "c", "M", "N", "a", "d", "batch_size")

# Load cuda results keyed by config
cuda_results = {}
with open(CUDA_FILE) as f:
    for row in csv.DictReader(f):
        key = tuple(row[k] for k in CONFIG_KEYS)
        cuda_results[key] = row

# Load nocuda (triton + bmm) and merge
merged = []
with open(NOCUDA_FILE) as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        key = tuple(row[k] for k in CONFIG_KEYS)
        # Take triton_ms and bmm_ms from nocuda file (already there)
        # Overlay cuda_ms and cuda_error from cuda file
        if key in cuda_results:
            row["cuda_ms"] = cuda_results[key]["cuda_ms"]
            row["cuda_error"] = cuda_results[key]["cuda_error"]
        merged.append(row)

# Stats
n_triton = sum(1 for r in merged if r.get("triton_ms"))
n_bmm = sum(1 for r in merged if r.get("bmm_ms"))
n_cuda = sum(1 for r in merged if r.get("cuda_ms") and not r.get("cuda_error", "").startswith("ERROR"))
print(f"Total configs: {len(merged)}")
print(f"  triton_ms available: {n_triton}")
print(f"  bmm_ms available:    {n_bmm}")
print(f"  cuda_ms available:   {n_cuda}")

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(merged)

print(f"Written to {OUTPUT_FILE}")
