#!/usr/bin/env python3
"""Merge CUDA kernel results from multiple CSV files.

For each unique config (group, b, c, M, N, a, d, batch_size), collects
valid cuda_ms results from any available file. Later files override earlier
ones, so the bs-specific files (which ran high-to-low) take priority over
the main file (which ran low-to-high and hit illegal memory on large bs).
"""
import csv
import glob
import os

RESULTS_DIR = "results"
MAIN_FILE = os.path.join(RESULTS_DIR, "group1_cuda.csv")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "group1_cuda_merged.csv")

CONFIG_KEYS = ("group", "b", "c", "M", "N", "a", "d", "batch_size")

# Collect all input files: main file first, then bs-specific files
input_files = [MAIN_FILE]
for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "group1_cuda_bs*.csv"))):
    input_files.append(f)
for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "group1_cuda_[0-9]*.csv"))):
    input_files.append(f)

print("Input files:")
for f in input_files:
    print(f"  {f}")

# Dict: config_tuple -> row dict (with valid cuda_ms)
# Also track all configs in order from main file
all_configs = {}  # config -> row (possibly with error)
valid_results = {}  # config -> row (only valid)

# First pass: load all rows from main file to get the full config space
with open(MAIN_FILE) as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        key = tuple(row[k] for k in CONFIG_KEYS)
        all_configs[key] = row

# Second pass: collect valid results from all files (later files override)
for filepath in input_files:
    count = 0
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = tuple(row[k] for k in CONFIG_KEYS)
            if row["cuda_ms"] and not row.get("cuda_error", "").startswith("ERROR"):
                valid_results[key] = row
                count += 1
            # Also register configs not in main file
            if key not in all_configs:
                all_configs[key] = row
    print(f"  {filepath}: {count} valid results collected")

# Merge: for each config, use valid result if available, otherwise keep error row
merged = []
for key, row in all_configs.items():
    if key in valid_results:
        merged.append(valid_results[key])
    else:
        merged.append(row)

# Count stats
n_valid = sum(1 for key in all_configs if key in valid_results)
n_error = len(all_configs) - n_valid
print(f"\nMerged: {n_valid} valid, {n_error} still errored, {len(all_configs)} total configs")

# Write output
with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(merged)

print(f"Written to {OUTPUT_FILE}")
