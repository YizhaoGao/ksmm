#!/usr/bin/env bash
set -euo pipefail
mkdir -p results

# ── Group 1 CUDA: one CSV per batch size, from 2048 down to 2 ──
for bs in 2048 1024 512 256 128 64 32 16 8 4 2; do
    echo "============================================================"
    echo "  Group 1 CUDA — batch_size=${bs}"
    echo "============================================================"
    python scripts/profile_ksmm.py group1 \
        --batch_sizes ${bs} \
        --cuda_only \
        --output results/group1_cuda_bs${bs}.csv
done

echo ""
echo "All done. Results:"
ls -lh results/group1_cuda_bs*.csv