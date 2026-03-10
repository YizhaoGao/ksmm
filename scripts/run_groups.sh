#!/usr/bin/env bash
set -euo pipefail
mkdir -p results



python scripts/profile_ksmm.py group1 --num_gpus 8 --output results/group1_multigpu_nocuda.csv  --triton_only


# python scripts/profile_ksmm.py group2 --num_gpus 8 --output results/group2_multigpu.csv 


# echo "=== Group 1: Single Factor ==="
# python scripts/profile_ksmm.py group1 \
#     --output results/group1.csv 

# echo ""
# echo "=== Group 2: Butterfly Chain ==="
# python scripts/profile_ksmm.py group2 \
#     --output results/group2.csv 

# python scripts/profile_ksmm.py sweep \
#     --a_list 32 --b_list  32 --c_list 32 --d_list 2 \
#             --batch_sizes 1024