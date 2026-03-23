#!/usr/bin/env python3
"""
Roofline model plots for KSMM benchmarks on A100 40GB SXM4 with BF16.

A100 40GB SXM4 specs:
  - Peak BF16 (tensor core): 312 TFLOPS
  - HBM2e bandwidth: 1555 GB/s
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# --- A100 40GB SXM4 specs ---
PEAK_FLOPS = 312e3  # GFLOPS (312 TFLOPS)
PEAK_BW = 1555  # GB/s
DTYPE_BYTES = 2  # bf16

RESULTS_DIR = Path(__file__).parent.parent / "results"


def compute_stage_flops_bytes(a, b, c, d, batch_size):
    """Compute FLOPs and memory bytes for a single KSMM stage."""
    flops = 2 * batch_size * a * b * c * d
    bytes_read_input = batch_size * a * c * d * DTYPE_BYTES
    bytes_read_weight = a * d * b * c * DTYPE_BYTES
    bytes_write_output = batch_size * a * b * d * DTYPE_BYTES
    total_bytes = bytes_read_input + bytes_read_weight + bytes_write_output
    return flops, total_bytes


def draw_roofline(ax, label_ridge=True):
    """Draw the roofline ceiling on the given axes."""
    ridge_point = PEAK_FLOPS / PEAK_BW  # FLOP/byte
    ai_range = np.logspace(-2, 4, 1000)

    roofline = np.minimum(PEAK_FLOPS, PEAK_BW * ai_range)
    ax.plot(ai_range, roofline, 'k-', linewidth=2, zorder=1)

    # Memory-bound region (slope)
    mem_bound = ai_range[ai_range <= ridge_point]
    ax.fill_between(mem_bound, PEAK_BW * mem_bound, alpha=0.05, color='blue')

    # Annotations
    ax.axhline(y=PEAK_FLOPS, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=ridge_point, color='gray', linestyle='--', alpha=0.3)

    if label_ridge:
        ax.annotate(f'Peak BF16: {PEAK_FLOPS/1e3:.0f} TFLOPS',
                     xy=(ridge_point * 2, PEAK_FLOPS * 0.85),
                     fontsize=8, color='gray')
        ax.annotate(f'BW: {PEAK_BW} GB/s',
                     xy=(0.02, PEAK_BW * 0.02 * 1.3),
                     fontsize=8, color='gray', rotation=38)
        ax.annotate(f'Ridge: {ridge_point:.1f} F/B',
                     xy=(ridge_point, 1),
                     fontsize=7, color='gray', ha='center')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)')
    ax.set_ylabel('Performance (GFLOPS)')
    ax.grid(True, alpha=0.3, which='both')


def plot_group1():
    """Roofline plot for group1_cuda_merged.csv (single stage benchmarks)."""
    df = pd.read_csv(RESULTS_DIR / "group1_cuda_merged.csv")

    fig, ax = plt.subplots(figsize=(14, 10))
    draw_roofline(ax)
    ax.set_title("Roofline Model — Group 1 Single Stage (A100 40GB SXM4, BF16)")

    # Collect all valid data points for each implementation
    impl_styles = {
        'triton_ms': ('o', 'Triton'),
        'bmm_ms': ('^', 'BMM'),
        'cuda_ms': ('s', 'CUDA'),
    }

    # Get unique (b,c) pairs to assign colors
    df['bc_key'] = list(zip(df['b'], df['c']))
    bc_pairs = sorted(df['bc_key'].unique(), key=lambda x: (x[0], x[1]))
    cmap = plt.cm.tab10
    bc_colors = {bc: cmap(i % 10) for i, bc in enumerate(bc_pairs)}

    # Plot each implementation
    for impl_col, (marker, impl_label) in impl_styles.items():
        mask = df[impl_col].notna() & (df[impl_col] > 0)
        if impl_col == 'cuda_ms':
            mask = mask & (df['cuda_error'].isna() | (df['cuda_error'] == ''))
        sub = df[mask].copy()
        if sub.empty:
            continue

        flops_arr = []
        bytes_arr = []
        for _, row in sub.iterrows():
            f, b_mem = compute_stage_flops_bytes(
                int(row['a']), int(row['b']), int(row['c']), int(row['d']),
                int(row['batch_size']))
            flops_arr.append(f)
            bytes_arr.append(b_mem)

        flops_arr = np.array(flops_arr, dtype=np.float64)
        bytes_arr = np.array(bytes_arr, dtype=np.float64)
        time_ms = sub[impl_col].values.astype(np.float64)

        ai = flops_arr / bytes_arr  # FLOP/Byte
        perf = flops_arr / (time_ms * 1e-3) / 1e9  # GFLOPS

        colors = [bc_colors[bc] for bc in sub['bc_key']]

        ax.scatter(ai, perf, c=colors, marker=marker, s=20, alpha=0.6,
                   label=impl_label, edgecolors='none')

    # Build legend: implementation markers + (b,c) colors
    from matplotlib.lines import Line2D
    legend_elements = []
    # Implementation markers
    for impl_col, (marker, impl_label) in impl_styles.items():
        legend_elements.append(
            Line2D([0], [0], marker=marker, color='gray', label=impl_label,
                   markersize=6, linestyle='None'))
    legend_elements.append(Line2D([0], [0], color='none', label=''))  # spacer
    # (b,c) colors
    for bc in bc_pairs:
        legend_elements.append(
            Line2D([0], [0], marker='o', color=bc_colors[bc],
                   label=f'b={bc[0]}, c={bc[1]}',
                   markersize=6, linestyle='None'))

    ax.legend(handles=legend_elements, fontsize=7, ncol=2,
              loc='lower right', framealpha=0.8)

    ax.set_xlim(1e-2, 1e4)
    ax.set_ylim(1e-1, PEAK_FLOPS * 2)

    out_path = RESULTS_DIR / "roofline_group1.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_group2():
    """Roofline plot for group2_multigpu.csv (total butterfly chain)."""
    df = pd.read_csv(RESULTS_DIR / "group2_multigpu.csv")

    # Only stage=0 rows have total chain times
    chain = df[df['stage'] == 0].copy()

    fig, ax = plt.subplots(figsize=(14, 10))
    draw_roofline(ax)
    ax.set_title("Roofline Model — Group 2 Butterfly Chain (A100 40GB SXM4, BF16)")

    # For each (N, batch_size), compute total chain FLOPs and bytes
    # Chain has n stages, each with b=c=2, and varying (a,d) but a*c*d = N
    # Per stage: FLOPs = 2*B*a*2*2*d = 8*B*(N/2) = 4*B*N  (since a*2*d = N → a*d = N/2)
    # Per stage bytes: B*N*2 (read input) + (N/2)*4*2 (read weight) + B*N*2 (write output)
    #                = 4*B*N + 4*N = 4*N*(B+1)
    # Total chain: n stages
    # Total FLOPs = n * 4 * B * N
    # Total bytes = n * 4 * N * (B + 1)

    N_values = sorted(chain['N'].unique())
    cmap = plt.cm.Set1
    n_colors = {N: cmap(i) for i, N in enumerate(N_values)}

    impl_data = {
        'total_chain_triton_ms': ('o', 'Triton', '-'),
        'total_chain_bmm_ms': ('^', 'BMM', '--'),
    }

    for impl_col, (marker, impl_label, ls) in impl_data.items():
        for N_val in N_values:
            sub = chain[chain['N'] == N_val].copy()
            mask = sub[impl_col].notna() & (sub[impl_col] > 0)
            sub = sub[mask]
            if sub.empty:
                continue

            n_stages = sub['n'].values.astype(int)
            batch_sizes = sub['batch_size'].values.astype(int)
            time_ms = sub[impl_col].values.astype(np.float64)

            total_flops = n_stages * 4.0 * batch_sizes * N_val
            total_bytes = n_stages * 4.0 * N_val * (batch_sizes + 1)

            ai = total_flops / total_bytes
            perf = total_flops / (time_ms * 1e-3) / 1e9  # GFLOPS

            # Sort by batch_size for line plot
            order = np.argsort(batch_sizes)
            ax.plot(ai[order], perf[order], marker=marker, color=n_colors[N_val],
                    linestyle=ls, markersize=5, alpha=0.8,
                    label=f'N={N_val} ({impl_label})')

            # Annotate batch sizes
            for i in order:
                bs = batch_sizes[i]
                if bs in [2, 64, 4096]:  # label a few
                    ax.annotate(f'B={bs}', (ai[i], perf[i]),
                                fontsize=5, alpha=0.6,
                                textcoords='offset points', xytext=(3, 3))

    ax.legend(fontsize=7, ncol=2, loc='lower right', framealpha=0.8)
    ax.set_xlim(1e-2, 1e4)
    ax.set_ylim(1e-1, PEAK_FLOPS * 2)

    out_path = RESULTS_DIR / "roofline_group2.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    plot_group1()
    plot_group2()
