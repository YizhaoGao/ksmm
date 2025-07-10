import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np
from typing import List, Tuple

def parse_pattern(pattern_str):
    """Parse pattern string to extract dimensions"""
    try:
        pattern = ast.literal_eval(pattern_str)
        return tuple(pattern[0])  # Return first pattern as tuple for comparison
    except:
        return None

def parse_pattern_list(pattern_str):
    """Parse pattern string to extract list of tuples"""
    try:
        pattern = ast.literal_eval(pattern_str)
        return [tuple(x) for x in pattern]
    except:
        return None

def calculate_dimensions(patterns: List[Tuple[int, int, int, int]]) -> Tuple[int, int]:
    """
    Calculate input and output dimensions from patterns.
    patterns[0] corresponds to rightmost factor (K_L)
    patterns[-1] corresponds to leftmost factor (K_1)
    """
    dim_in = patterns[0][0] * patterns[0][2] * patterns[0][3]
    dim_out = patterns[-1][0] * patterns[-1][1] * patterns[-1][3]
    return dim_in, dim_out

def main():
    # Read the CSV files
    kernel_df = pd.read_csv('../results/kernel.csv')
    dense_df = pd.read_csv('../results/dense.csv')
    
    # Parse patterns for both dataframes
    kernel_df['parsed_pattern'] = kernel_df['patterns'].apply(parse_pattern)
    dense_df['parsed_pattern'] = dense_df['patterns'].apply(parse_pattern)
    kernel_df['pattern_list'] = kernel_df['patterns'].apply(parse_pattern_list)
    
    # Remove rows where pattern parsing failed
    kernel_df = kernel_df.dropna(subset=['parsed_pattern', 'pattern_list'])
    dense_df = dense_df.dropna(subset=['parsed_pattern'])
    
    # Find matching patterns between kernel and dense
    common_patterns = set(kernel_df['parsed_pattern']) & set(dense_df['parsed_pattern'])
    
    # Filter dataframes to only include common patterns
    kernel_filtered = kernel_df[kernel_df['parsed_pattern'].isin(common_patterns)]
    dense_filtered = dense_df[dense_df['parsed_pattern'].isin(common_patterns)]
    
    # Create dictionaries for quick lookup
    kernel_times = {}
    dense_times = {}
    kernel_pattern_lists = {}
    
    for _, row in kernel_filtered.iterrows():
        pattern = row['parsed_pattern']
        kernel_times[pattern] = row['mean']
        kernel_pattern_lists[pattern] = row['pattern_list']
    
    for _, row in dense_filtered.iterrows():
        pattern = row['parsed_pattern']
        dense_times[pattern] = row['mean']
    
    # Calculate speedups
    patterns = []
    speedups = []
    theoretical_speedups = []
    
    for pattern in common_patterns:
        if pattern in kernel_times and pattern in dense_times and pattern in kernel_pattern_lists:
            speedup = dense_times[pattern] / kernel_times[pattern]
            patterns.append(pattern)
            speedups.append(speedup)
            # Theoretical speedup
            pattern_list = kernel_pattern_lists[pattern]
            dim_in, dim_out = calculate_dimensions(pattern_list)
            a, b, c, d = pattern
            theoretical = (dim_in * dim_out) / (a * b * c * d)
            theoretical_speedups.append(theoretical)
    
    # Sort by speedup for better visualization
    sorted_data = sorted(zip(patterns, speedups, theoretical_speedups), key=lambda x: x[1])
    patterns, speedups, theoretical_speedups = zip(*sorted_data)
    
    # Create pattern labels for y-axis
    pattern_labels = [f"{p[0]}×{p[1]}×{p[2]}×{p[3]}" for p in patterns]
    
    # Create the transposed plot
    plt.figure(figsize=(8, max(12, len(speedups) // 2)))
    bars = plt.barh(range(len(speedups)), speedups, color='skyblue', alpha=0.7, label='Measured Speedup')
    plt.plot(theoretical_speedups, range(len(theoretical_speedups)), color='orange', marker='o', linestyle='-', label='Theoretical Speedup')
    
    plt.axvline(x=1, color='red', linestyle='--', alpha=0.5, label='No speedup (1x)')
    plt.ylabel('Pattern (Batch×H×W×Channels)')
    plt.xlabel('Speedup (Dense time / Kernel time)')
    plt.title('Speedup of Kernel vs Dense Implementation')
    plt.yticks(range(len(pattern_labels)), pattern_labels)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add value labels on bars
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                 f'{speedup:.2f}x', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('../results/speedup_plot_transposed.png', dpi=300)
    plt.show()
    
    # Print summary statistics
    print(f"Number of matching patterns: {len(speedups)}")
    print(f"Average measured speedup: {np.mean(speedups):.2f}x")
    print(f"Average theoretical speedup: {np.mean(theoretical_speedups):.2f}x")
    print(f"Median measured speedup: {np.median(speedups):.2f}x")
    print(f"Median theoretical speedup: {np.median(theoretical_speedups):.2f}x")
    print(f"Max measured speedup: {max(speedups):.2f}x")
    print(f"Max theoretical speedup: {max(theoretical_speedups):.2f}x")
    print(f"Min measured speedup: {min(speedups):.2f}x")
    print(f"Min theoretical speedup: {min(theoretical_speedups):.2f}x")
    
    # Print top 5 measured speedups
    print("\nTop 5 measured speedups:")
    for i, (pattern, speedup, theoretical) in enumerate(sorted_data[-5:]):
        print(f"{i+1}. {pattern}: {speedup:.2f}x (theoretical: {theoretical:.2f}x)")

if __name__ == "__main__":
    main()