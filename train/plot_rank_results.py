#!/usr/bin/env python3
"""
Plot rank sweep results from saved data.

This script can re-create plots from previously saved rank sweep study results.
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Plot rank sweep results')
    
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing rank sweep results')
    parser.add_argument('--save_prefix', type=str, default='rank_sweep',
                        help='Prefix used for saved files (default: rank_sweep)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save plots (default: same as results_dir)')
    parser.add_argument('--log_scale', action='store_true',
                        help='Use log scale for loss axis')
    parser.add_argument('--title', type=str, default='Rank vs Training Loss Convergence',
                        help='Title for the plots')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for saved plots (default: 300)')
    parser.add_argument('--figsize', type=int, nargs=2, default=[15, 6],
                        help='Figure size for loss curves plot (default: [15, 6])')
    
    return parser.parse_args()

def load_all_loss_curves(results_dir: str):
    """Load loss curves from all rank directories."""
    loss_data = {}
    
    # Look for rank directories
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path) and item.startswith('rank_'):
            try:
                rank = int(item.split('_')[1])
                
                # Look for CSV files in the rank directory
                csv_files = [f for f in os.listdir(item_path) if f.endswith('_loss_curves.csv')]
                
                if csv_files:
                    csv_file = os.path.join(item_path, csv_files[0])
                    df = pd.read_csv(csv_file)
                    
                    loss_data[rank] = {
                        'epochs': df['epoch'].tolist(),
                        'train_losses': df['train_loss'].tolist(),
                        'val_losses': df['val_loss'].tolist()
                    }
                    print(f"Loaded loss curves for rank {rank}")
                else:
                    print(f"No loss curves CSV found for rank {rank}")
                    
            except (ValueError, IndexError, Exception) as e:
                print(f"Error processing {item}: {e}")
    
    return loss_data

def plot_loss_curves(loss_data: dict, args):
    """Plot loss curves for all ranks."""
    if not loss_data:
        print("No loss data to plot!")
        return
    
    output_dir = args.output_dir if args.output_dir else args.results_dir
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=args.figsize)
    
    # Color map for different ranks
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(loss_data), 10)))
    if len(loss_data) > 10:
        colors = plt.cm.viridis(np.linspace(0, 1, len(loss_data)))
    
    # Plot training losses
    for i, (rank, data) in enumerate(sorted(loss_data.items())):
        ax1.plot(data['epochs'], data['train_losses'], 
                label=f'Rank {rank}', color=colors[i], linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss vs Epoch')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    if args.log_scale:
        ax1.set_yscale('log')
    
    # Plot validation losses
    for i, (rank, data) in enumerate(sorted(loss_data.items())):
        ax2.plot(data['epochs'], data['val_losses'], 
                label=f'Rank {rank}', color=colors[i], linewidth=2)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Validation Loss vs Epoch')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    if args.log_scale:
        ax2.set_yscale('log')
    
    plt.suptitle(args.title, fontsize=16)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f'{args.save_prefix}_loss_curves_replot.png')
    plt.savefig(plot_path, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")
    
    plt.show()
    
    # Create final loss comparison plot
    fig2, ax3 = plt.subplots(1, 1, figsize=(10, 6))
    
    ranks_list = sorted(loss_data.keys())
    final_train_losses = [loss_data[rank]['train_losses'][-1] for rank in ranks_list]
    final_val_losses = [loss_data[rank]['val_losses'][-1] for rank in ranks_list]
    
    x_pos = np.arange(len(ranks_list))
    width = 0.35
    
    ax3.bar(x_pos - width/2, final_train_losses, width, label='Final Training Loss', alpha=0.8)
    ax3.bar(x_pos + width/2, final_val_losses, width, label='Final Validation Loss', alpha=0.8)
    
    ax3.set_xlabel('Rank')
    ax3.set_ylabel('Final Loss')
    ax3.set_title('Final Loss vs Rank')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(ranks_list)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    if args.log_scale:
        ax3.set_yscale('log')
    
    plt.tight_layout()
    
    # Save final loss plot
    final_loss_path = os.path.join(output_dir, f'{args.save_prefix}_final_losses_replot.png')
    plt.savefig(final_loss_path, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved plot: {final_loss_path}")
    
    plt.show()

def print_summary(loss_data: dict):
    """Print a summary of the results."""
    if not loss_data:
        print("No data to summarize!")
        return
    
    print(f"\n{'='*80}")
    print("RANK SWEEP RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Final Train':<12} {'Final Val':<12} {'Min Train':<12} {'Min Val':<12} {'Epochs':<8}")
    print(f"{'='*80}")
    
    for rank in sorted(loss_data.keys()):
        data = loss_data[rank]
        final_train = data['train_losses'][-1]
        final_val = data['val_losses'][-1]
        min_train = min(data['train_losses'])
        min_val = min(data['val_losses'])
        num_epochs = len(data['epochs'])
        
        print(f"{rank:<6} {final_train:<12.6f} {final_val:<12.6f} {min_train:<12.6f} {min_val:<12.6f} {num_epochs:<8}")
    
    print(f"{'='*80}")

def main():
    args = parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Results directory not found: {args.results_dir}")
        return
    
    print(f"Loading results from: {args.results_dir}")
    
    # Load loss curves
    loss_data = load_all_loss_curves(args.results_dir)
    
    if not loss_data:
        print("No loss curve data found!")
        return
    
    print(f"Found results for ranks: {sorted(loss_data.keys())}")
    
    # Plot results
    plot_loss_curves(loss_data, args)
    
    # Print summary
    print_summary(loss_data)
    
    print(f"\nRe-plotting completed!")

if __name__ == "__main__":
    main()
