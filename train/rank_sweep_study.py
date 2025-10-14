#!/usr/bin/env python3
"""
Rank Sweep Study Script

This script runs training with different rank values and plots the loss curves
to compare how rank affects training convergence.
"""

import os
import subprocess
import json
import csv
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
import pandas as pd

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run rank sweep study for KS Linear Chain')
    
    # Required arguments
    parser.add_argument('--weight_path', type=str, required=True,
                        help='Path to the .pth file containing the dense weight matrix')
    
    # Rank sweep parameters
    parser.add_argument('--ranks', type=int, nargs='+', default=[4, 8, 16, 32, 64, 128],
                        help='List of ranks to test (default: [2, 4, 8, 16, 32, 64, 128, 256, 512])')
    parser.add_argument('--max_rank', type=int, default=None,
                        help='Maximum rank to test (will filter ranks list)')
    parser.add_argument('--block_sizes', type=int, nargs='+', default=[2],
                        help='List of block sizes to test (default: [1, 2, 4, 8])')
    parser.add_argument('--max_block_size', type=int, default=None,
                        help='Maximum block size to test (will filter block_sizes list)')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs for each rank (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Number of random samples for training (default: 10000)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./rank_sweep_results',
                        help='Directory to save results (default: ./rank_sweep_results)')
    parser.add_argument('--save_prefix', type=str, default='rank_sweep',
                        help='Prefix for saved files (default: rank_sweep)')
    
    # Training script parameters
    parser.add_argument('--impl', type=str, default='bmm', choices=['triton', 'bmm'],
                        help='Implementation choice (default: bmm)')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'sgd', 'adamw'],
                        help='Optimizer type (default: adamw)')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        help='Learning rate scheduler type (default: plateau)')
    
    # Device and precision
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training (default: cuda)')
    parser.add_argument('--dtype', type=str, default='float32', choices=['float16', 'float32'],
                        help='Data type for training (default: float32)')
    
    # Plot parameters
    parser.add_argument('--plot_title', type=str, default='Rank vs Training Loss Convergence',
                        help='Title for the loss curve plot')
    parser.add_argument('--log_scale', action='store_true',
                        help='Use log scale for loss axis')
    parser.add_argument('--save_plots', action='store_true', default=True,
                        help='Save plots to files')
    
    return parser.parse_args()


def run_training_for_rank_and_block_size(rank: int, block_size: int, args, rank_block_output_dir: str):
    """Run training for a specific rank and block size combination."""
    print(f"\n{'='*60}")
    print(f"Training with rank {rank}, block_size {block_size}")
    print(f"{'='*60}")
    
    # Construct command for training script
    cmd = [
        'python', 'train_with_random.py',
        '--weight_path', args.weight_path,
        '--rank', str(rank),
        '--block_size', str(block_size),
        # '--num_epochs', str(args.num_epochs),
        # '--batch_size', str(args.batch_size),
        # '--lr', str(args.lr),
        # '--num_samples', str(args.num_samples),
        '--output_dir', rank_block_output_dir,
        '--save_prefix', f'rank_{rank}_block_{block_size}',
        '--impl', args.impl,
        # '--optimizer', args.optimizer,
        # '--scheduler', args.scheduler,
        '--device', args.device,
        '--dtype', args.dtype,
        '--save_interval', '999999',  # Only save final model to save space
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the training script with live output but also capture for error handling
        print(f"Starting training process...")
        process = subprocess.Popen(cmd, cwd=os.path.dirname(os.path.abspath(__file__)),
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 text=True, bufsize=1, universal_newlines=True)
        
        # Print output in real-time
        for line in process.stdout:
            print(line.rstrip())
        
        # Wait for process to complete and check return code
        return_code = process.wait()
        
        if return_code == 0:
            print(f"Training completed successfully for rank {rank}, block_size {block_size}")
            return True
        else:
            print(f"Training failed for rank {rank}, block_size {block_size} with return code {return_code}")
            return False
            
    except Exception as e:
        print(f"Training failed for rank {rank}, block_size {block_size}!")
        print(f"Error: {e}")
        return False


def load_loss_curves(output_dir: str, rank_block_combinations: list):
    """Load loss curves from all rank and block size experiments."""
    loss_data = {}
    
    for rank, block_size in rank_block_combinations:
        rank_block_dir = os.path.join(output_dir, f'rank_{rank}_block_{block_size}')
        csv_file = os.path.join(rank_block_dir, f'rank_{rank}_block_{block_size}_rank_{rank}_loss_curves.csv')
        
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                loss_data[(rank, block_size)] = {
                    'epochs': df['epoch'].tolist(),
                    'train_losses': df['train_loss'].tolist(),
                    'val_losses': df['val_loss'].tolist()
                }
                print(f"Loaded loss curves for rank {rank}, block_size {block_size}")
            except Exception as e:
                print(f"Failed to load loss curves for rank {rank}, block_size {block_size}: {e}")
        else:
            print(f"Loss curves file not found for rank {rank}, block_size {block_size}: {csv_file}")
    
    return loss_data


def plot_loss_curves(loss_data: dict, args, output_dir: str):
    """Plot loss curves for all rank and block size combinations."""
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Color map for different combinations
    colors = plt.cm.tab10(np.linspace(0, 1, len(loss_data)))
    
    # Plot training losses
    for i, ((rank, block_size), data) in enumerate(sorted(loss_data.items())):
        ax1.plot(data['epochs'], data['train_losses'], 
                label=f'Rank {rank}, Block {block_size}', color=colors[i], linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss vs Epoch')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    if args.log_scale:
        ax1.set_yscale('log')
    
    # Plot validation losses
    for i, ((rank, block_size), data) in enumerate(sorted(loss_data.items())):
        ax2.plot(data['epochs'], data['val_losses'], 
                label=f'Rank {rank}, Block {block_size}', color=colors[i], linewidth=2)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Validation Loss vs Epoch')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    if args.log_scale:
        ax2.set_yscale('log')
    
    plt.suptitle(args.plot_title, fontsize=16)
    plt.tight_layout()
    
    if args.save_plots:
        plot_path = os.path.join(output_dir, f'{args.save_prefix}_loss_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {plot_path}")
    
    plt.show()
    
    # Create final loss comparison plot
    fig2, ax3 = plt.subplots(1, 1, figsize=(12, 8))
    
    combinations_list = sorted(loss_data.keys())
    final_train_losses = [loss_data[combo]['train_losses'][-1] for combo in combinations_list]
    final_val_losses = [loss_data[combo]['val_losses'][-1] for combo in combinations_list]
    
    x_pos = np.arange(len(combinations_list))
    width = 0.35
    
    ax3.bar(x_pos - width/2, final_train_losses, width, label='Final Training Loss', alpha=0.8)
    ax3.bar(x_pos + width/2, final_val_losses, width, label='Final Validation Loss', alpha=0.8)
    
    ax3.set_xlabel('Rank, Block Size')
    ax3.set_ylabel('Final Loss')
    ax3.set_title('Final Loss vs Rank and Block Size')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'R{rank}, B{block_size}' for rank, block_size in combinations_list], rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    if args.log_scale:
        ax3.set_yscale('log')
    
    plt.tight_layout()
    
    if args.save_plots:
        final_loss_path = os.path.join(output_dir, f'{args.save_prefix}_final_losses.png')
        plt.savefig(final_loss_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {final_loss_path}")
    
    plt.show()


def save_summary_results(loss_data: dict, args, output_dir: str):
    """Save summary results of the rank and block size sweep study."""
    
    summary_data = []
    
    for (rank, block_size) in sorted(loss_data.keys()):
        data = loss_data[(rank, block_size)]
        final_train_loss = data['train_losses'][-1]
        final_val_loss = data['val_losses'][-1]
        min_train_loss = min(data['train_losses'])
        min_val_loss = min(data['val_losses'])
        
        # Find epochs where minimum losses occurred
        min_train_epoch = data['epochs'][data['train_losses'].index(min_train_loss)]
        min_val_epoch = data['epochs'][data['val_losses'].index(min_val_loss)]
        
        summary_data.append({
            'rank': rank,
            'block_size': block_size,
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'min_train_loss': min_train_loss,
            'min_val_loss': min_val_loss,
            'min_train_epoch': min_train_epoch,
            'min_val_epoch': min_val_epoch,
            'num_epochs': len(data['epochs'])
        })
    
    # Save to CSV
    summary_csv = os.path.join(output_dir, f'{args.save_prefix}_summary.csv')
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(summary_csv, index=False)
    print(f"Saved summary: {summary_csv}")
    
    # Save to JSON
    summary_json = os.path.join(output_dir, f'{args.save_prefix}_summary.json')
    with open(summary_json, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved summary: {summary_json}")
    
    # Print summary table
    print(f"\n{'='*90}")
    print("RANK AND BLOCK SIZE SWEEP STUDY SUMMARY")
    print(f"{'='*90}")
    print(f"{'Rank':<6} {'Block':<6} {'Final Train':<12} {'Final Val':<12} {'Min Train':<12} {'Min Val':<12}")
    print(f"{'='*90}")
    
    for data_point in summary_data:
        print(f"{data_point['rank']:<6} "
              f"{data_point['block_size']:<6} "
              f"{data_point['final_train_loss']:<12.6f} "
              f"{data_point['final_val_loss']:<12.6f} "
              f"{data_point['min_train_loss']:<12.6f} "
              f"{data_point['min_val_loss']:<12.6f}")
    
    return summary_data


def main():
    args = parse_args()
    
    # Filter ranks if max_rank is specified
    ranks = args.ranks
    if args.max_rank is not None:
        ranks = [r for r in ranks if r <= args.max_rank]
    
    # Filter block_sizes if max_block_size is specified  
    block_sizes = args.block_sizes
    if args.max_block_size is not None:
        block_sizes = [bs for bs in block_sizes if bs <= args.max_block_size]
    
    print(f"Starting rank and block size sweep study")
    print(f"Ranks: {ranks}")
    print(f"Block sizes: {block_sizes}")
    print(f"Weight path: {args.weight_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of epochs per combination: {args.num_epochs}")
    
    # Create main output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run training for each rank and block size combination
    successful_combinations = []
    
    for rank in ranks:
        for block_size in block_sizes:
            rank_block_output_dir = os.path.join(args.output_dir, f'rank_{rank}_block_{block_size}')
            os.makedirs(rank_block_output_dir, exist_ok=True)
            
            success = run_training_for_rank_and_block_size(rank, block_size, args, rank_block_output_dir)
            if success:
                successful_combinations.append((rank, block_size))
            else:
                print(f"Skipping rank {rank}, block_size {block_size} due to training failure")
    
    if not successful_combinations:
        print("No successful training runs. Exiting.")
        return
    
    print(f"\nSuccessful combinations: {successful_combinations}")
    
    # Load loss curves
    print(f"\nLoading loss curves...")
    loss_data = load_loss_curves(args.output_dir, successful_combinations)
    
    if not loss_data:
        print("No loss curve data found. Exiting.")
        return
    
    # Plot loss curves
    print(f"\nCreating plots...")
    plot_loss_curves(loss_data, args, args.output_dir)
    
    # Save summary results
    print(f"\nSaving summary results...")
    summary_data = save_summary_results(loss_data, args, args.output_dir)
    
    print(f"\nRank and block size sweep study completed!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
