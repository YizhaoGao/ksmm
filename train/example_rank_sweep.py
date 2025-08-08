#!/usr/bin/env python3
"""
Example usage of the rank sweep study scripts.

This script demonstrates how to:
1. Generate a test weight matrix
2. Run the rank sweep study
3. Analyze the results
"""

import subprocess
import os

def run_command(cmd, description):
    """Run a command and print its output."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("SUCCESS!")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("FAILED!")
        print(f"Error: {e}")
        if e.stderr:
            print("Error output:")
            print(e.stderr)
        return False

def main():
    print("Rank Sweep Study Example")
    print("This will demonstrate the complete workflow")
    
    # Step 1: Generate a test weight matrix
    print("\nStep 1: Generating test weight matrix...")
    weight_path = "./test_weight.pth"
    
    cmd1 = [
        'python', 'generate_test_weight.py',
        '--output_features', '1024',
        '--input_features', '2048', 
        '--output_path', weight_path,
        '--dtype', 'float32',
        '--init_method', 'xavier_uniform'
    ]
    
    if not run_command(cmd1, "Generating test weight matrix"):
        print("Failed to generate test weight. Exiting.")
        return
    
    # Step 2: Run rank sweep study
    print("\nStep 2: Running rank sweep study...")
    
    cmd2 = [
        'python', 'rank_sweep_study.py',
        '--weight_path', weight_path,
        '--ranks', '2', '4', '8', '16', '32', '64',  # Smaller ranks for demo
        '--num_epochs', '20',  # Fewer epochs for demo
        '--batch_size', '32',
        '--lr', '1e-3',
        '--num_samples', '5000',  # Fewer samples for demo
        '--output_dir', './demo_rank_sweep_results',
        '--save_prefix', 'demo',
        '--impl', 'bmm',
        '--log_scale'
    ]
    
    if not run_command(cmd2, "Running rank sweep study"):
        print("Rank sweep study failed.")
        return
    
    print("\n" + "="*60)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Check the following files:")
    print("1. test_weight.pth - Generated test weight matrix")
    print("2. ./demo_rank_sweep_results/ - Rank sweep results")
    print("   - Individual rank training results in rank_X/ subdirectories")
    print("   - demo_loss_curves.png - Loss curves comparison plot")
    print("   - demo_final_losses.png - Final loss comparison plot")
    print("   - demo_summary.csv - Summary of results")
    print("   - demo_summary.json - Summary in JSON format")
    
    # Show how to analyze results programmatically
    print("\nTo analyze results programmatically, you can:")
    print("import pandas as pd")
    print("df = pd.read_csv('./demo_rank_sweep_results/demo_summary.csv')")
    print("print(df)")

if __name__ == "__main__":
    main()
