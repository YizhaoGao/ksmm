#!/usr/bin/env python3
"""
Generate a test weight matrix for rank sweep studies.

This script creates a dense weight matrix and saves it as a .pth file
for use in the rank sweep study.
"""

import torch
import argparse
import os

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate test weight matrix')
    
    parser.add_argument('--output_features', type=int, default=2048,
                        help='Number of output features (default: 2048)')
    parser.add_argument('--input_features', type=int, default=4096,
                        help='Number of input features (default: 4096)')
    parser.add_argument('--output_path', type=str, default='./test_weight.pth',
                        help='Output path for the weight matrix (default: ./test_weight.pth)')
    parser.add_argument('--dtype', type=str, default='float32', choices=['float16', 'float32'],
                        help='Data type for the weight matrix (default: float32)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--init_method', type=str, default='xavier_uniform', 
                        choices=['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'normal'],
                        help='Weight initialization method (default: xavier_uniform)')
    
    return parser.parse_args()

def create_weight_matrix(out_features: int, in_features: int, dtype: torch.dtype, 
                        init_method: str, seed: int = 42):
    """Create a weight matrix with specified initialization."""
    torch.manual_seed(seed)
    
    # Create weight matrix
    weight = torch.empty(out_features, in_features, dtype=dtype)
    
    # Initialize based on method
    if init_method == 'xavier_uniform':
        torch.nn.init.xavier_uniform_(weight)
    elif init_method == 'xavier_normal':
        torch.nn.init.xavier_normal_(weight)
    elif init_method == 'kaiming_uniform':
        torch.nn.init.kaiming_uniform_(weight)
    elif init_method == 'kaiming_normal':
        torch.nn.init.kaiming_normal_(weight)
    elif init_method == 'normal':
        torch.nn.init.normal_(weight, mean=0.0, std=0.1)
    else:
        raise ValueError(f"Unknown initialization method: {init_method}")
    
    return weight

def main():
    args = parse_args()
    
    # Set dtype
    dtype = torch.float16 if args.dtype == 'float16' else torch.float32
    
    print(f"Generating weight matrix:")
    print(f"  Shape: [{args.output_features}, {args.input_features}]")
    print(f"  Dtype: {dtype}")
    print(f"  Initialization: {args.init_method}")
    print(f"  Seed: {args.seed}")
    
    # Create weight matrix
    weight = create_weight_matrix(
        args.output_features, 
        args.input_features, 
        dtype, 
        args.init_method, 
        args.seed
    )
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    
    # Save weight matrix
    torch.save(weight, args.output_path)
    
    print(f"Weight matrix saved to: {args.output_path}")
    print(f"File size: {os.path.getsize(args.output_path) / 1024 / 1024:.2f} MB")
    
    # Print some statistics
    print(f"\nWeight statistics:")
    print(f"  Min: {weight.min().item():.6f}")
    print(f"  Max: {weight.max().item():.6f}")
    print(f"  Mean: {weight.mean().item():.6f}")
    print(f"  Std: {weight.std().item():.6f}")
    print(f"  Total parameters: {weight.numel():,}")

if __name__ == "__main__":
    main()
