#!/usr/bin/env python3
"""
Example usage of KSLinearTriton module.

This script demonstrates how to use the KSLinearTriton module for 
creating and using Kronecker-sparse linear layers with Triton kernels.
"""

import torch
from ksmm_triton.ksmm_module import KSLinearTriton, create_simple_ks_layer


def example_single_layer():
    """Example with a single Kronecker-sparse layer."""
    print("=== Single Layer Example ===")
    
    # Define pattern (a, b, c, d) = (6, 64, 64, 1)
    pattern = (6, 64, 64, 1)
    a, b, c, d = pattern
    
    # Calculate dimensions
    in_features = a * c * d   # 6 * 64 * 1 = 384
    out_features = a * b * d  # 6 * 64 * 1 = 384
    batch_size = 32
    
    print(f"Pattern: {pattern}")
    print(f"Input features: {in_features}")
    print(f"Output features: {out_features}")
    print(f"Batch size: {batch_size}")
    
    # Create the layer
    ks_layer = create_simple_ks_layer(
        in_features=in_features,
        out_features=out_features, 
        pattern=pattern,
        dtype=torch.float16,
        bs_last=False,  # Use batch-size-first layout
        device='cuda'
    )
    
    print(f"Created layer: {ks_layer}")
    print(f"Weight parameters: {ks_layer.get_weights_size()}")
    print(f"Theoretical speedup: {ks_layer.get_theoretical_speedup():.2f}x")
    
    # Create input tensor
    x = torch.randn(batch_size, in_features, dtype=torch.float16, device='cuda')
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = ks_layer(x)
    print(f"Output shape: {output.shape}")
    
    # Test backward pass
    loss = output.sum()
    loss.backward()
    print("Backward pass successful!")
    print()


def example_chain():
    """Example with a chain of Kronecker-sparse layers."""
    print("=== Chain Example ===")
    
    # Define a chain of patterns
    patterns = [
        (6, 64, 64, 1),   # Layer 1: 384 -> 384
        (6, 32, 64, 1),   # Layer 2: 384 -> 192  
        (6, 16, 32, 1),   # Layer 3: 192 -> 96
    ]
    
    batch_size = 32
    
    print(f"Chain patterns: {patterns}")
    
    # Create the chain layer
    ks_chain = KSLinearTriton(
        patterns=patterns,
        weights=None,  # Random initialization
        dtype=torch.float16,
        bs_last=False,
        device='cuda'
    )
    
    print(f"Created chain: {ks_chain}")
    print(f"Input features: {ks_chain.in_features}")
    print(f"Output features: {ks_chain.out_features}")
    print(f"Total weight parameters: {ks_chain.get_weights_size()}")
    print(f"Theoretical speedup: {ks_chain.get_theoretical_speedup():.2f}x")
    
    # Create input tensor
    x = torch.randn(batch_size, ks_chain.in_features, dtype=torch.float16, device='cuda')
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = ks_chain(x)
    print(f"Output shape: {output.shape}")
    
    # Test backward pass
    loss = output.sum()
    loss.backward()
    print("Backward pass successful!")
    print()


def example_batch_size_last():
    """Example using batch-size-last layout."""
    print("=== Batch-Size-Last Example ===")
    
    pattern = (4, 32, 32, 2)
    a, b, c, d = pattern
    
    in_features = a * c * d   # 4 * 32 * 2 = 256
    out_features = a * b * d  # 4 * 32 * 2 = 256
    batch_size = 64
    
    print(f"Pattern: {pattern}")
    print(f"Using BSL (batch-size-last) layout")
    
    # Create layer with batch-size-last layout
    ks_layer = KSLinearTriton(
        patterns=[pattern],
        dtype=torch.float16,
        bs_last=True,  # Use batch-size-last layout
        device='cuda'
    )
    
    # Create input tensor in BSL format: (features, batch)
    x = torch.randn(in_features, batch_size, dtype=torch.float16, device='cuda')
    print(f"Input shape (BSL): {x.shape}")
    
    # Forward pass
    output = ks_layer(x)
    print(f"Output shape (BSL): {output.shape}")
    
    # Test backward pass
    loss = output.sum()
    loss.backward()
    print("Backward pass successful!")
    print()


def example_custom_weights():
    """Example with custom weight initialization."""
    print("=== Custom Weights Example ===")
    
    pattern = (2, 16, 16, 1)
    a, b, c, d = pattern
    
    # Create custom weight tensor in (a, b, c, d) format
    custom_weight = torch.randn(a, b, c, d, dtype=torch.float16, device='cuda') * 0.1
    
    print(f"Custom weight shape: {custom_weight.shape}")
    
    # Create layer with custom weights
    ks_layer = KSLinearTriton(
        patterns=[pattern],
        weights=[custom_weight],  # Provide custom weights
        dtype=torch.float16,
        device='cuda'
    )
    
    print("Layer created with custom weights")
    
    # Test forward pass
    in_features = a * c * d
    batch_size = 16
    x = torch.randn(batch_size, in_features, dtype=torch.float16, device='cuda')
    
    output = ks_layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print()


def example_comparison_with_dense():
    """Compare performance and accuracy with dense linear layer."""
    print("=== Comparison with Dense Layer ===")
    
    pattern = (8, 32, 32, 1)
    a, b, c, d = pattern
    
    in_features = a * c * d   # 8 * 32 * 1 = 256
    out_features = a * b * d  # 8 * 32 * 1 = 256
    batch_size = 128
    
    # Create KS layer
    ks_layer = KSLinearTriton(
        patterns=[pattern],
        dtype=torch.float16,
        device='cuda'
    )
    
    # Create equivalent dense layer
    dense_layer = torch.nn.Linear(
        in_features, out_features, 
        dtype=torch.float16, device='cuda'
    )
    
    # Get dense representation of KS layer for comparison
    ks_dense_weight = ks_layer.get_dense_product()
    
    print(f"KS layer parameters: {ks_layer.get_weights_size()}")
    print(f"Dense layer parameters: {dense_layer.weight.numel()}")
    print(f"Parameter reduction: {ks_layer.get_theoretical_speedup():.2f}x")
    
    # Test with same input
    x = torch.randn(batch_size, in_features, dtype=torch.float16, device='cuda')
    
    # Forward passes
    ks_output = ks_layer(x)
    dense_output = dense_layer(x)
    
    print(f"KS output shape: {ks_output.shape}")
    print(f"Dense output shape: {dense_output.shape}")
    print(f"Output difference (should be large): {torch.norm(ks_output - dense_output):.4f}")
    print()


if __name__ == "__main__":
    print("KSLinearTriton Module Examples")
    print("=" * 50)
    
    try:
        example_single_layer()
        example_chain()
        example_batch_size_last()
        example_custom_weights()
        example_comparison_with_dense()
        
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
