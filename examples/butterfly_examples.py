#!/usr/bin/env python3
"""
Examples of using butterfly decomposition with KSLinearTriton.

This script demonstrates how to use the butterfly pattern generation
and butterfly chain creation functions.
"""

import torch
from ksmm_triton import create_butterfly_patterns, create_butterfly_chain, KSLinearTriton


def example_butterfly_patterns():
    """Demonstrate butterfly pattern generation."""
    print("=== Butterfly Pattern Generation ===")
    
    for n in [2, 3, 4]:
        print(f"\nButterfly patterns for n={n} (matrix size 2^{n} = {2**n}):")
        patterns = create_butterfly_patterns(n)
        
        total_params = 0
        for i, (a, b, c, d) in enumerate(patterns):
            params = a * b * c * d
            total_params += params
            input_dim = a * c * d
            output_dim = a * b * d
            print(f"  Pattern {i}: (a={a}, b={b}, c={c}, d={d})")
            print(f"    Input: {input_dim}, Output: {output_dim}, Params: {params}")
        
        matrix_size = 2 ** n
        dense_params = matrix_size * matrix_size
        compression_ratio = dense_params / total_params
        
        print(f"  Total butterfly params: {total_params}")
        print(f"  Dense matrix params: {dense_params}")
        print(f"  Compression ratio: {compression_ratio:.2f}x")


def example_butterfly_chains():
    """Demonstrate butterfly chain creation for different matrix sizes."""
    print("\n=== Butterfly Chain Creation ===")
    
    test_shapes = [
        [64, 64],    # Square power-of-2
        [128, 64],   # Rectangular, larger output
        [64, 128],   # Rectangular, larger input  
        [96, 48],    # Non-power-of-2
        [100, 50],   # Non-power-of-2
    ]
    
    for shape in test_shapes:
        out_features, in_features = shape
        print(f"\nCreating butterfly chain for {out_features}x{in_features} matrix:")
        
        try:
            chain = create_butterfly_chain(
                shape=shape,
                dtype=torch.float16,
                device='cuda',
                bias=False  # Disable bias for cleaner analysis
            )
            
            print(f"  Successfully created chain with {len(chain.patterns)} patterns")
            print(f"  Patterns: {chain.patterns}")
            
            # Calculate compression
            total_params = chain.get_weights_size()
            dense_params = out_features * in_features
            compression_ratio = dense_params / total_params if total_params > 0 else float('inf')
            
            print(f"  Butterfly params: {total_params}")
            print(f"  Dense params: {dense_params}")
            print(f"  Compression ratio: {compression_ratio:.2f}x")
            
            # Test functionality
            batch_size = 8
            x = torch.randn(batch_size, in_features, dtype=torch.float16, device='cuda')
            output = chain(x)
            
            expected_shape = (batch_size, out_features)
            assert output.shape == expected_shape, f"Shape mismatch: {output.shape} != {expected_shape}"
            print(f"  ✓ Forward pass successful: {x.shape} -> {output.shape}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")


def example_butterfly_vs_dense():
    """Compare butterfly decomposition with dense linear layer."""
    print("\n=== Butterfly vs Dense Comparison ===")
    
    # Test on a moderately sized matrix
    shape = [128, 64]
    out_features, in_features = shape
    batch_size = 32
    
    print(f"Comparing {out_features}x{in_features} transformation:")
    
    # Create butterfly chain
    butterfly_layer = create_butterfly_chain(
        shape=shape,
        dtype=torch.float16,
        device='cuda',
        bias=True
    )
    
    # Create dense layer
    dense_layer = torch.nn.Linear(
        in_features, out_features,
        dtype=torch.float16, device='cuda'
    )
    
    # Create test input
    x = torch.randn(batch_size, in_features, dtype=torch.float16, device='cuda')
    
    # Forward passes
    butterfly_output = butterfly_layer(x)
    dense_output = dense_layer(x)
    
    # Parameter comparison
    butterfly_params = butterfly_layer.get_weights_size()
    if butterfly_layer.bias is not None:
        butterfly_params += butterfly_layer.bias.numel()
    
    dense_params = dense_layer.weight.numel()
    if dense_layer.bias is not None:
        dense_params += dense_layer.bias.numel()
    
    compression_ratio = dense_params / butterfly_params
    
    print(f"  Butterfly parameters: {butterfly_params:,}")
    print(f"  Dense parameters: {dense_params:,}")
    print(f"  Parameter reduction: {compression_ratio:.2f}x")
    print(f"  Butterfly output shape: {butterfly_output.shape}")
    print(f"  Dense output shape: {dense_output.shape}")
    
    # Memory usage estimation (rough)
    butterfly_memory = butterfly_params * 2  # float16 = 2 bytes
    dense_memory = dense_params * 2
    memory_saving = (dense_memory - butterfly_memory) / dense_memory * 100
    
    print(f"  Memory usage - Butterfly: {butterfly_memory/1024:.1f} KB")
    print(f"  Memory usage - Dense: {dense_memory/1024:.1f} KB")
    print(f"  Memory saving: {memory_saving:.1f}%")


def example_butterfly_training():
    """Demonstrate training with butterfly decomposition."""
    print("\n=== Butterfly Training Example ===")
    
    # Create a simple model with butterfly layer
    shape = [64, 32]
    model = create_butterfly_chain(
        shape=shape,
        dtype=torch.float16,
        device='cuda'
    )
    
    # Setup for training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    
    # Generate dummy data
    batch_size = 16
    x = torch.randn(batch_size, 32, dtype=torch.float16, device='cuda')
    y_target = torch.randn(batch_size, 64, dtype=torch.float16, device='cuda')
    
    print(f"Model parameters: {model.get_weights_size():,}")
    print(f"Training data: {x.shape} -> {y_target.shape}")
    
    # Training steps
    initial_loss = None
    for step in range(5):
        optimizer.zero_grad()
        
        y_pred = model(x)
        loss = loss_fn(y_pred, y_target)
        
        if step == 0:
            initial_loss = loss.item()
        
        loss.backward()
        optimizer.step()
        
        print(f"  Step {step}: Loss = {loss.item():.6f}")
    
    final_loss = loss.item()
    print(f"  Loss reduction: {initial_loss:.6f} -> {final_loss:.6f}")
    print("  ✓ Training successful - gradients flow through butterfly decomposition")


def example_different_butterfly_sizes():
    """Show butterfly decomposition for various matrix sizes."""
    print("\n=== Butterfly Decomposition Analysis ===")
    
    # Test various sizes
    sizes = [
        (16, 16),    # Small square
        (64, 64),    # Medium square
        (256, 128),  # Large rectangular
        (128, 256),  # Large rectangular (flipped)
        (100, 200),  # Non-power-of-2
    ]
    
    for out_size, in_size in sizes:
        print(f"\nMatrix {out_size}x{in_size}:")
        
        try:
            chain = create_butterfly_chain([out_size, in_size], device='cuda')
            
            # Analysis
            num_patterns = len(chain.patterns)
            butterfly_params = chain.get_weights_size()
            dense_params = out_size * in_size
            efficiency = dense_params / butterfly_params
            
            print(f"  Patterns: {num_patterns}")
            print(f"  Butterfly params: {butterfly_params:,}")
            print(f"  Dense params: {dense_params:,}")
            print(f"  Efficiency: {efficiency:.2f}x compression")
            
            # Check if it's actually using butterfly structure
            has_butterfly = any(
                (a, b, c, d)[1] == 2 and (a, b, c, d)[2] == 2 
                for a, b, c, d in chain.patterns
            )
            
            if has_butterfly:
                print("  ✓ Contains butterfly patterns")
            else:
                print("  ⚠ No butterfly patterns (too small/irregular)")
                
        except Exception as e:
            print(f"  ❌ Failed: {e}")


if __name__ == "__main__":
    print("Butterfly Decomposition Examples")
    print("=" * 50)
    
    try:
        example_butterfly_patterns()
        example_butterfly_chains()
        example_butterfly_vs_dense()
        example_butterfly_training()
        example_different_butterfly_sizes()
        
        print("\n" + "=" * 50)
        print("✅ All butterfly examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()
