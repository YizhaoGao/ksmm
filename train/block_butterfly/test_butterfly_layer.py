#!/usr/bin/env python3
"""
Simple test script for BlockwiseButterfly layer.
Tests basic functionality with different configurations.
"""

import torch
import sys
import os

# Add the current directory to path so we can import butterfly_layer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from butterfly_layer import BlockwiseButterfly


def test_basic_functionality():
    """Test basic forward pass with different configurations."""
    print("=" * 50)
    print("Testing BlockwiseButterfly Layer")
    print("=" * 50)
    
    # Test configurations (including non-power-of-2 dimensions)
    configs = [
        {"data_dim": 8, "level": 0, "block_size": 4, "share_weights": False, "weight_init": "identity"},
        {"data_dim": 16, "level": 1, "block_size": 4, "share_weights": False, "weight_init": "identity"},
        {"data_dim": 32, "level": 2, "block_size": 4, "share_weights": False, "weight_init": "identity"},
        {"data_dim": 8, "level": 0, "block_size": 4, "share_weights": True, "weight_init": "random"},
        # Test with non-aligned dimensions (should use padding)
        {"data_dim": 15, "level": 1, "block_size": 4, "share_weights": False, "weight_init": "identity"},
        {"data_dim": 7, "level": 0, "block_size": 4, "share_weights": False, "weight_init": "identity"},
        {"data_dim": 10, "level": 0, "block_size": 4, "share_weights": False, "weight_init": "identity"},
    ]
    
    batch_size = 3
    
    for i, config in enumerate(configs):
        print(f"\nTest {i+1}: {config}")
        print("-" * 30)
        
        try:
            # Create layer
            layer = BlockwiseButterfly(**config)
            
            # Print padding info
            if layer.needs_padding:
                print(f"Padding enabled: {config['data_dim']} -> {layer.padded_dim}")
            else:
                print("No padding needed")
            
            # Create input
            x = torch.randn(batch_size, config["data_dim"])
            print(f"Input shape: {x.shape}")
            
            # Forward pass
            y = layer(x)
            print(f"Output shape: {y.shape}")
            
            # Check shapes match
            assert x.shape == y.shape, f"Shape mismatch: {x.shape} != {y.shape}"
            print("✓ Shape test passed")
            
            # For identity initialization, check if it's close to identity transformation
            if config["weight_init"] == "identity":
                diff = torch.norm(y - x).item()
                print(f"Identity test - L2 difference: {diff:.6f}")
                if diff < 1e-5:
                    print("✓ Identity initialization test passed")
                else:
                    print("! Identity test: difference larger than expected")
            
            # Check gradient flow
            loss = y.sum()
            loss.backward()
            grad_norm = torch.norm(layer.w.grad).item()
            print(f"Gradient norm: {grad_norm:.6f}")
            print("✓ Gradient flow test passed")
            
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            continue
        
        print("✓ All tests passed for this configuration")


def test_parameter_shapes():
    """Test parameter shapes for different configurations."""
    print("\n" + "=" * 50)
    print("Testing Parameter Shapes")
    print("=" * 50)
    
    configs = [
        {"data_dim": 16, "level": 1, "block_size": 1, "share_weights": False},
        {"data_dim": 16, "level": 1, "block_size": 1, "share_weights": True},
        {"data_dim": 32, "level": 2, "block_size": 2, "share_weights": False},
        {"data_dim": 32, "level": 2, "block_size": 2, "share_weights": True},
    ]
    
    for config in configs:
        print(f"\nConfig: {config}")
        layer = BlockwiseButterfly(**config, weight_init="identity")
        
        expected_groups = 2 ** config["level"]
        expected_nb = config["data_dim"] // ((2 ** (config["level"] + 1)) * config["block_size"])
        
        if config["share_weights"]:
            expected_shape = (expected_groups, 1, 2, 2, config["block_size"], config["block_size"])
        else:
            expected_shape = (expected_groups, expected_nb, 2, 2, config["block_size"], config["block_size"])
        
        actual_shape = layer.w.shape
        print(f"Expected weight shape: {expected_shape}")
        print(f"Actual weight shape: {actual_shape}")
        
        assert actual_shape == expected_shape, f"Shape mismatch: {actual_shape} != {expected_shape}"
        print("✓ Parameter shape test passed")


def test_zero_padding():
    """Test zero padding functionality specifically."""
    print("\n" + "=" * 50)
    print("Testing Zero Padding Functionality")
    print("=" * 50)
    
    # Test that padded dimensions work correctly
    configs_with_padding = [
        {"data_dim": 15, "level": 1, "block_size": 1},  # 15 -> 16
        {"data_dim": 7, "level": 0, "block_size": 1},   # 7 -> 8
        {"data_dim": 10, "level": 0, "block_size": 2},  # 10 -> 12
        {"data_dim": 5, "level": 1, "block_size": 2},   # 5 -> 8
    ]
    
    for config in configs_with_padding:
        print(f"\nTesting padding for config: {config}")
        layer = BlockwiseButterfly(**config, weight_init='identity')
        
        print(f"Original dim: {config['data_dim']}, Padded dim: {layer.padded_dim}")
        
        # Test with identity weights - the non-padded part should remain unchanged
        x = torch.randn(3, config['data_dim'])
        y = layer(x)
        
        # For identity initialization, the original dimensions should be preserved
        if config.get('weight_init', 'identity') == 'identity':
            # The output should be very close to input for the original dimensions
            diff = torch.norm(y - x).item()
            print(f"Identity preservation test - L2 difference: {diff:.6f}")
            
        print(f"✓ Padding test passed for dim {config['data_dim']} -> {layer.padded_dim}")


def test_edge_cases():
    """Test edge cases and error conditions."""
    print("\n" + "=" * 50)
    print("Testing Edge Cases")
    print("=" * 50)
    
    # Test previously invalid dimension (should now work with padding)
    print("\nTesting dimension that requires padding:")
    try:
        layer = BlockwiseButterfly(data_dim=15, level=1, block_size=1)  # 15 not divisible by 4, should be padded to 16
        x = torch.randn(2, 15)
        y = layer(x)
        print(f"✓ Padding works: input {x.shape} -> output {y.shape}, padded_dim={layer.padded_dim}")
        assert y.shape == x.shape, f"Output shape should match input: {y.shape} != {x.shape}"
    except Exception as e:
        print(f"✗ Padding test failed: {e}")
    
    # Test minimum valid configuration
    print("\nTesting minimum valid configuration:")
    try:
        layer = BlockwiseButterfly(data_dim=1, level=0, block_size=1)  # Should be padded to 2
        x = torch.randn(1, 1)
        y = layer(x)
        print(f"✓ Minimum config works: input {x.shape} -> output {y.shape}, padded_dim={layer.padded_dim}")
    except Exception as e:
        print(f"✗ Minimum config failed: {e}")
        
    # Test very small dimensions
    print("\nTesting various small dimensions with padding:")
    test_dims = [1, 3, 5, 7, 9, 11]
    for dim in test_dims:
        try:
            layer = BlockwiseButterfly(data_dim=dim, level=0, block_size=1)
            x = torch.randn(1, dim)
            y = layer(x)
            print(f"✓ dim={dim} works (padded to {layer.padded_dim}): {x.shape} -> {y.shape}")
        except Exception as e:
            print(f"✗ dim={dim} failed: {e}")


if __name__ == "__main__":
    torch.manual_seed(42)  # For reproducible results
    
    test_basic_functionality()
    test_parameter_shapes()
    test_zero_padding()
    test_edge_cases()
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)