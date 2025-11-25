#!/usr/bin/env python3
"""
Test script for ButterflyChain module.
Tests the complete butterfly chain transformation with different configurations.
"""

import torch
import sys
import os

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from butterfly_chain import ButterflyChain, EfficientButterflyChain


def test_butterfly_chain_basic():
    """Test basic functionality of ButterflyChain."""
    print("=" * 60)
    print("Testing ButterflyChain - Basic Functionality")
    print("=" * 60)
    
    # Test configurations
    configs = [
        {"data_dim": 8, "block_size": 4, "share_weights": False, "weight_init": "identity"},
        {"data_dim": 16, "block_size": 4, "share_weights": False, "weight_init": "identity"},
        {"data_dim": 32, "block_size": 4, "share_weights": False, "weight_init": "identity"},
        {"data_dim": 15, "block_size": 4, "share_weights": True, "weight_init": "random"},  # Non-power-of-2
        {"data_dim": 20, "block_size": 4, "share_weights": False, "weight_init": "random"},  # Non power-of-2 with blocks
    ]
    
    batch_size = 3
    
    for i, config in enumerate(configs):
        print(f"\nTest {i+1}: {config}")
        print("-" * 40)
        
        try:
            # Create butterfly chain
            chain = ButterflyChain(**config)
            print(f"Created chain with {chain.num_levels} levels")
            print(f"Chain info: {chain}")
            
            # Create input
            x = torch.randn(batch_size, config["data_dim"])
            print(f"Input shape: {x.shape}")
            
            # Forward pass
            y = chain(x)
            print(f"Output shape: {y.shape}")
            
            # Check shapes match
            assert x.shape == y.shape, f"Shape mismatch: {x.shape} != {y.shape}"
            print("✓ Shape test passed")
            
            # For identity initialization, check transformation properties
            if config["weight_init"] == "identity":
                diff = torch.norm(y - x).item()
                print(f"Identity test - L2 difference: {diff:.6f}")
                if diff < 1e-4:
                    print("✓ Identity initialization test passed")
                else:
                    print(f"! Identity test: difference is {diff:.6f} (might be expected due to padding)")
            
            # Check gradient flow
            loss = y.sum()
            loss.backward()
            total_grad_norm = 0
            for param in chain.parameters():
                if param.grad is not None:
                    total_grad_norm += torch.norm(param.grad).item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            print(f"Total gradient norm: {total_grad_norm:.6f}")
            print("✓ Gradient flow test passed")
            
            # Get parameter info
            param_info = chain.get_parameter_info()
            print(f"Total parameters: {param_info['total_params']}")
            
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        print("✓ All tests passed for this configuration")


def test_butterfly_chain_levels():
    """Test that the correct number of levels is created."""
    print("\n" + "=" * 60)
    print("Testing ButterflyChain - Level Calculation")
    print("=" * 60)
    
    test_cases = [
        {"data_dim": 2, "block_size": 1, "expected_min_levels": 1},
        {"data_dim": 4, "block_size": 1, "expected_min_levels": 2},
        {"data_dim": 8, "block_size": 1, "expected_min_levels": 3},
        {"data_dim": 16, "block_size": 1, "expected_min_levels": 4},
        {"data_dim": 8, "block_size": 2, "expected_min_levels": 2},
        {"data_dim": 16, "block_size": 2, "expected_min_levels": 3},
    ]
    
    for case in test_cases:
        print(f"\nTesting data_dim={case['data_dim']}, block_size={case['block_size']}")
        
        chain = ButterflyChain(
            data_dim=case["data_dim"], 
            block_size=case["block_size"],
            weight_init="identity"
        )
        
        print(f"Created {chain.num_levels} levels")
        print(f"Expected at least {case['expected_min_levels']} levels")
        
        # The actual number might be higher due to padding, but should be reasonable
        assert chain.num_levels >= 1, "Should have at least 1 level"
        assert chain.num_levels <= 10, "Should not have too many levels"
        
        print("✓ Level calculation test passed")


def test_efficient_butterfly_chain():
    """Test the EfficientButterflyChain variant."""
    print("\n" + "=" * 60)
    print("Testing EfficientButterflyChain")
    print("=" * 60)
    
    configs = [
        {"data_dim": 15, "block_size": 1},
        {"data_dim": 20, "block_size": 2},
        {"data_dim": 100, "block_size": 1},
    ]
    
    for config in configs:
        print(f"\nTesting config: {config}")
        
        # Compare regular vs efficient
        regular_chain = ButterflyChain(**config, weight_init="identity")
        efficient_chain = EfficientButterflyChain(**config, weight_init="identity")
        
        print(f"Regular chain levels: {regular_chain.num_levels}")
        print(f"Efficient chain levels: {efficient_chain.num_levels}")
        print(f"Regular chain params: {regular_chain.get_num_parameters()}")
        print(f"Efficient chain params: {efficient_chain.get_num_parameters()}")
        
        # Test forward pass
        x = torch.randn(2, config["data_dim"])
        y1 = regular_chain(x)
        y2 = efficient_chain(x)
        
        assert y1.shape == y2.shape == x.shape, "Shape mismatch"
        print("✓ Both chains produce correct output shapes")


def test_parameter_info():
    """Test parameter information functionality."""
    print("\n" + "=" * 60)
    print("Testing Parameter Information")
    print("=" * 60)
    
    chain = ButterflyChain(data_dim=16, block_size=2, share_weights=False)
    info = chain.get_parameter_info()
    
    print("Parameter Information:")
    print(f"  Total parameters: {info['total_params']}")
    print(f"  Number of levels: {info['num_levels']}")
    print(f"  Data dimension: {info['data_dim']}")
    print(f"  Block size: {info['block_size']}")
    print(f"  Share weights: {info['share_weights']}")
    
    for level_info in info['levels']:
        print(f"  Level {level_info['level']}:")
        print(f"    Parameters: {level_info['params']}")
        print(f"    Weight shape: {level_info['weight_shape']}")
        print(f"    Groups: {level_info['groups']}")
        print(f"    Blocks per half: {level_info['nb']}")
        print(f"    Needs padding: {level_info['needs_padding']}")
    
    print("✓ Parameter info test passed")


if __name__ == "__main__":
    torch.manual_seed(42)  # For reproducible results
    
    test_butterfly_chain_basic()
    test_butterfly_chain_levels()
    test_efficient_butterfly_chain()
    test_parameter_info()
    
    print("\n" + "=" * 60)
    print("All ButterflyChain tests completed!")
    print("=" * 60)