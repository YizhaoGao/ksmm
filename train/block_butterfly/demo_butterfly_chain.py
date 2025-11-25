#!/usr/bin/env python3
"""
Simple demo script for ButterflyChain module.
Shows basic usage and key features.
"""

import torch
import sys
import os

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from butterfly_chain import ButterflyChain, EfficientButterflyChain


def demo_basic_usage():
    """Demonstrate basic usage of ButterflyChain."""
    print("=" * 50)
    print("ButterflyChain Basic Usage Demo")
    print("=" * 50)
    
    # Create a butterfly chain for dimension 16
    data_dim = 16
    chain = ButterflyChain(
        data_dim=data_dim,
        block_size=1,
        share_weights=False,
        weight_init='identity'
    )
    
    print(f"Created ButterflyChain: {chain}")
    print(f"Number of levels: {chain.num_levels}")
    print(f"Total parameters: {chain.get_num_parameters()}")
    
    # Create some input data
    batch_size = 4
    x = torch.randn(batch_size, data_dim)
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    y = chain(x)
    print(f"Output shape: {y.shape}")
    
    # Show parameter information
    info = chain.get_parameter_info()
    print(f"\nDetailed parameter info:")
    for i, level_info in enumerate(info['levels']):
        print(f"  Level {i}: {level_info['params']} params, "
              f"groups={level_info['groups']}, "
              f"weight_shape={level_info['weight_shape']}")


def demo_different_configurations():
    """Show different configuration options."""
    print("\n" + "=" * 50)
    print("Different Configuration Examples")
    print("=" * 50)
    
    configs = [
        {
            "name": "Standard 8D",
            "data_dim": 8,
            "block_size": 1,
            "share_weights": False,
            "weight_init": "identity"
        },
        {
            "name": "Block-wise 16D",
            "data_dim": 16,
            "block_size": 2,
            "share_weights": False,
            "weight_init": "identity"
        },
        {
            "name": "Weight-shared 12D",
            "data_dim": 12,
            "block_size": 1,
            "share_weights": True,
            "weight_init": "random"
        },
        {
            "name": "With Dropout",
            "data_dim": 8,
            "block_size": 1,
            "share_weights": False,
            "weight_init": "random",
            "dropout": 0.1
        }
    ]
    
    for config in configs:
        name = config.pop("name")
        print(f"\n{name}:")
        print(f"  Config: {config}")
        
        chain = ButterflyChain(**config)
        print(f"  Levels: {chain.num_levels}")
        print(f"  Parameters: {chain.get_num_parameters()}")
        
        # Test with sample input
        x = torch.randn(2, config["data_dim"])
        y = chain(x)
        print(f"  I/O: {x.shape} -> {y.shape}")


def demo_efficient_variant():
    """Demonstrate the efficient butterfly chain variant."""
    print("\n" + "=" * 50)
    print("Efficient ButterflyChain Demo")
    print("=" * 50)
    
    data_dim = 50  # Non-power-of-2 dimension
    
    # Compare regular vs efficient
    regular = ButterflyChain(data_dim=data_dim, block_size=1, weight_init="identity")
    efficient = EfficientButterflyChain(data_dim=data_dim, block_size=1, weight_init="identity")
    
    print(f"Regular ButterflyChain:")
    print(f"  Levels: {regular.num_levels}")
    print(f"  Parameters: {regular.get_num_parameters()}")
    
    print(f"\nEfficient ButterflyChain:")
    print(f"  Levels: {efficient.num_levels}")
    print(f"  Parameters: {efficient.get_num_parameters()}")
    
    # Test forward pass
    x = torch.randn(3, data_dim)
    y1 = regular(x)
    y2 = efficient(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Regular output: {y1.shape}")
    print(f"Efficient output: {y2.shape}")


def demo_padding_behavior():
    """Demonstrate how padding works with non-standard dimensions."""
    print("\n" + "=" * 50)
    print("Padding Behavior Demo")
    print("=" * 50)
    
    # Test with dimensions that require padding
    test_dims = [5, 7, 10, 15, 20]
    
    for dim in test_dims:
        print(f"\nTesting dimension {dim}:")
        
        chain = ButterflyChain(
            data_dim=dim,
            block_size=1,
            weight_init="identity"
        )
        
        # Show padding info for each level
        info = chain.get_parameter_info()
        for level_info in info['levels']:
            if level_info['needs_padding']:
                print(f"  Level {level_info['level']}: "
                      f"{dim} -> {level_info['padded_dim']} (padded)")
            else:
                print(f"  Level {level_info['level']}: no padding needed")
        
        # Test forward pass
        x = torch.randn(2, dim)
        y = chain(x)
        print(f"  Forward pass: {x.shape} -> {y.shape}")
        
        # For identity init, check if padding is handled correctly
        if torch.allclose(x, y, atol=1e-4):
            print("  ✓ Identity transformation preserved")
        else:
            diff = torch.norm(y - x).item()
            print(f"  ! Difference from identity: {diff:.6f}")


if __name__ == "__main__":
    torch.manual_seed(42)
    
    try:
        demo_basic_usage()
        demo_different_configurations()
        demo_efficient_variant()
        demo_padding_behavior()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("=" * 50)
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure PyTorch is installed.")
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()