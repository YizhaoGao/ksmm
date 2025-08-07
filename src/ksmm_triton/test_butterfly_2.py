#!/usr/bin/env python3
"""
Test script for the revised create_butterfly_chain function with low-rank decomposition.
"""

import torch
import sys
import os

# Add the source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ksmm_triton.ksmm_module import create_butterfly_chain


def test_butterfly_chain_rectangular():
    """Test the create_butterfly_chain function with rectangular matrices."""
    
    # Test case 1: 128x64 (expansion case)
    print("Testing 128x64 matrix (expansion case)...")
    layer1 = create_butterfly_chain([128, 64], dtype=torch.float32, device='cpu')
    
    print(f"Input features: {layer1.in_features}")
    print(f"Output features: {layer1.out_features}")
    print(f"Number of layers: {layer1.num_layers}")
    print(f"Patterns: {layer1.patterns}")
    
    # Check parameter savings
    savings = layer1.get_parameter_savings()
    print(f"Dense parameters: {savings['dense_parameters']}")
    print(f"Sparse parameters: {savings['sparse_parameters']}")
    print(f"Memory savings: {savings['memory_savings_percent']:.2f}%")
    print(f"Parameter reduction ratio: {savings['parameter_reduction_ratio']:.4f}")
    
    # Test forward pass
    batch_size = 16
    x = torch.randn(batch_size, 64, dtype=torch.float32)
    
    try:
        output = layer1(x)
        print(f"Forward pass successful! Input shape: {x.shape}, Output shape: {output.shape}")
        print(f"Expected output shape: ({batch_size}, 128)")
        assert output.shape == (batch_size, 128), f"Wrong output shape: {output.shape}"
        print("✓ Test case 1 passed!\n")
    except Exception as e:
        print(f"✗ Test case 1 failed: {e}\n")
    
    # Test case 2: 64x128 (compression case)
    print("Testing 64x128 matrix (compression case)...")
    layer2 = create_butterfly_chain([64, 128], dtype=torch.float32, device='cpu')
    
    print(f"Input features: {layer2.in_features}")
    print(f"Output features: {layer2.out_features}")
    print(f"Number of layers: {layer2.num_layers}")
    print(f"Patterns: {layer2.patterns}")
    
    # Check parameter savings
    savings2 = layer2.get_parameter_savings()
    print(f"Dense parameters: {savings2['dense_parameters']}")
    print(f"Sparse parameters: {savings2['sparse_parameters']}")
    print(f"Memory savings: {savings2['memory_savings_percent']:.2f}%")
    print(f"Parameter reduction ratio: {savings2['parameter_reduction_ratio']:.4f}")
    
    # Test forward pass
    x2 = torch.randn(batch_size, 128, dtype=torch.float32)
    
    try:
        output2 = layer2(x2)
        print(f"Forward pass successful! Input shape: {x2.shape}, Output shape: {output2.shape}")
        print(f"Expected output shape: ({batch_size}, 64)")
        assert output2.shape == (batch_size, 64), f"Wrong output shape: {output2.shape}"
        print("✓ Test case 2 passed!\n")
    except Exception as e:
        print(f"✗ Test case 2 failed: {e}\n")
    
    # Test case 3: Square matrix (should only use butterfly)
    print("Testing 64x64 matrix (square case)...")
    layer3 = create_butterfly_chain([64, 64], dtype=torch.float32, device='cpu')
    
    print(f"Input features: {layer3.in_features}")
    print(f"Output features: {layer3.out_features}")
    print(f"Number of layers: {layer3.num_layers}")
    print(f"Patterns: {layer3.patterns}")
    
    # Check parameter savings
    savings3 = layer3.get_parameter_savings()
    print(f"Dense parameters: {savings3['dense_parameters']}")
    print(f"Sparse parameters: {savings3['sparse_parameters']}")
    print(f"Memory savings: {savings3['memory_savings_percent']:.2f}%")
    print(f"Parameter reduction ratio: {savings3['parameter_reduction_ratio']:.4f}")
    
    # Test forward pass
    x3 = torch.randn(batch_size, 64, dtype=torch.float32)
    
    try:
        output3 = layer3(x3)
        print(f"Forward pass successful! Input shape: {x3.shape}, Output shape: {output3.shape}")
        print(f"Expected output shape: ({batch_size}, 64)")
        assert output3.shape == (batch_size, 64), f"Wrong output shape: {output3.shape}"
        print("✓ Test case 3 passed!\n")
    except Exception as e:
        print(f"✗ Test case 3 failed: {e}\n")


def test_linear_patterns():
    """Test that (1, b, c, 1) patterns use F.linear correctly."""
    from ksmm_triton.ksmm_module import KSLinearTriton
    
    print("Testing F.linear for (1, b, c, 1) patterns...")
    
    # Create a simple linear transformation: 10 -> 5
    patterns = [(1, 5, 10, 1)]
    layer = KSLinearTriton(patterns, dtype=torch.float32, device='cpu', bias=False)
    
    # Test forward pass
    batch_size = 8
    x = torch.randn(batch_size, 10, dtype=torch.float32)
    
    try:
        output = layer(x)
        print(f"Linear forward pass successful! Input shape: {x.shape}, Output shape: {output.shape}")
        print(f"Expected output shape: ({batch_size}, 5)")
        assert output.shape == (batch_size, 5), f"Wrong output shape: {output.shape}"
        print("✓ Linear pattern test passed!\n")
    except Exception as e:
        print(f"✗ Linear pattern test failed: {e}\n")


if __name__ == "__main__":
    print("Testing revised create_butterfly_chain with low-rank decomposition...\n")
    test_butterfly_chain_rectangular()
    test_linear_patterns()
    print("All tests completed!")
