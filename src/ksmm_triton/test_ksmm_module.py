#!/usr/bin/env python3
"""
Test script for KSLinearTriton module.
This script validates the correctness and basic functionality of the module.
"""

import torch
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ksmm_triton.ksmm_module import KSLinearTriton, create_simple_ks_layer
from ksmm_triton.ksmm_triton_tc import ks_triton, kronecker_bmm_reference


def test_single_layer():
    """Test a single Kronecker-sparse layer."""
    print("Testing single layer...")
    
    pattern = (4, 16, 16, 1)
    a, b, c, d = pattern
    
    in_features = a * c * d  # 64
    out_features = a * b * d  # 64
    batch_size = 8
    
    # Create layer
    ks_layer = create_simple_ks_layer(
        in_features=in_features,
        out_features=out_features,
        pattern=pattern,
        dtype=torch.float16,
        device='cuda'
    )
    
    # Test forward pass
    x = torch.randn(batch_size, in_features, dtype=torch.float16, device='cuda')
    output = ks_layer(x)
    
    assert output.shape == (batch_size, out_features), f"Expected shape {(batch_size, out_features)}, got {output.shape}"
    assert output.dtype == torch.float16, f"Expected dtype float16, got {output.dtype}"
    assert output.device.type == 'cuda', f"Expected cuda device, got {output.device}"
    
    # Test backward pass
    loss = output.sum()
    loss.backward()
    
    # Check gradients exist
    assert ks_layer.weights[0].grad is not None, "No gradient for weights"
    if ks_layer.bias is not None:
        assert ks_layer.bias.grad is not None, "No gradient for bias"
    
    print("✓ Single layer test passed")


def test_chain():
    """Test a chain of Kronecker-sparse layers."""
    print("Testing chain...")
    
    patterns = [
        (4, 16, 16, 1),  # 64 -> 64
        (4, 8, 16, 1),   # 64 -> 32  
        (2, 8, 16, 1),   # 32 -> 16
    ]
    
    batch_size = 8
    
    # Create chain
    ks_chain = KSLinearTriton(
        patterns=patterns,
        dtype=torch.float16,
        device='cuda'
    )
    
    # Test forward pass
    x = torch.randn(batch_size, ks_chain.in_features, dtype=torch.float16, device='cuda')
    output = ks_chain(x)
    
    expected_shape = (batch_size, ks_chain.out_features)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    # Test backward pass
    loss = output.sum()
    loss.backward()
    
    # Check gradients for all layers
    for i, weight in enumerate(ks_chain.weights):
        assert weight.grad is not None, f"No gradient for weight {i}"
    
    print("✓ Chain test passed")


def test_batch_size_last():
    """Test batch-size-last layout."""
    print("Testing batch-size-last layout...")
    
    pattern = (2, 8, 8, 2)
    a, b, c, d = pattern
    
    in_features = a * c * d   # 32
    out_features = a * b * d  # 32
    batch_size = 16
    
    # Create BSL layer
    ks_layer = KSLinearTriton(
        patterns=[pattern],
        bs_last=True,
        dtype=torch.float16,
        device='cuda'
    )
    
    # Test with BSL input format
    x = torch.randn(in_features, batch_size, dtype=torch.float16, device='cuda')
    output = ks_layer(x)
    
    expected_shape = (out_features, batch_size)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    # Test backward pass
    loss = output.sum()
    loss.backward()
    
    assert ks_layer.weights[0].grad is not None, "No gradient for weights"
    
    print("✓ Batch-size-last test passed")


def test_custom_weights():
    """Test custom weight initialization."""
    print("Testing custom weights...")
    
    pattern = (2, 4, 4, 1)
    a, b, c, d = pattern
    
    # Create custom weight
    custom_weight = torch.ones(a, b, c, d, dtype=torch.float16, device='cuda') * 0.5
    
    # Create layer with custom weights
    ks_layer = KSLinearTriton(
        patterns=[pattern],
        weights=[custom_weight],
        dtype=torch.float16,
        device='cuda'
    )
    
    # Check that weights were set correctly
    # The weight is stored in BMM format, so we need to convert back to check
    weight_bmm = ks_layer.weights[0]
    weight_reconstructed = weight_bmm.reshape(a, d, c, b).permute(0, 3, 2, 1)
    
    assert torch.allclose(weight_reconstructed, custom_weight, rtol=1e-3), "Custom weights not set correctly"
    
    print("✓ Custom weights test passed")


def test_correctness_against_reference():
    """Test correctness against the reference implementation."""
    print("Testing correctness against reference...")
    
    pattern = (2, 8, 8, 1)
    a, b, c, d = pattern
    batch_size = 4
    
    # Create layer
    ks_layer = KSLinearTriton(
        patterns=[pattern],
        bias=False,  # Disable bias for clean comparison
        dtype=torch.float16,
        device='cuda'
    )
    
    # Get the weight in BMM format
    K_bmm = ks_layer.weights[0]
    
    # Create input
    x = torch.randn(batch_size, a * c * d, dtype=torch.float16, device='cuda')
    
    # Forward pass with KSLinearTriton
    output_triton = ks_layer(x)
    
    # Forward pass with reference BMM implementation
    K_bmm_ref = K_bmm.transpose(-1, -2).contiguous()  # Adjust format for reference
    output_bmm = kronecker_bmm_reference(x, K_bmm_ref, pattern)
    
    # Compare outputs
    max_diff = torch.max(torch.abs(output_triton - output_bmm)).item()
    relative_error = torch.norm(output_triton - output_bmm) / torch.norm(output_bmm)
    
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Relative error: {relative_error:.6f}")
    
    # Allow for small numerical differences due to different computation orders
    assert max_diff < 1e-2, f"Too large difference: {max_diff}"
    assert relative_error < 1e-3, f"Too large relative error: {relative_error}"
    
    print("✓ Correctness test passed")


def test_error_handling():
    """Test error handling and validation."""
    print("Testing error handling...")
    
    # Test invalid pattern chain
    try:
        invalid_patterns = [
            (2, 4, 4, 1),  # Output: 8
            (2, 4, 8, 1),  # Input: 16 (mismatch!)
        ]
        KSLinearTriton(patterns=invalid_patterns, device='cuda')
        assert False, "Should have raised ValueError for incompatible chain"
    except ValueError:
        pass  # Expected
    
    # Test invalid weight shape
    try:
        pattern = (2, 4, 4, 1)
        wrong_weight = torch.randn(3, 4, 4, 1, dtype=torch.float16, device='cuda')  # Wrong 'a'
        KSLinearTriton(patterns=[pattern], weights=[wrong_weight], device='cuda')
        assert False, "Should have raised ValueError for wrong weight shape"
    except ValueError:
        pass  # Expected
    
    # Test invalid input dimension
    try:
        pattern = (2, 4, 4, 1)
        ks_layer = KSLinearTriton(patterns=[pattern], device='cuda')
        wrong_input = torch.randn(8, 10, dtype=torch.float16, device='cuda')  # Wrong last dim
        ks_layer(wrong_input)
        assert False, "Should have raised ValueError for wrong input dimension"
    except ValueError:
        pass  # Expected
    
    print("✓ Error handling test passed")


def test_utilities():
    """Test utility methods."""
    print("Testing utility methods...")
    
    patterns = [
        (2, 8, 8, 1),   # 16 params
        (2, 4, 8, 1),   # 8 params  
    ]
    
    ks_layer = KSLinearTriton(patterns=patterns, device='cuda')
    
    # Test weight size calculation
    expected_size = 2*8*8*1 + 2*4*8*1  # 128 + 64 = 192
    actual_size = ks_layer.get_weights_size()
    assert actual_size == expected_size, f"Expected {expected_size} parameters, got {actual_size}"
    
    # Test theoretical speedup calculation
    dense_size = ks_layer.in_features * ks_layer.out_features
    sparse_size = ks_layer.get_weights_size()
    expected_speedup = dense_size / sparse_size
    actual_speedup = ks_layer.get_theoretical_speedup()
    assert abs(actual_speedup - expected_speedup) < 1e-6, "Speedup calculation incorrect"
    
    # Test dense product generation
    dense_weight = ks_layer.get_dense_product()
    expected_shape = (ks_layer.out_features, ks_layer.in_features)
    assert dense_weight.shape == expected_shape, f"Dense weight shape {dense_weight.shape} != {expected_shape}"
    
    print("✓ Utility methods test passed")


def run_all_tests():
    """Run all tests."""
    print("Running KSLinearTriton tests...")
    print("=" * 50)
    
    try:
        test_single_layer()
        test_chain()
        test_batch_size_last()
        test_custom_weights()
        test_correctness_against_reference()
        test_error_handling()
        test_utilities()
        
        print("=" * 50)
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print("=" * 50)
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
