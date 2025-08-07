#!/usr/bin/env python3
"""
Test script for butterfly pattern generation and butterfly chain creation.
"""

import torch
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ksmm_triton.ksmm_module import create_butterfly_patterns, create_butterfly_chain


def test_butterfly_patterns():
    """Test butterfly pattern generation."""
    print("Testing butterfly pattern generation...")
    
    # Test n=3 (8x8 matrix)
    patterns = create_butterfly_patterns(3)
    print(f"n=3 patterns: {patterns}")
    
    # Verify we get 3 patterns
    assert len(patterns) == 3, f"Expected 3 patterns for n=3, got {len(patterns)}"
    
    # Check each pattern has correct dimensions
    # For n=3: 2^3 = 8, so we should get 8x8 -> 8x8 chain
    input_size = patterns[0][0] * patterns[0][2] * patterns[0][3]  # First pattern input
    output_size = patterns[-1][0] * patterns[-1][1] * patterns[-1][3]  # Last pattern output
    
    assert input_size == 8, f"Expected input size 8, got {input_size}"
    assert output_size == 8, f"Expected output size 8, got {output_size}"
    
    # Check pattern chain compatibility
    for i in range(len(patterns) - 1):
        current_out = patterns[i][0] * patterns[i][1] * patterns[i][3]
        next_in = patterns[i + 1][0] * patterns[i + 1][2] * patterns[i + 1][3]
        assert current_out == next_in, f"Pattern {i} output {current_out} != Pattern {i+1} input {next_in}"
    
    print("✓ Butterfly patterns test passed")


def test_butterfly_chain_square_power_of_2():
    """Test butterfly chain for square power-of-2 matrices."""
    print("Testing butterfly chain for square power-of-2...")
    
    # Test 64x64 (2^6 x 2^6)
    shape = [64, 64]
    chain = create_butterfly_chain(shape, dtype=torch.float16, device='cuda')
    
    assert chain.in_features == 64, f"Expected in_features=64, got {chain.in_features}"
    assert chain.out_features == 64, f"Expected out_features=64, got {chain.out_features}"
    
    # Test forward pass
    batch_size = 8
    x = torch.randn(batch_size, 64, dtype=torch.float16, device='cuda')
    output = chain(x)
    
    assert output.shape == (batch_size, 64), f"Expected output shape ({batch_size}, 64), got {output.shape}"
    
    print(f"Created butterfly chain with {len(chain.patterns)} patterns")
    print(f"Pattern chain: {chain.patterns}")
    
    print("✓ Square power-of-2 butterfly chain test passed")


def test_butterfly_chain_rectangular():
    """Test butterfly chain for rectangular matrices."""
    print("Testing butterfly chain for rectangular matrices...")
    
    # Test 128x64 (larger output)
    shape = [128, 64]
    chain = create_butterfly_chain(shape, dtype=torch.float16, device='cuda')
    
    assert chain.in_features == 64, f"Expected in_features=64, got {chain.in_features}"
    assert chain.out_features == 128, f"Expected out_features=128, got {chain.out_features}"
    
    # Test forward pass
    batch_size = 8
    x = torch.randn(batch_size, 64, dtype=torch.float16, device='cuda')
    output = chain(x)
    
    assert output.shape == (batch_size, 128), f"Expected output shape ({batch_size}, 128), got {output.shape}"
    
    print(f"Created butterfly chain (128x64) with {len(chain.patterns)} patterns")
    print(f"Pattern chain: {chain.patterns}")
    
    # Test 64x128 (larger input)
    shape = [64, 128]
    chain2 = create_butterfly_chain(shape, dtype=torch.float16, device='cuda')
    
    assert chain2.in_features == 128, f"Expected in_features=128, got {chain2.in_features}"
    assert chain2.out_features == 64, f"Expected out_features=64, got {chain2.out_features}"
    
    # Test forward pass
    x2 = torch.randn(batch_size, 128, dtype=torch.float16, device='cuda')
    output2 = chain2(x2)
    
    assert output2.shape == (batch_size, 64), f"Expected output shape ({batch_size}, 64), got {output2.shape}"
    
    print(f"Created butterfly chain (64x128) with {len(chain2.patterns)} patterns")
    print(f"Pattern chain: {chain2.patterns}")
    
    print("✓ Rectangular butterfly chain test passed")


def test_butterfly_chain_non_power_of_2():
    """Test butterfly chain for non-power-of-2 dimensions."""
    print("Testing butterfly chain for non-power-of-2 dimensions...")
    
    # Test 96x48 (neither is power of 2)
    shape = [96, 48]
    chain = create_butterfly_chain(shape, dtype=torch.float16, device='cuda')
    
    assert chain.in_features == 48, f"Expected in_features=48, got {chain.in_features}"
    assert chain.out_features == 96, f"Expected out_features=96, got {chain.out_features}"
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 48, dtype=torch.float16, device='cuda')
    output = chain(x)
    
    assert output.shape == (batch_size, 96), f"Expected output shape ({batch_size}, 96), got {output.shape}"
    
    print(f"Created butterfly chain (96x48) with {len(chain.patterns)} patterns")
    print(f"Pattern chain: {chain.patterns}")
    
    print("✓ Non-power-of-2 butterfly chain test passed")


def test_butterfly_chain_backward():
    """Test backward pass through butterfly chain."""
    print("Testing butterfly chain backward pass...")
    
    shape = [32, 16]
    chain = create_butterfly_chain(shape, dtype=torch.float16, device='cuda')
    
    batch_size = 4
    x = torch.randn(batch_size, 16, dtype=torch.float16, device='cuda', requires_grad=True)
    output = chain(x)
    
    # Test backward pass
    loss = output.sum()
    loss.backward()
    
    # Check gradients exist
    assert x.grad is not None, "No gradient for input"
    for i, weight in enumerate(chain.weights):
        assert weight.grad is not None, f"No gradient for weight {i}"
    
    print("✓ Butterfly chain backward test passed")


def test_butterfly_pattern_properties():
    """Test mathematical properties of butterfly patterns."""
    print("Testing butterfly pattern mathematical properties...")
    
    for n in [2, 3, 4, 5]:
        patterns = create_butterfly_patterns(n)
        matrix_size = 2 ** n
        
        print(f"n={n}, matrix_size={matrix_size}")
        
        # Check all patterns have b=c=2 (butterfly property)
        for i, (a, b, c, d) in enumerate(patterns):
            assert b == 2 and c == 2, f"Pattern {i}: b={b}, c={c}, expected b=c=2"
            print(f"  Pattern {i}: (a={a}, b={b}, c={c}, d={d})")
        
        # Check input/output sizes
        first_pattern = patterns[0]
        last_pattern = patterns[-1]
        
        input_size = first_pattern[0] * first_pattern[2] * first_pattern[3]
        output_size = last_pattern[0] * last_pattern[1] * last_pattern[3]
        
        assert input_size == matrix_size, f"Input size {input_size} != {matrix_size}"
        assert output_size == matrix_size, f"Output size {output_size} != {matrix_size}"
        
        # Check parameter efficiency
        total_params = sum(a * b * c * d for a, b, c, d in patterns)
        dense_params = matrix_size * matrix_size
        compression_ratio = dense_params / total_params
        
        print(f"  Total params: {total_params}, Dense params: {dense_params}")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        
        assert compression_ratio > 1, f"No compression achieved: {compression_ratio}"
    
    print("✓ Butterfly pattern properties test passed")


def run_all_tests():
    """Run all butterfly tests."""
    print("Running Butterfly Pattern Tests...")
    print("=" * 60)
    
    try:
        test_butterfly_patterns()
        test_butterfly_chain_square_power_of_2()
        test_butterfly_chain_rectangular()
        test_butterfly_chain_non_power_of_2()
        test_butterfly_chain_backward()
        test_butterfly_pattern_properties()
        
        print("=" * 60)
        print("✅ All butterfly tests passed!")
        return True
        
    except Exception as e:
        print("=" * 60)
        print(f"❌ Butterfly test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
