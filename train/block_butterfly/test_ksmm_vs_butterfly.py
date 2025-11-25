"""
Test to verify that ksmm_module and block_butterfly produce identical outputs
when initialized with the same weights and given the same inputs.

This test:
1. Creates a single test configuration (data_dim=16, block_size=1)
2. Generates random weights externally
3. Injects these weights into both ButterflyChain and KSLinearTriton
4. Verifies they produce identical outputs for the same input
"""

import torch
import torch.nn as nn
import sys
import os
import math

# Add paths to import both modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src/ksmm_triton'))
sys.path.insert(0, os.path.dirname(__file__))

from ksmm_module import KSLinearTriton, create_butterfly_patterns
from butterfly_chain import ButterflyChain


def generate_external_weights(data_dim, block_size=1, dtype=torch.float32, device='cuda', seed=42):
    """
    Generate weights externally to be injected into both models.
    
    For butterfly with block_size=1 and data_dim as power of 2:
    - Number of levels: log2(data_dim)
    - Level l has pattern (a=2^l, b=2, c=2, d=2^(n-l-1))
    - Each level has 'a' groups and 'd' blocks per group
    - Each block is a 2x2 matrix (when block_size=1)
    
    Args:
        data_dim: must be a power of 2 for block_size=1
        block_size: block size (default 1)
        dtype: data type for weights
        device: device for weights
        seed: random seed
    
    Returns:
        List of weight tensors, one per level
    """
    torch.manual_seed(seed)
    
    # Calculate number of levels
    n = int(math.log2(data_dim))
    num_levels = n
    
    weights = []
    
    # Generate weights for each level
    # Butterfly patterns are ordered from right to left in the matrix product
    for level in range(num_levels):
        # Pattern for this level
        a = 2 ** level
        b = 2
        c = 2
        d = 2 ** (n - level - 1)
        
        # For ButterflyChain: shape is (groups=a, nb=d, 2, 2, C, C)
        # For block_size=1: (a, d, 2, 2, 1, 1)
        level_weights = torch.randn(a, d, b, c, block_size, block_size, 
                                     dtype=dtype, device=device)
        weights.append(level_weights)
    
    return weights


def inject_weights_into_butterfly(butterfly_model, external_weights):
    """
    Inject externally generated weights into ButterflyChain.
    
    Args:
        butterfly_model: ButterflyChain instance
        external_weights: list of weight tensors with shape (a, d, 2, 2, C, C)
    """
    assert len(external_weights) == len(butterfly_model.butterfly_layers), \
        f"Weight count mismatch: {len(external_weights)} vs {len(butterfly_model.butterfly_layers)}"
    
    for layer, weights in zip(butterfly_model.butterfly_layers, external_weights):
        # ButterflyChain expects shape (groups, nb, 2, 2, C, C)
        assert layer.w.shape == weights.shape, \
            f"Shape mismatch: layer expects {layer.w.shape}, got {weights.shape}"
        layer.w.data = weights.clone()


def inject_weights_into_ksmm(ksmm_model, external_weights, patterns):
    """
    Inject externally generated weights into KSLinearTriton.
    
    Args:
        ksmm_model: KSLinearTriton instance
        external_weights: list of weight tensors with shape (a, d, 2, 2, C, C)
        patterns: list of (a, b, c, d) patterns
    """
    assert len(external_weights) == len(ksmm_model.weights), \
        f"Weight count mismatch: {len(external_weights)} vs {len(ksmm_model.weights)}"
    
    for weight_idx, (butterfly_w, pattern) in enumerate(zip(external_weights, patterns)):
        a, b, c, d = pattern
        
        # butterfly_w shape: (a, d, 2, 2, C, C)
        groups, nb, _, _, C, _ = butterfly_w.shape
        
        # Verify dimensions match pattern
        assert groups == a and nb == d, \
            f"Weight {weight_idx}: dimensions don't match pattern"
        
        # Convert butterfly format to BMM format
        # BMM format for ksmm: (a*d, c*C, b*C)
        bmm_weights = torch.zeros(a * d, c * C, b * C, 
                                   dtype=butterfly_w.dtype, device=butterfly_w.device)
        
        for g in range(a):
            for n in range(d):
                block_idx = g * d + n
                # Place the four CxC blocks in the correct positions
                # butterfly_w[g, n, i, j, :, :] is the (i,j)-th block
                for i in range(b):
                    for j in range(c):
                        bmm_weights[block_idx, j*C:(j+1)*C, i*C:(i+1)*C] = butterfly_w[g, n, i, j, :, :]
        
        # Set the weights in KSLinearTriton
        ksmm_model.weights[weight_idx].data = bmm_weights


def test_single_config():
    """
    Test a single configuration with externally generated weights.
    
    Configuration:
    - data_dim: 16
    - block_size: 1
    - batch_size: 4
    - dtype: float32
    """
    print("\n" + "="*80)
    print("Testing KSMM vs Butterfly Chain with External Weights")
    print("="*80)
    
    # Configuration
    data_dim = 16
    block_size = 1
    batch_size = 4
    dtype = torch.float32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 42
    
    print(f"\nConfiguration:")
    print(f"  data_dim: {data_dim}")
    print(f"  block_size: {block_size}")
    print(f"  batch_size: {batch_size}")
    print(f"  dtype: {dtype}")
    print(f"  device: {device}")
    print(f"  seed: {seed}")
    
    # Generate external weights
    print(f"\n{'='*80}")
    print("Step 1: Generate External Weights")
    print("="*80)
    
    external_weights = generate_external_weights(
        data_dim=data_dim,
        block_size=block_size,
        dtype=dtype,
        device=device,
        seed=seed
    )
    
    print(f"\nGenerated {len(external_weights)} weight tensors:")
    for i, w in enumerate(external_weights):
        print(f"  Level {i}: shape={w.shape}, mean={w.mean().item():.4f}, std={w.std().item():.4f}")
    
    # Create ButterflyChain model
    print(f"\n{'='*80}")
    print("Step 2: Create ButterflyChain Model")
    print("="*80)
    
    butterfly_model = ButterflyChain(
        data_dim=data_dim,
        block_size=block_size,
        share_weights=False,
        weight_init='identity'  # Will be overwritten
    ).to(device).to(dtype)
    
    # Inject weights into ButterflyChain
    inject_weights_into_butterfly(butterfly_model, external_weights)
    
    butterfly_info = butterfly_model.get_parameter_info()
    print(f"\nButterflyChain Info:")
    print(f"  Total parameters: {butterfly_info['total_params']}")
    print(f"  Number of levels: {butterfly_info['num_levels']}")
    for level_info in butterfly_info['levels']:
        print(f"  Level {level_info['level']}: {level_info['params']} params, "
              f"groups={level_info['groups']}, nb={level_info['nb']}")
    
    # Create KSLinearTriton model
    print(f"\n{'='*80}")
    print("Step 3: Create KSLinearTriton Model")
    print("="*80)
    
    n = int(math.log2(data_dim))
    patterns = create_butterfly_patterns(n, block_size=block_size)
    
    print(f"\nButterfly patterns (n={n}):")
    for i, pattern in enumerate(patterns):
        print(f"  Pattern {i}: {pattern}")
    
    ksmm_model = KSLinearTriton(
        patterns=patterns,
        bias=False,
        dtype=dtype,
        bs_last=False,
        device=device,
        impl='bmm'
    )
    
    # Inject weights into KSLinearTriton
    inject_weights_into_ksmm(ksmm_model, external_weights, patterns)
    
    print(f"\nKSLinearTriton Info:")
    print(f"  Total parameters: {ksmm_model.get_weights_size()}")
    print(f"  Number of patterns: {len(patterns)}")
    
    # Create test input
    print(f"\n{'='*80}")
    print("Step 4: Create Test Input")
    print("="*80)
    
    torch.manual_seed(seed + 1)  # Different seed for input
    x = torch.randn(batch_size, data_dim, dtype=dtype, device=device)
    
    print(f"\nInput:")
    print(f"  Shape: {x.shape}")
    print(f"  Mean: {x.mean().item():.4f}")
    print(f"  Std: {x.std().item():.4f}")
    print(f"  Min: {x.min().item():.4f}")
    print(f"  Max: {x.max().item():.4f}")
    
    # Forward pass
    print(f"\n{'='*80}")
    print("Step 5: Forward Pass")
    print("="*80)
    
    with torch.no_grad():
        butterfly_out = butterfly_model(x)
        ksmm_out = ksmm_model(x)
    
    print(f"\nButterflyChain output:")
    print(f"  Shape: {butterfly_out.shape}")
    print(f"  Mean: {butterfly_out.mean().item():.4f}")
    print(f"  Std: {butterfly_out.std().item():.4f}")
    print(f"  Min: {butterfly_out.min().item():.4f}")
    print(f"  Max: {butterfly_out.max().item():.4f}")
    
    print(f"\nKSLinearTriton output:")
    print(f"  Shape: {ksmm_out.shape}")
    print(f"  Mean: {ksmm_out.mean().item():.4f}")
    print(f"  Std: {ksmm_out.std().item():.4f}")
    print(f"  Min: {ksmm_out.min().item():.4f}")
    print(f"  Max: {ksmm_out.max().item():.4f}")
    
    # Compare outputs
    print(f"\n{'='*80}")
    print("Step 6: Compare Outputs")
    print("="*80)
    
    diff = torch.abs(butterfly_out - ksmm_out)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"\nAbsolute difference:")
    print(f"  Max: {max_diff:.6e}")
    print(f"  Mean: {mean_diff:.6e}")
    print(f"  Median: {diff.median().item():.6e}")
    
    # Relative difference
    rel_diff = diff / (torch.abs(butterfly_out) + 1e-8)
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()
    
    print(f"\nRelative difference:")
    print(f"  Max: {max_rel_diff:.6e}")
    print(f"  Mean: {mean_rel_diff:.6e}")
    
    # Check if outputs are close
    rtol = 1e-4
    atol = 1e-5
    are_close = torch.allclose(butterfly_out, ksmm_out, rtol=rtol, atol=atol)
    
    print(f"\nTolerance check (rtol={rtol}, atol={atol}):")
    if are_close:
        print(f"  ✓ PASS: Outputs are equivalent!")
    else:
        print(f"  ✗ FAIL: Outputs differ beyond tolerance")
        
        # Show sample differences
        print(f"\n  Sample differences (first 5 elements of batch 0):")
        for i in range(min(5, data_dim)):
            b_val = butterfly_out[0, i].item()
            k_val = ksmm_out[0, i].item()
            d_val = diff[0, i].item()
            print(f"    [{i}] butterfly={b_val:.6f}, ksmm={k_val:.6f}, diff={d_val:.6e}")
        
        # Show worst differences
        print(f"\n  Worst 5 differences:")
        flat_diff = diff.flatten()
        flat_butterfly = butterfly_out.flatten()
        flat_ksmm = ksmm_out.flatten()
        worst_indices = torch.topk(flat_diff, min(5, flat_diff.numel())).indices
        for idx in worst_indices:
            b_val = flat_butterfly[idx].item()
            k_val = flat_ksmm[idx].item()
            d_val = flat_diff[idx].item()
            print(f"    [idx={idx}] butterfly={b_val:.6f}, ksmm={k_val:.6f}, diff={d_val:.6e}")
    
    print(f"\n{'='*80}")
    
    return are_close


if __name__ == "__main__":
    # Run the test
    try:
        passed = test_single_config()
        
        if passed:
            print("\n✓ SUCCESS: Test passed!")
            exit(0)
        else:
            print("\n✗ FAILURE: Test failed!")
            exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
