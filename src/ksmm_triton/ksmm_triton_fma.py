import torch
import triton
import triton.language as tl
from ksmm_py.layer.kronecker_sparse.interface import KSLinear

import argparse
from typing import List, Tuple
import time


def benchmark_function(func, *args, warmup_runs=10, test_runs=100):
    """
    Benchmark a function with warmup and multiple test runs.
    Returns (mean_time_ms, std_time_ms).
    """
    device = args[0].device if hasattr(args[0], 'device') else 'cuda'
    
    # Warmup runs
    torch.cuda.synchronize()
    for _ in range(warmup_runs):            
        _ = func(*args)
    torch.cuda.synchronize()
    
    # Actual timing runs
    times = []
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(test_runs):
        _ = func(*args)
    torch.cuda.synchronize()       
    end_time = time.perf_counter()
    times.append((end_time - start_time) * 1000)  # Convert to milliseconds
    
    mean_time = sum(times) / test_runs
    std_time = (sum((t - mean_time) ** 2 for t in times) / test_runs) ** 0.5
    
    return mean_time, std_time


def parse_patterns(pattern_str: str) -> List[Tuple[int, int, int, int]]:
    """
    Parse pattern string like "[(6,64,64,1),(1,768,192,2)]" into list of tuples.
    """
    # Remove brackets and spaces
    pattern_str = pattern_str.strip("[]").replace(" ", "")
    
    # Split by ),( to get individual patterns
    pattern_parts = pattern_str.split("),(")
    patterns = []
    
    for i, part in enumerate(pattern_parts):
        # Clean up the pattern string
        if i == 0:
            part = part.lstrip("(")
        if i == len(pattern_parts) - 1:
            part = part.rstrip(")")
        
        # Parse the four integers
        values = [int(x) for x in part.split(",")]
        if len(values) != 4:
            raise ValueError(f"Each pattern must have exactly 4 values, got {len(values)}")
        patterns.append(tuple(values))
    
    return patterns

def torch_mm_forward(x, weight_transposed):
    """Standard torch matrix multiplication."""
    return torch.matmul(x, weight_transposed)


def ksl_forward(ksl, x):
    """KSLinear forward pass."""
    return ksl(x)


# This is a reference implementation of the BMM baseline from Appendix A.1 of the paper
# used for verification purposes.
def kronecker_bmm_reference(X_bsf, K_bmm, pattern):
    """
    Reference implementation of the BMM baseline from the paper's appendix.
    This is for verification, not performance.
    """
    a, b, c, d = pattern
    B = X_bsf.shape[0]
    
    # Step 1: Permute Input X (equivalent to XQ^T)
    # Reshape and transpose to gather the correct elements
    X_perm = X_bsf.reshape(B, a, c, d).transpose(-1, -2).reshape(B, a * d, c).contiguous().transpose(0, 1)

    # Step 2: Batched Matrix Multiply (Ỹ = X̃K̃^T)
    # K_bmm has shape (ad, b, c). We need its transpose for the matmul.
    K_bmm_T = K_bmm.transpose(-1, -2)  # Shape (ad, c, b)
    Y_perm = torch.bmm(X_perm, K_bmm_T)

    # Step 3: Permute Output Y back (equivalent to ỸP^T)
    Y_bsf = Y_perm.transpose(0, 1).reshape(B, a, d, b).transpose(-1, -2).reshape(B, a * b * d)
    return Y_bsf



# Reduced configuration space with sensible constraints
configs = []

# block_sizes = [16, 32, 64, 128]  
batch_sizes = [16, 32, 64, 128] 

for BBATCH in batch_sizes:
    # Strategy 3: Limit num_warps based on total work per block
    total_work = 16 * 16 * BBATCH
    if total_work <= 1024:  # Small blocks
        warp_options = [1, 2]
    elif total_work <= 4096:  # Medium blocks
        warp_options = [2, 4]
    else:  # Large blocks
        warp_options = [4, 8]
    
    for num_warps in warp_options:
        # Strategy 4: Limit num_stages based on memory requirements
        # More stages = more shared memory usage
        if total_work <= 2048:
            stage_options = [2, 3, 4]  # Conservative for small blocks
        elif total_work <= 8192:
            stage_options = [3, 4, 5]  # Medium staging
        else:
            stage_options = [4, 5]     # Fewer stages for large blocks
        
        for num_stages in stage_options:
            configs.append(
                triton.Config(
                    {'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_C': 16, 'BLOCK_SIZE_BATCH': BBATCH},
                    num_warps=num_warps,
                    num_stages=num_stages
                )
            )

print(configs)

# configs.extend([
#     triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_C': 16, 'BLOCK_SIZE_BATCH': 128}, num_warps=8, num_stages=4),
# ])

print(f"Generated {len(configs)} configurations")


@triton.autotune(
    configs=configs,
    key=['B'],
)
@triton.jit
def ks_fused_kernel(
    # Pointers to matrices
    Y_ptr,
    X_ptr,
    K_ptr,
    # Matrix dimensions
    B,
    # Strides for BSL layout
    stride_X_batch, stride_X_feat,
    stride_Y_batch, stride_Y_feat,
    stride_K_tile, stride_K_c, stride_K_b,
    a: tl.constexpr, 
    b: tl.constexpr,
    c: tl.constexpr,
    d: tl.constexpr,
    # Tile dimensions
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
):
    """
    Triton kernel for Fused Kronecker-Sparse Matrix Multiplication (Y = XK^T).
    This implements the paper's proposed fused, output-stationary kernel.
    It assumes a Batch-Size-Last (BSL) memory layout for X and Y for best performance.
    """
    # 1. WORK DISTRIBUTION: Map program ID to the (i, j) tile
    # This corresponds to the parallel loop in Algorithm 2.
    pid = tl.program_id(axis=0)
    batch_pid = tl.program_id(axis=1)

    
    i = pid // d
    j = pid % d

    # 2. TILING: Create tiles for the inner matmul loop
    # This kernel computes a (BLOCK_SIZE_BATCH, BLOCK_SIZE_B) tile of the output Y.
    batch_offsets = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    b_offsets = tl.arange(0, BLOCK_SIZE_B)
    c_offsets = tl.arange(0, BLOCK_SIZE_C)

    # 4. POINTERS: Set up pointers to K and the first tile of X and Y
    # Pointer to the specific (c, b) block of K for this (i, j) tile
    k_ptr = K_ptr + pid * stride_K_tile
    

    x_col_base = i * c * d + j
    y_row_base = i * b * d + j

    accumulator = tl.zeros((BLOCK_SIZE_BATCH, BLOCK_SIZE_B), dtype=tl.float32)
    # Calculate the current tile's offsets
    b_offsets = tl.arange(0, BLOCK_SIZE_B)
    c_offsets = tl.arange(0, BLOCK_SIZE_C)

    # --- Load X tile (fused permutation) ---
    # Calculate strided column indices for X
    x_feat_offsets = x_col_base + (c_offsets) * d
    # Create pointers for the X tile
    x_ptrs = X_ptr + (batch_offsets[:, None] * stride_X_batch + x_feat_offsets[None, :] * stride_X_feat)

    # Load X tile with boundary checks
    x_mask = (batch_offsets[:, None] < B) & ((c_offsets)[None, :] < c)
    x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

    # --- Load K tile ---
    # K is stored contiguously for each (i,j) block, so this is a simple load.
    k_ptrs = k_ptr + ((c_offsets)[:, None] * stride_K_c + b_offsets[None, :] * stride_K_b)
    k_mask = ((c_offsets)[:, None] < c) & (b_offsets[None, :] < b)
    k_tile = tl.load(k_ptrs, mask=k_mask, other=0.0)
    # --- Matrix Multiply and Accumulate ---
    # accumulator += tl.dot(x_tile, k_tile)
    x_tile_f32 = x_tile.to(tl.float32)
    k_tile_f32 = k_tile.to(tl.float32)
    accumulator = tl.sum(x_tile_f32[:, :, None] * k_tile_f32[None, :, :], axis=1)
    

    # 6. STORE Y TILE: Write result back to global memory (fused permutation)
    # Calculate strided column indices for Y
    y_feat_offsets = y_row_base + b_offsets * d
    # Create pointers for the Y tile
    y_ptrs = Y_ptr + (batch_offsets[:, None] * stride_Y_batch + y_feat_offsets[None, :] * stride_Y_feat)
    # Store Y tile with boundary checks
    y_mask = (batch_offsets[:, None] < B) & (b_offsets[None, :] < b)
    tl.store(y_ptrs, accumulator, mask=y_mask)



def ks_triton(X, K_bmm, pattern, layout='BSF'):
    """
    PyTorch wrapper for the Fused Kronecker-Sparse kernel.
    
    Args:
        X (torch.Tensor): Input tensor. Shape (B, N) for BSF or (N, B) for BSL.
        K_bmm (torch.Tensor): Pre-permuted weight tensor of shape (a*d, c, b),
                              as used in the paper's BMM baseline.
        pattern (tuple): The (a, b, c, d) Kronecker-sparse pattern.
        layout (str): Memory layout, 'BSF' (batch-size-first) or 'BSL' (batch-size-last).
    """
    # 1. Setup and Input Validation
    a, b, c, d = pattern
    if layout == 'BSF':
        B, N = X.shape
        # The kernel is optimized for BSL, so we transpose.
        # This is a one-time cost. In a full BSL pipeline, this would be avoided.
        X_bsl = X.transpose(0, 1).contiguous()
    elif layout == 'BSL':
        N, B = X.shape
        X_bsl = X
    else:
        raise ValueError("layout must be 'BSF' or 'BSL'")

    M = a * b * d
    assert N == a * c * d, "Input dimension N does not match pattern"
    assert K_bmm.shape == (a * d, c, b), "K_bmm shape is incorrect"


    # 2. Output Tensor
    # Output is created in BSL layout
    Y_bsl = torch.empty((M, B), device=X.device, dtype=X.dtype)

    # 3. Triton Grid and Kernel Launch
    # Grid is (a*d) for (i,j) tiles, and B / BLOCK_BATCH for the batch dimension
    grid = lambda META: (
        a * d, triton.cdiv(B, META['BLOCK_SIZE_BATCH']), 1
    )
    
    # Kernel call
    ks_fused_kernel[grid](
        Y_bsl, X_bsl, K_bmm,
        B,
        # Strides for BSL layout
        X_bsl.stride(1), X_bsl.stride(0),
        Y_bsl.stride(1), Y_bsl.stride(0),
        K_bmm.stride(0), K_bmm.stride(1), K_bmm.stride(2),
        # Tuning parameters (BLOCK_SIZES)
        a, b, c, d,
    )

    # 4. Post-processing
    if layout == 'BSF':
        # Transpose back to BSF if that was the original layout
        return Y_bsl.transpose(0, 1).contiguous()
    else:
        return Y_bsl

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Benchmark KSLinear vs torch.mm speed')
    parser.add_argument('--patterns', type=str, required=True,
                       help='Patterns as string, e.g., "[(6,64,64,1)]"')
    parser.add_argument('--batch_size', type=int, default=25088,
                       help='Batch size for input tensor')
    parser.add_argument('--warmup_runs', type=int, default=10,
                       help='Number of warmup runs')
    parser.add_argument('--test_runs', type=int, default=100,
                       help='Number of test runs for averaging')
    # parser.add_argument('--output_file', type=str, default="output.json",
    #                    help='Output JSON file for results')

    args = parser.parse_args()

    patterns = parse_patterns(args.patterns)
    assert len(patterns) == 1, "Only one pattern is supported for this script"
    
    a, b, c, d = patterns[0]
    pattern = patterns[0]

    # Input dimensions
    N = a * c * d
    M = a * b * d
    B = args.batch_size
    
    print(f"Pattern (a,b,c,d): {pattern}")
    print(f"Input shape (B, N): ({B}, {N})")
    print(f"Output shape (B, M): ({B}, {M})")

    ksl = KSLinear(
        patterns=patterns,
        weights=None,  # Let it initialize randomly
        algo="kernel",
        dtype=torch.float16,
        bs_last=True,
        device='cuda'
    )
    dense_weight = ksl.get_dense_product()
    dense_weight_T = dense_weight.T    
    K_bmm = ksl.factors[0].view(a * d, c, b).transpose(-1, -2).contiguous()  # Shape (a*d, c, b)
    print(f"K_bmm shape: {K_bmm.shape}")
    

    X = torch.randn((B, N), device='cuda', dtype=torch.float16)
    X_T = X.T.contiguous()  # Transpose to (N, B) for BSL layout
    

    print("Running BMM")
    # K_bmm = torch.randn((a * d, b, c), device='cuda', dtype=torch.float16)
    Y_bmm = kronecker_bmm_reference(X, K_bmm, pattern)
    

    print("Running Triton Fused Kernel")
    K_bmm_T = K_bmm.transpose(-1, -2).contiguous()  # Ensure K_bmm is in (a*d, b, c) format
    Y_triton = ks_triton(X_T, K_bmm_T, pattern, layout='BSL')
    Y_triton = Y_triton.T  # Transpose back to BSF if needed

    print("Running Torch MM")
    Y_torch = torch_mm_forward(X, dense_weight_T)


    # --- Compare results --- all compare to Y_torch
    print("\nVerifying results...")
    bmm_close = torch.allclose(Y_bmm, Y_torch, rtol=1e-2, atol=1e-2)
    triton_close = torch.allclose(Y_triton, Y_torch, rtol=1e-2, atol=1e-2)
    print(f"BMM close to Torch MM: {bmm_close}", "max diff:", torch.max(torch.abs(Y_bmm - Y_torch)).item())
    print(f"Triton close to Torch MM: {triton_close}", "max diff:", torch.max(torch.abs(Y_triton - Y_torch)).item())

        
    torch_mean, torch_std = benchmark_function(
        torch_mm_forward, X, dense_weight_T,
        warmup_runs=args.warmup_runs,
        test_runs=args.test_runs
    )

    bmm_mean, bmm_std = benchmark_function(
        kronecker_bmm_reference, X, K_bmm, pattern,
        warmup_runs=args.warmup_runs,
        test_runs=args.test_runs
    )

    triton_mean, triton_std = benchmark_function(
        ks_triton, X_T, K_bmm_T, pattern, 'BSL',
        warmup_runs=args.warmup_runs,
        test_runs=args.test_runs
    )

    
    speedup_bmm = torch_mean / bmm_mean 
    speedup_triton = torch_mean / triton_mean
    theoretical_speedup = dense_weight.numel() / ksl.get_weights_size()

    results = {
        'patterns': patterns,
        'dim_in': N,
        'dim_out': M,
        'batch_size': B,
        'batch_size_last': True,
        'dtype': 'float16',
        'algo': 'kernel',
        'warmup_runs': args.warmup_runs,
        'test_runs': args.test_runs,
        'torch_time_ms': {
            'mean': torch_mean,
            'std': torch_std
        },
        'bmm_time_ms': {
            'mean': bmm_mean,
            'std': bmm_std
        },
        'triton_time_ms': {
            'mean': triton_mean,
            'std': triton_std
        },
        'speedup_bmm': speedup_bmm,
        'speedup_triton': speedup_triton,
        # 'speedup_ksl': speedup_ksl,
        'theoretical_speedup': theoretical_speedup,
    }
    # Print results
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Torch MM:              {torch_mean:.3f} ± {torch_std:.3f} ms")
    print(f"BMM:                   {bmm_mean:.3f} ± {bmm_std:.3f} ms")
    print(f"Triton:              {triton_mean:.3f} ± {triton_std:.3f} ms")
    print(f"Speedup BMM:           {speedup_bmm:.3f}x")
    print(f"Speedup Triton:        {speedup_triton:.3f}x")
    print(f"Theoretical Speedup:   {theoretical_speedup:.3f}x")
