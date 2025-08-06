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

# block_sizes = [16, 32, 64]  
# batch_sizes = [16, 32, 64] 
block_sizes = [16,]  
batch_sizes = [64,] 

for BB in block_sizes:
    for BC in block_sizes:
        for BBATCH in batch_sizes:
            # Strategy 3: Limit num_warps based on total work per block
            total_work = BB * BC * BBATCH
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
                            {'BLOCK_SIZE_B': BB, 'BLOCK_SIZE_C': BC, 'BLOCK_SIZE_BATCH': BBATCH},
                            num_warps=num_warps,
                            num_stages=num_stages
                        )
                    )

# # Optional: Add some hand-picked configurations based on your problem size
# # You can uncomment and modify these based on your typical (a,b,c,d) patterns
# configs.extend([
#     triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_C': 16, 'BLOCK_SIZE_BATCH': 128}, num_warps=8, num_stages=4),
# ])

print(f"Generated {len(configs)} configurations")


@triton.autotune(
    configs=configs,
    key=['a', 'b', 'c', 'd', 'B'],
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
    
    # Base pointers for the strided access into X and Y
    # These calculations match Equations 1 & 2 from the paper.
    # col_base = i*N/a + j = i*c*d + j
    # row_base = i*M/a + j = i*b*d + j
    x_col_base = i * c * d + j
    y_row_base = i * b * d + j

    # 5. MAIN LOOP: Iterate over the reduction dimension 'c' 
    for b_j in range(0, b, BLOCK_SIZE_B):
        accumulator = tl.zeros((BLOCK_SIZE_BATCH, BLOCK_SIZE_B), dtype=tl.float32)
        for c_i in range(0, c, BLOCK_SIZE_C):
            # Calculate the current tile's offsets
            b_offsets = b_j + tl.arange(0, BLOCK_SIZE_B)
            c_offsets = c_i + tl.arange(0, BLOCK_SIZE_C)

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
            accumulator += tl.dot(x_tile, k_tile)

        # 6. STORE Y TILE: Write result back to global memory (fused permutation)
        # Calculate strided column indices for Y
        y_feat_offsets = y_row_base + b_offsets * d
        # Create pointers for the Y tile
        y_ptrs = Y_ptr + (batch_offsets[:, None] * stride_Y_batch + y_feat_offsets[None, :] * stride_Y_feat)
        # Store Y tile with boundary checks
        y_mask = (batch_offsets[:, None] < B) & (b_offsets[None, :] < b)
        tl.store(y_ptrs, accumulator, mask=y_mask)


def ks_triton(X, K_bmm, pattern, layout='BSF'):
    return KSTritonFunction.apply(X, K_bmm, pattern, layout)


class KSTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, K_bmm, pattern, layout='BSF'):
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
        original_layout = layout
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

        ctx.save_for_backward(X_bsl, K_bmm)
        ctx.pattern = pattern
        ctx.original_layout = original_layout
        
        # 4. Post-processing
        if original_layout == 'BSF':
            # Transpose back to BSF if that was the original layout
            return Y_bsl.transpose(0, 1).contiguous()
        else:
            return Y_bsl

    @staticmethod
    def backward(ctx, dY):
        X_bsl, K_bmm = ctx.saved_tensors
        pattern = ctx.pattern
        original_layout = ctx.original_layout
        a, b, c, d = pattern
        
        if original_layout == 'BSF':
            dY_bsl = dY.transpose(0, 1).contiguous()
        else:
            dY_bsl = dY

        N, B = X_bsl.shape
        M = a * b * d
        
        dX_bsl = torch.zeros_like(X_bsl)
        dK_bmm = torch.zeros_like(K_bmm)

        grid = lambda META: (
            a * d, triton.cdiv(B, META['BLOCK_SIZE_BATCH']), 1
        )

        ks_fused_kernel_bwd[grid](
            dX_bsl, dK_bmm,
            dY_bsl, X_bsl, K_bmm,
            B,
            # Strides for BSL layout
            dX_bsl.stride(1), dX_bsl.stride(0),
            dY_bsl.stride(1), dY_bsl.stride(0),
            dK_bmm.stride(0), dK_bmm.stride(1), dK_bmm.stride(2),
            a, b, c, d,
        )

        # The gradient dK_bmm is computed for a weight of shape (a*d, c, b).
        # The autograd engine expects the gradient to have the same shape as the input tensor.
        # The original K_bmm input to forward had shape (a*d, c, b).
        
        if original_layout == 'BSF':
            # Transpose dX back to BSF if that was the original layout
            dX = dX_bsl.transpose(0, 1).contiguous()
        else:
            dX = dX_bsl

        return dX, dK_bmm, None, None

@triton.autotune(
    configs=configs,
    key=['a', 'b', 'c', 'd', 'B'],
)
@triton.jit
def ks_fused_kernel_bwd(
    # Pointers to gradients
    dX_ptr,
    dK_ptr,
    # Pointers to matrices
    dY_ptr,
    X_ptr,
    K_ptr,
    # Matrix dimensions
    B,
    # Strides for BSL layout
    stride_dX_batch, stride_dX_feat,
    stride_dY_batch, stride_dY_feat,
    stride_dK_tile, stride_dK_c, stride_dK_b,
    a: tl.constexpr,
    b: tl.constexpr,
    c: tl.constexpr,
    d: tl.constexpr,
    # Tile dimensions
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_pid = tl.program_id(axis=1)
    
    i = pid // d
    j = pid % d

    batch_offsets = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    
    x_col_base = i * c * d + j
    y_row_base = i * b * d + j

    k_ptr = K_ptr + pid * stride_dK_tile
    dk_ptr = dK_ptr + pid * stride_dK_tile

    for b_j in range(0, b, BLOCK_SIZE_B):
        b_offsets = b_j + tl.arange(0, BLOCK_SIZE_B)
        y_feat_offsets = y_row_base + b_offsets * d
        dy_ptrs = dY_ptr + (batch_offsets[:, None] * stride_dY_batch + y_feat_offsets[None, :] * stride_dY_feat)
        dy_mask = (batch_offsets[:, None] < B) & (b_offsets[None, :] < b)
        dy_tile = tl.load(dy_ptrs, mask=dy_mask, other=0.0)

        for c_i in range(0, c, BLOCK_SIZE_C):
            c_offsets = c_i + tl.arange(0, BLOCK_SIZE_C)
            
            # Load K for dX calculation
            k_ptrs = k_ptr + (c_offsets[:, None] * stride_dK_c + b_offsets[None, :] * stride_dK_b)
            k_mask = (c_offsets[:, None] < c) & (b_offsets[None, :] < b)
            k_tile = tl.load(k_ptrs, mask=k_mask, other=0.0)

            # Calculate dX: dX = dY @ K (since forward was X @ K^T)
            dx_tile = tl.dot(dy_tile, k_tile)
            x_feat_offsets = x_col_base + c_offsets * d
            dx_ptrs = dX_ptr + (batch_offsets[:, None] * stride_dX_batch + x_feat_offsets[None, :] * stride_dX_feat)
            dx_mask = (batch_offsets[:, None] < B) & (c_offsets[None, :] < c)
            tl.atomic_add(dx_ptrs, dx_tile, mask=dx_mask)

            # Load X for dK calculation
            x_feat_offsets = x_col_base + c_offsets * d
            x_ptrs = X_ptr + (batch_offsets[:, None] * stride_dX_batch + x_feat_offsets[None, :] * stride_dX_feat)
            x_mask = (batch_offsets[:, None] < B) & (c_offsets[None, :] < c)
            x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

            # Calculate dK: dK = X^T @ dY
            dk_tile = tl.dot(x_tile.trans(), dy_tile)
            dk_ptrs = dk_ptr + (c_offsets[:, None] * stride_dK_c + b_offsets[None, :] * stride_dK_b)
            dk_mask = (c_offsets[:, None] < c) & (b_offsets[None, :] < b)
            tl.atomic_add(dk_ptrs, dk_tile, mask=dk_mask)


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


    print("Running KSLinear")
    try:
        Y_ksl = ksl(X_T)
        Y_ksl = Y_ksl.T  # Transpose back to BSF if needed
        
    except Exception as e:
        print(f"KSLinear forward pass failed: {e}")
        Y_ksl = None


    # --- Compare results --- all compare to Y_torch
    print("\nVerifying results...")
    bmm_close = torch.allclose(Y_bmm, Y_torch, rtol=1e-2, atol=1e-2)
    triton_close = torch.allclose(Y_triton, Y_torch, rtol=1e-2, atol=1e-2)
    # ksl_close = torch.allclose(Y_ksl, Y_torch, rtol=1e-2, atol=1e-2)
    print(f"BMM close to Torch MM: {bmm_close}", "max diff:", torch.max(torch.abs(Y_bmm - Y_torch)).item())
    print(f"Triton close to Torch MM: {triton_close}", "max diff:", torch.max(torch.abs(Y_triton - Y_torch)).item())
    # print(f"KSLinear close to Torch MM: {ksl_close}", "max diff:", torch.max(torch.abs(Y_ksl - Y_torch)).item() if Y_ksl is not None else "N/A")

        
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

    try:
        ksl_mean, ksl_std = benchmark_function(
            ksl_forward, ksl, X_T,
            warmup_runs=args.warmup_runs,
            test_runs=args.test_runs
        )
    except Exception as e:
        print(f"KSLinear benchmark failed: {e}")
        ksl_mean, ksl_std = None, None

    
    speedup_bmm = torch_mean / bmm_mean 
    speedup_triton = torch_mean / triton_mean
    speedup_ksl = torch_mean / ksl_mean if ksl_mean is not None else None
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
        'ksl_time_ms': {
            'mean': ksl_mean,
            'std': ksl_std
        } if ksl_mean is not None else None,
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
    print(f"KSLinear :             {ksl_mean:.3f} ± {ksl_std:.3f} ms" if ksl_mean is not None else "KSLinear benchmark failed")
    print(f"Torch MM:              {torch_mean:.3f} ± {torch_std:.3f} ms")
    print(f"BMM:                   {bmm_mean:.3f} ± {bmm_std:.3f} ms")
    print(f"Triton:              {triton_mean:.3f} ± {triton_std:.3f} ms")
    print(f"Speedup BMM:           {speedup_bmm:.3f}x")
    print(f"Speedup Triton:        {speedup_triton:.3f}x")
    if ksl_mean is not None:
        print(f"Theoretical Speedup:   {theoretical_speedup:.3f}x")

    # --- Backward pass verification ---
    print("\nVerifying backward pass...")
    X_ref = X.clone().requires_grad_()
    K_bmm_ref = K_bmm.clone().requires_grad_()
    K_bmm_ref.retain_grad()  # Ensure gradient is retained for non-leaf tensor
    Y_ref = kronecker_bmm_reference(X_ref, K_bmm_ref, pattern)
    
    # Dummy gradient for dY
    dY = torch.randn_like(Y_ref)
    Y_ref.backward(dY)

    X_triton = X.clone().requires_grad_()
    K_bmm_triton = K_bmm_T.clone().requires_grad_()
    K_bmm_triton.retain_grad()  # Ensure gradient is retained for non-leaf tensor
    Y_triton_bw = ks_triton(X_triton, K_bmm_triton, pattern, layout='BSF')
    Y_triton_bw.backward(dY)

    dX_ref = X_ref.grad
    dK_ref = K_bmm_ref.grad

    dX_triton = X_triton.grad
    dK_triton = K_bmm_triton.grad

    print(f"dX_ref shape: {dX_ref.shape}, dX_triton shape: {dX_triton.shape}")
    print(f"dK_ref shape: {dK_ref.shape}, dK_triton shape: {dK_triton.shape}")
    print(f"K_bmm_ref shape: {K_bmm_ref.shape}, K_bmm_triton shape: {K_bmm_triton.shape}")

    if dX_ref is not None and dX_triton is not None:
        dx_close = torch.allclose(dX_ref, dX_triton, rtol=1e-1, atol=1e-1)
        print(f"dX close to reference: {dx_close}", "max diff:", torch.max(torch.abs(dX_ref - dX_triton)).item())
    
    if dK_ref is not None and dK_triton is not None:
        # K_bmm_ref has shape (a*d, c, b), K_bmm_triton has shape (a*d, b, c)
        # So we need to transpose dK_triton to match dK_ref
        dk_close = torch.allclose(dK_ref, dK_triton.transpose(-1, -2).contiguous(), rtol=1e-1, atol=1e-1)
        print(f"dK close to reference: {dk_close}", "max diff:", torch.max(torch.abs(dK_ref - dK_triton.transpose(-1, -2).contiguous())).item())