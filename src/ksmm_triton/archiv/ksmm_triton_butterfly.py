from ksmm_triton_tc import benchmark_function, torch_mm_forward
from typing import List, Tuple
import subprocess
import torch
import triton
import triton.language as tl
from ksmm_py.layer.kronecker_sparse.interface import KSLinear
import argparse


def create_butterfly_patterns(n: int) -> List[Tuple[int, int, int, int]]:
    """
    Generates the chain of Kronecker-Sparse patterns for a butterfly
    decomposition of a 2^n x 2^n square matrix.

    A butterfly matrix W of size 2^n x 2^n can be factored into n
    Kronecker-Sparse matrices: W = K_1 * K_2 * ... * K_n.

    The ℓ-th factor, K_ℓ, has a pattern corresponding to (a, b, c, d) where:
    a = 2**(ℓ-1)
    b = 2
    c = 2
    d = 2**(n-ℓ)

    Args:
        n: The power of 2 defining the matrix dimension (2^n x 2^n).
           Must be a positive integer.

    Returns:
        A list of n tuples, where each tuple is an (a, b, c, d) pattern.
        The list is ordered for direct use in libraries like ksmm,
        meaning patterns[0] corresponds to the right-most matrix factor (K_n)
        and patterns[-1] corresponds to the left-most factor (K_1).
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("Input 'n' must be a positive integer.")

    patterns = []
    # The loop variable 'l' corresponds to the index in the matrix product K_1 * ... * K_n
    for l in range(1, n + 1):
        a = 2**(l - 1)
        b = 2
        c = 2
        d = 2**(n - l)
        pattern = (a, b, c, d)
        patterns.append(pattern)

    # Reverse the list to match the convention where the first pattern in the list
    # corresponds to the right-most matrix in the product.
    # W = K_1 * K_2 * ... * K_n
    # patterns = [pattern_for_Kn, ..., pattern_for_K2, pattern_for_K1]
    return patterns[::-1]



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
                    {'BLOCK_SIZE_B': 2, 'BLOCK_SIZE_C': 2, 'BLOCK_SIZE_BATCH': BBATCH},
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



def ks_chain_triton(
    X: torch.Tensor,
    weight_list: List[torch.Tensor],
    patterns: List[Tuple[int, int, int, int]],
    layout: str = 'BSF',
    print_log: bool = False,
) -> torch.Tensor:
    """
    Apply a chain of Kronecker‐Sparse factors using the Triton fused kernel.
    Args:
        X:            Input tensor, shape (B, N) if BSF or (N, B) if BSL.
        weight_list:  List of factor tensors of shape (a, b, c, d).
        patterns:     List of (a, b, c, d) tuples, in the same order as weight_list.
        layout:       'BSF' (batch‐size‐first) or 'BSL' (batch‐size‐last).
    Returns:
        Output tensor after applying all factors, in the same layout as X.
    """
    # Validate
    assert len(weight_list) == len(patterns), "weights and patterns must match in length"
    # Convert BSF -> BSL once
    is_bsf = (layout == 'BSF')
    if is_bsf:
        # X: (B, N) -> (N, B)
        X_bsl = X.transpose(0, 1).contiguous()
    elif layout == 'BSL':
        X_bsl = X
    else:
        raise ValueError("layout must be 'BSF' or 'BSL'")
    # Sequentially apply each Kronecker‐sparse factor

    for (a, b, c, d), K in zip(patterns, weight_list):
        if print_log:
            print(f"Applying Triton Kronecker-Sparse factor: {a, b, c, d}")
        X_bsl = ks_triton(X_bsl, K, (a, b, c, d), layout='BSL')
    if is_bsf:
        return X_bsl.transpose(0, 1).contiguous()
    return X_bsl


if __name__ == "__main__":
    ## parse batch size
    parser = argparse.ArgumentParser(description="Benchmark Triton implementation of Kronecker-Sparse matrix chain.")
    parser.add_argument("--batch_size", type=int, default=25088, help="Batch size for input tensor")
    args = parser.parse_args()

    batch_size = args.batch_size

    for n in range(11, 12):
        N = 2**n
        patterns =  create_butterfly_patterns(n)
        print(f"---------Butterfly patterns for size {2**n} n={n}: {patterns}----------\n")
        # Run the benchmark script with the generated patterns
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
        weight_triton = []
        for i, (a, b, c, d) in enumerate(patterns):
            weight_triton.append(ksl.factors[i].view(a * d, c, b).transpose(-1, -2).contiguous())
        # Benchmark the Triton implementation
        print("Benchmarking Triton implementation...")

        input_dim, output_dim = dense_weight.shape
        X = torch.randn(batch_size, N, dtype=torch.float16, device='cuda')
        X_T = X.T.contiguous()

        ## Compile and autotune the Triton kernel
        ks_chain_triton(X_T, weight_triton, patterns, layout='BSL', print_log=True)


        triton_mean_time_ms, triton_std_time_ms = benchmark_function(
            ks_chain_triton, X_T, weight_triton, patterns, 'BSL',
            warmup_runs=10,
            test_runs=100
        )

        torch_mean_time_ms, torch_std_time_ms = benchmark_function(
            torch_mm_forward, X, dense_weight_T,
            warmup_runs=10,
            test_runs=100
        )

        # Print results
        print(f"n={n}, Triton mean time: {triton_mean_time_ms:.4f} ms, std: {triton_std_time_ms:.4f} ms")
        print(f"n={n}, Torch mean time: {torch_mean_time_ms:.4f} ms, std: {torch_std_time_ms:.4f} ms")
        print(f"Speedup: {torch_mean_time_ms / triton_mean_time_ms:.4f}x")
        

        print(f"----------Benchmark for n={n} completed.----------\n\n")