import torch
import triton
import triton.language as tl

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



configs=[
    triton.Config({'BLOCK_SIZE_B': BB, 'BLOCK_SIZE_C': BC, 'BLOCK_SIZE_BATCH': BBATCH}, num_warps=num_warps, num_stages=num_stages)
    for BB in [16, 32, 64]
    for BC in [16, 32, 64]
    for BBATCH in [16, 32, 64, 128]
    for num_warps in [2, 4, 8]
    for num_stages in [2, 3, 4]
]


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

    # 3. ACCUMULATOR: Initialize output tile
    # This is the output-stationary part. Accumulator stays in registers.
    accumulator = tl.zeros((BLOCK_SIZE_BATCH, BLOCK_SIZE_B), dtype=tl.float32)

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
    for c_i in range(0, c, BLOCK_SIZE_C):
        for b_j in range(0, b, BLOCK_SIZE_B):
            # Calculate the current tile's offsets
            b_offsets = b_j + tl.arange(0, BLOCK_SIZE_B)
            c_offsets = c_i + tl.arange(0, BLOCK_SIZE_C)

            # --- Load X tile (fused permutation) ---
            # Calculate strided column indices for X
            x_feat_offsets = x_col_base + (c_i + c_offsets) * d
            # Create pointers for the X tile
            x_ptrs = X_ptr + (batch_offsets[:, None] * stride_X_batch + x_feat_offsets[None, :] * stride_X_feat)

            # Load X tile with boundary checks
            x_mask = (batch_offsets[:, None] < B) & ((c_i + c_offsets)[None, :] < c)
            x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

            # --- Load K tile ---
            # K is stored contiguously for each (i,j) block, so this is a simple load.
            k_ptrs = k_ptr + ((c_i + c_offsets)[:, None] * stride_K_c + b_offsets[None, :] * stride_K_b)
            k_mask = ((c_i + c_offsets)[:, None] < c) & (b_offsets[None, :] < b)
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
    """
    PyTorch wrapper for the Fused Kronecker-Sparse kernel.
    
    Args:
        X (torch.Tensor): Input tensor. Shape (B, N) for BSF or (N, B) for BSL.
        K_bmm (torch.Tensor): Pre-permuted weight tensor of shape (a*d, b, c),
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
    assert K_bmm.shape == (a * d, b, c), "K_bmm shape is incorrect"

    # The kernel expects K^T relative to the BMM baseline's K.
    # BMM computes X_perm @ K_bmm.T, so we need K_bmm.T which is (ad, c, b)
    K_T = K_bmm.transpose(1, 2).contiguous()

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
        Y_bsl, X_bsl, K_T,
        B,
        # Strides for BSL layout
        X_bsl.stride(1), X_bsl.stride(0),
        Y_bsl.stride(1), Y_bsl.stride(0),
        K_T.stride(0), K_T.stride(1), K_T.stride(2),
        # Tuning parameters (BLOCK_SIZES)
        a, b, c, d,
    )

    # 4. Post-processing
    if layout == 'BSF':
        # Transpose back to BSF if that was the original layout
        return Y_bsl.transpose(0, 1).contiguous()
    else:
        return Y_bsl

# --- Verification ---
if __name__ == "__main__":
    # Use a sample pattern from the paper's experiments
    # ViT-S/16 square matrix: (1, 192, 48, 2) and (2, 48, 192, 1)
    # Let's use a simpler, more balanced pattern for a clear test
    pattern = (a, b, c, d) = (4, 2, 2, 1)
    
    # Input dimensions
    N = a * c * d
    M = a * b * d
    B = 128 # Batch size
    
    print(f"Pattern (a,b,c,d): {pattern}")
    print(f"Input shape (B, N): ({B}, {N})")
    print(f"Output shape (B, M): ({B}, {M})")
    
    # Create random tensors on GPU
    X = torch.randn((B, N), device='cuda', dtype=torch.float16)
    
    # Create the pre-permuted K tensor, as used by BMM baseline
    # Shape: (ad, b, c)
    K_bmm = torch.randn((a * d, b, c), device='cuda', dtype=torch.float16)
    print("K_bmm shape:", K_bmm.shape)

    # --- Run reference implementation ---
    print("\nRunning reference BMM implementation...")
    Y_reference = kronecker_bmm_reference(X, K_bmm, pattern)
    
    # --- Run Triton implementation ---
    print("Running Triton fused kernel...")
    X_triton = X.T
    Y_triton = ks_triton(X_triton, K_bmm, pattern, layout='BSL')
    Y_triton = Y_triton.T  # Transpose back to BSF if needed

    # --- Compare results ---
    print("\nVerifying results...")
    are_close = torch.allclose(Y_reference, Y_triton, atol=1e-2, rtol=1e-3)
    print(f"Results are close: {are_close}")

    if are_close:
        print("✅ Verification successful!")
        
        # Optional: Benchmark
        try:
            print("\n--- Benchmarking (ms) ---")
            ms_ref = triton.testing.do_bench(lambda: kronecker_bmm_reference(X, K_bmm, pattern))
            ms_triton = triton.testing.do_bench(lambda: ks_triton(X_triton, K_bmm, pattern, layout='BSL'))
            print(f"Reference BMM: {ms_ref:.4f} ms")
            print(f"Triton Fused:  {ms_triton:.4f} ms")
            speedup = ms_ref / ms_triton
            print(f"Speedup: {speedup:.2f}x")
        except Exception as e:
            print(f"Could not run benchmark: {e}")

    else:
        print("❌ Verification failed!")
        print("Max absolute difference:", torch.max(torch.abs(Y_reference - Y_triton)).item())