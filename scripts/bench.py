import argparse
import torch
import time
import json
from typing import List, Tuple
from ksmm_py.layer.kronecker_sparse.interface import KSLinear


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


def calculate_dimensions(patterns: List[Tuple[int, int, int, int]]) -> Tuple[int, int]:
    """
    Calculate input and output dimensions from patterns.
    patterns[0] corresponds to rightmost factor (K_L)
    patterns[-1] corresponds to leftmost factor (K_1)
    """
    # Input dimension: a_L * c_L * d_L (from rightmost factor)
    dim_in = patterns[0][0] * patterns[0][2] * patterns[0][3]
    
    # Output dimension: a_1 * b_1 * d_1 (from leftmost factor)  
    dim_out = patterns[-1][0] * patterns[-1][1] * patterns[-1][3]
    
    return dim_in, dim_out


def create_dense_equivalent(ksl: KSLinear) -> torch.Tensor:
    """
    Create the dense weight matrix equivalent to the KSLinear factorization.
    """
    # Get dense product of all factors
    dense_weight = ksl.get_dense_product()
    return dense_weight


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


def torch_mm_forward(x, weight_transposed):
    """Standard torch matrix multiplication."""
    return torch.matmul(x, weight_transposed)


def ksl_forward(ksl, x):
    """KSLinear forward pass."""
    return ksl(x)


def main():
    parser = argparse.ArgumentParser(description='Benchmark KSLinear vs torch.mm speed')
    parser.add_argument('--patterns', type=str, required=True,
                       help='Patterns as string, e.g., "[(6,64,64,1),(1,768,192,2)]"')
    parser.add_argument('--batch_size', type=int, default=25088,
                       help='Batch size for input tensor')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help='Device to run on')
    parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'float32'],
                       help='Data type to use')
    parser.add_argument('--batch_size_last', type=bool, default=True, help='Whether batch size is the last dimension of input tensor')
    parser.add_argument('--algo', type=str, default='kernel', 
                       choices=['dense', 'kernel', 'bmm', 'bsr', 'einsum', 'sparse'],
                       help='KSLinear algorithm to use')
    parser.add_argument('--warmup_runs', type=int, default=10,
                       help='Number of warmup runs')
    parser.add_argument('--test_runs', type=int, default=100,
                       help='Number of test runs for averaging')
    parser.add_argument('--output_file', type=str, default="output.json",
                       help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Parse inputs
    patterns = parse_patterns(args.patterns)
    dim_in, dim_out = calculate_dimensions(patterns)
    
    # Override dimensions if provided (for validation)
    print(f"Calculated dimensions: dim_in={dim_in}, dim_out={dim_out}")
    
    dtype = torch.float16 if args.dtype == 'float16' else torch.float32
    device = args.device
    
    # Create input tensor
    if args.batch_size_last:
        x_shape = (dim_in, args.batch_size)
        x = torch.randn(x_shape, dtype=dtype, device=device)
        x_torch = x.T
    else:
        x_shape = (args.batch_size, dim_in)
        x = torch.randn(x_shape, dtype=dtype, device=device)
        x_torch = x
    
    print(f"Input tensor shape: {x.shape}")
    print(f"Patterns: {patterns}")
    print(f"Algorithm: {args.algo}")
    print(f"Device: {device}, dtype: {dtype}")
    
    # Create KSLinear layer
    try:
        ksl = KSLinear(
            patterns=patterns,
            weights=None,  # Let it initialize randomly
            algo=args.algo,
            dtype=dtype,
            bs_last=args.batch_size_last,
            device=device
        )
        print(f"KSLinear created successfully")
        
        # Get dense equivalent for torch.mm comparison
        dense_weight = create_dense_equivalent(ksl)
        dense_weight = dense_weight.T
        print(f"Dense weight: {dense_weight}")
        print(f"Dense weight shape: {dense_weight.shape}")

        # Verify correctness first
        ksl_output = ksl(x)
        torch_output = torch.matmul(x_torch, dense_weight)
        if args.batch_size_last:
            torch_output = torch_output.T  # Transpose to match KSLinear output shape
        
        print(f"KSLinear output shape: {ksl_output.shape}")
        print(f"Torch output shape: {torch_output.shape}")
        
        # Check if outputs match
        if torch.allclose(ksl_output, torch_output, atol=1e-2, rtol=1e-2):
            print("✓ Outputs match within tolerance")
        else:
            print("⚠ Outputs do not match - there may be an issue with the implementation")
            max_diff = torch.max(torch.abs(ksl_output - torch_output))
            print(f"Max difference: {max_diff}")
        
        # Benchmark KSLinear
        print(f"\nBenchmarking KSLinear ({args.algo})...")
        ksl_mean, ksl_std = benchmark_function(
            ksl_forward, ksl, x,
            warmup_runs=args.warmup_runs,
            test_runs=args.test_runs
        )
        
        # Benchmark torch.mm
        print(f"Benchmarking torch.mm...")

        torch_mean, torch_std = benchmark_function(
            torch_mm_forward, x_torch, dense_weight,
            warmup_runs=args.warmup_runs,
            test_runs=args.test_runs
        )
        
        # Calculate speedup
        speedup = torch_mean / ksl_mean
        theoretical_speedup = dense_weight.numel() / ksl.get_weights_size()

        # Results
        results = {
            'patterns': patterns,
            'dim_in': dim_in,
            'dim_out': dim_out,
            'batch_size': args.batch_size,
            'batch_size_last': args.batch_size_last,
            'dtype': args.dtype,
            'algo': args.algo,
            'warmup_runs': args.warmup_runs,
            'test_runs': args.test_runs,
            'ksl_time_ms': {
                'mean': ksl_mean,
                'std': ksl_std
            },
            'torch_time_ms': {
                'mean': torch_mean,
                'std': torch_std
            },
            'speedup': speedup,
            'theoretical_speedup': theoretical_speedup,
        }
        
        # Print results
        print(f"\n{'='*60}")
        print(f"BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"KSLinear ({args.algo}): {ksl_mean:.3f} ± {ksl_std:.3f} ms")
        print(f"Torch MM:              {torch_mean:.3f} ± {torch_std:.3f} ms")
        print(f"Speedup:               {speedup:.3f}x")
        print(f"Theoretical Speedup:   {theoretical_speedup:.3f}x")
        print(f"KSLinear is {'FASTER' if speedup > 1.0 else 'SLOWER'}")
        
        # Save results if requested
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()