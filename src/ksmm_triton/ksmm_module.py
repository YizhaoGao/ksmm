import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union

from .ksmm_triton_tc import ks_triton


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

    # Step 2: Batched Matrix Multiply (Ỹ = X̃K̃^T)
    # K_bmm has shape (ad, b, c). We need its transpose for the matmul.
    K_bmm_T = K_bmm.transpose(-1, -2)  # Shape (ad, c, b)
    Y_perm = torch.bmm(X_perm, K_bmm_T)

    # Step 3: Permute Output Y back (equivalent to ỸP^T)
    Y_bsf = Y_perm.transpose(0, 1).reshape(B, a, d, b).transpose(-1, -2).reshape(B, a * b * d)
    return Y_bsf


class KSLinearTriton(nn.Module):
    """
    Kronecker-Sparse Linear layer with configurable implementation.
    
    A high-level PyTorch module that implements a chain of Kronecker-sparse 
    matrix multiplications using either Triton kernels or BMM reference implementation.
    This module can be used as a drop-in replacement for torch.nn.Linear with improved 
    performance for structured sparse patterns.
    
    Args:
        patterns (List[Tuple[int, int, int, int]]): List of (a, b, c, d) patterns 
            defining the Kronecker-sparse structure for each layer in the chain.
        weights (Optional[List[torch.Tensor]]): Optional pre-initialized weights 
            for each pattern. If None, weights are randomly initialized.
        bias (bool): Whether to include bias terms. Default: True.
        dtype (torch.dtype): Data type for weights and computations. Default: torch.float16.
        bs_last (bool): Whether to use batch-size-last layout (BSL) instead of 
            batch-size-first (BSF). Default: False.
        device (Union[str, torch.device]): Device to place the module on. Default: 'cuda'.
        impl (str): Implementation choice - 'triton' or 'bmm'. Default: 'bmm'.
    """
    
    def __init__(
        self,
        patterns: List[Tuple[int, int, int, int]],
        weights: Optional[List[torch.Tensor]] = None,
        bias: bool = True,
        dtype: torch.dtype = torch.float16,
        bs_last: bool = False,
        device: Union[str, torch.device] = 'cuda',
        impl: str = 'bmm'
    ):
        super().__init__()
        
        # Validate inputs
        if not patterns:
            raise ValueError("At least one pattern must be provided")
        
        for i, pattern in enumerate(patterns):
            if len(pattern) != 4:
                raise ValueError(f"Pattern {i} must have exactly 4 elements (a, b, c, d)")
            a, b, c, d = pattern
            if any(x <= 0 for x in pattern):
                raise ValueError(f"All pattern values must be positive, got {pattern}")
        
        # Validate chain compatibility
        for i in range(len(patterns) - 1):
            current_out = patterns[i][0] * patterns[i][1] * patterns[i][3]  # a * b * d
            next_in = patterns[i + 1][0] * patterns[i + 1][2] * patterns[i + 1][3]  # a * c * d
            if current_out != next_in:
                raise ValueError(
                    f"Pattern chain incompatible: pattern {i} output size {current_out} "
                    f"doesn't match pattern {i+1} input size {next_in}"
                )
        
        self.patterns = patterns
        self.num_layers = len(patterns)
        self.dtype = dtype
        self.bs_last = bs_last
        self.device = device
        self.impl = impl
        
        # Validate implementation choice
        if impl not in ['triton', 'bmm']:
            raise ValueError(f"Implementation must be 'triton' or 'bmm', got '{impl}'")
        
        # Calculate input and output dimensions
        self.in_features = patterns[0][0] * patterns[0][2] * patterns[0][3]  # a * c * d
        self.out_features = patterns[-1][0] * patterns[-1][1] * patterns[-1][3]  # a * b * d
        
        # Initialize weights
        self._init_weights(weights)
        
        # Initialize bias if requested
        if bias:
            self._init_bias()
        else:
            self.register_parameter("bias", None)

        self.impl = impl

    
    def _init_weights(self, weights: Optional[List[torch.Tensor]]):
        """Initialize weight parameters for each layer in the chain."""
        if weights is not None and len(weights) != self.num_layers:
            raise ValueError(f"Number of weight tensors ({len(weights)}) must match number of patterns ({self.num_layers})")
        
        weight_params = []
        
        for i, pattern in enumerate(self.patterns):
            a, b, c, d = pattern
            
            # Calculate proper scaling for initialization
            fan_in = a * c * d
            scaling = 1.0 / math.sqrt(fan_in)
            
            if weights is not None and weights[i] is not None:
                # Use provided weights
                weight_tensor = weights[i]
                if weight_tensor.shape != (a, b, c, d):
                    raise ValueError(
                        f"Weight tensor {i} has shape {weight_tensor.shape}, "
                        f"expected {(a, b, c, d)}"
                    )
                # Convert to BMM format: (a, b, c, d) -> (a*d, c, b)
                weight_bmm = weight_tensor.permute(0, 3, 2, 1).reshape(a * d, c, b)
            else:
                # Random initialization in BMM format using Kaiming uniform
                weight_bmm = torch.empty(
                    (a * d, c, b),
                    dtype=self.dtype,
                    device=self.device
                )
                nn.init.kaiming_uniform_(weight_bmm, a=math.sqrt(5))
            
            weight_params.append(nn.Parameter(weight_bmm))
        
        self.weights = nn.ParameterList(weight_params)
    
    def _init_bias(self):
        """Initialize bias parameter."""
        bound = 1.0 / math.sqrt(self.in_features)
        
        if self.bs_last:
            # Batch-size-last: bias shape (out_features, 1)
            bias_shape = (self.out_features, 1)
        else:
            # Batch-size-first: bias shape (out_features,)
            bias_shape = (self.out_features,)
        
        self.bias = nn.Parameter(
            torch.empty(*bias_shape, device=self.device, dtype=self.dtype)
            .uniform_(-bound, bound)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Kronecker-sparse chain.
        
        Args:
            x (torch.Tensor): Input tensor. 
                - For BSF layout: shape (batch_size, in_features)
                - For BSL layout: shape (in_features, batch_size)
        
        Returns:
            torch.Tensor: Output tensor.
                - For BSF layout: shape (batch_size, out_features) 
                - For BSL layout: shape (out_features, batch_size)
        """
        # Validate input shape
        if self.bs_last:
            expected_shape = (self.in_features, -1)  # (in_features, batch_size)
            if x.shape[0] != self.in_features:
                raise ValueError(
                    f"Input tensor first dimension {x.shape[0]} doesn't match "
                    f"expected in_features {self.in_features} for BSL layout"
                )
        else:
            expected_shape = (-1, self.in_features)  # (batch_size, in_features)
            if x.shape[-1] != self.in_features:
                raise ValueError(
                    f"Input tensor last dimension {x.shape[-1]} doesn't match "
                    f"expected in_features {self.in_features} for BSF layout"
                )
        
        # Process through the chain of Kronecker-sparse layers
        output = x
        layout = 'BSL' if self.bs_last else 'BSF'
        
        for i, (weight, pattern) in enumerate(zip(self.weights, self.patterns)):
            a, b, c, d = pattern
            
            if a == 1 and d == 1:
                # Special case: (1, b, c, 1) pattern means we can use a simple F.linear
                # weight is in BMM format (a*d, c, b) = (1, c, b)
                # We need to transpose it to (b, c) for F.linear
                linear_weight = weight.squeeze(0).t()  # (1, c, b) -> (c, b) -> (b, c)
                
                if self.bs_last:
                    # BSL layout: input is (features, batch_size)
                    output = F.linear(output.t(), linear_weight).t()
                else:
                    # BSF layout: input is (batch_size, features)
                    output = F.linear(output, linear_weight)
            else:
                # Choose implementation based on self.impl
                if self.impl == 'triton':
                    output = ks_triton(output, weight, pattern, layout=layout)
                elif self.impl == 'bmm':
                    if self.bs_last:
                        # Convert BSL to BSF for BMM reference implementation
                        output_bsf = output.t()  # (features, batch) -> (batch, features)
                        output_bsf = kronecker_bmm_reference(output_bsf, weight, pattern)
                        output = output_bsf.t()  # Convert back to BSL
                    else:
                        # Direct BSF processing
                        output = kronecker_bmm_reference(output, weight, pattern)
                else:
                    raise ValueError(f"Unknown implementation: {self.impl}")
        
        # Add bias if present
        if self.bias is not None:
            if self.bs_last:
                # BSL: bias shape is (out_features, 1)
                output = output + self.bias
            else:
                # BSF: bias shape is (out_features,)
                output = output + self.bias
        
        return output
    
    def get_dense_product(self) -> torch.Tensor:
        """
        Compute the equivalent dense weight matrix for the entire chain.
        
        This is useful for verification and comparison purposes, but should
        not be used in performance-critical code paths.
        
        Returns:
            torch.Tensor: Dense weight matrix of shape (out_features, in_features)
        """
        # Convert each K_bmm weight back to dense format
        dense_weights = []
        
        for weight, pattern in zip(self.weights, self.patterns):
            a, b, c, d = pattern
            
            # Convert from BMM format (a*d, c, b) back to (a, b, c, d)
            weight_abcd = weight.reshape(a, d, c, b).permute(0, 3, 2, 1)
            
            # Convert to dense matrix
            dense_matrix = self._abcd_to_dense(weight_abcd, pattern)
            dense_weights.append(dense_matrix)
        
        # Multiply all dense matrices in the chain
        result = dense_weights[0]
        for dense_weight in dense_weights[1:]:
            result = torch.matmul(dense_weight, result)
        
        return result
    
    def _abcd_to_dense(self, weight_abcd: torch.Tensor, pattern: Tuple[int, int, int, int]) -> torch.Tensor:
        """Convert weight tensor from (a, b, c, d) format to dense matrix."""
        a, b, c, d = pattern
        
        # Create block diagonal structure
        device = weight_abcd.device
        dtype = weight_abcd.dtype
        
        dense_matrix = torch.zeros(a * b * d, a * c * d, dtype=dtype, device=device)
        
        for i in range(a):
            for j in range(d):
                block_idx = i * d + j
                row_start = i * b * d + j * b
                row_end = row_start + b
                col_start = i * c * d + j * c  
                col_end = col_start + c
                
                dense_matrix[row_start:row_end, col_start:col_end] = weight_abcd[i, :, :, j]
        
        return dense_matrix
    
    def get_weights_size(self) -> int:
        """
        Return the total number of parameters in the weight tensors.
        
        Returns:
            int: Total number of weight parameters
        """
        total_size = 0
        for pattern in self.patterns:
            a, b, c, d = pattern
            total_size += a * b * c * d
        return total_size
    
    def get_theoretical_speedup(self) -> float:
        """
        Calculate the theoretical speedup compared to a dense linear layer.
        
        Returns:
            float: Theoretical speedup factor
        """
        dense_params = self.in_features * self.out_features
        sparse_params = self.get_weights_size()
        return dense_params / sparse_params
    
    def get_parameter_savings(self) -> dict:
        """
        Calculate detailed parameter savings information.
        
        Returns:
            dict: Dictionary containing parameter counts and savings ratios
        """
        dense_params = self.in_features * self.out_features
        sparse_params = self.get_weights_size()
        
        return {
            'dense_parameters': dense_params,
            'sparse_parameters': sparse_params,
            'parameter_reduction_ratio': sparse_params / dense_params,
            'parameter_savings_factor': dense_params / sparse_params,
            'memory_savings_percent': (1 - sparse_params / dense_params) * 100
        }
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'bias={self.bias is not None}, patterns={self.patterns}, '
            f'dtype={self.dtype}, bs_last={self.bs_last}, impl={self.impl}'
        )


def create_chain_from_dense_decomposition(
    dense_weight: torch.Tensor,
    patterns: List[Tuple[int, int, int, int]], 
    **kwargs
) -> KSLinearTriton:
    """
    Create a KSLinearTriton module by decomposing a dense weight matrix.
    
    This is a utility function for approximating existing dense layers
    with Kronecker-sparse structures.
    
    Args:
        dense_weight (torch.Tensor): Dense weight matrix to approximate
        patterns (List[Tuple[int, int, int, int]]): Kronecker-sparse patterns
        **kwargs: Additional arguments for KSLinearTriton constructor
    
    Returns:
        KSLinearTriton: Initialized module with approximated weights
    """
    # This is a placeholder - actual decomposition would require
    # sophisticated algorithms like alternating least squares
    # For now, just create with random initialization
    return KSLinearTriton(patterns=patterns, weights=None, **kwargs)


# Example usage and convenience functions
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


def create_butterfly_chain(
    shape: List[int],
    rank: int = 2,
    **kwargs
) -> KSLinearTriton:
    """
    Create a KSLinearTriton module using butterfly decomposition for rectangular matrices.
    
    For an input shape [a, b], this function:
    1. Finds the larger dimension and ensures it's a power of 2
    2. Creates a butterfly decomposition for that dimension
    3. Uses low-rank matrices to handle rectangular transformations
    
    The rectangular adjustments use two consecutive patterns [1, A, rank, 1] -> [1, rank, B, 1]
    instead of a single dense pattern [1, A, B, 1], saving parameters while maintaining
    representational capacity.
    
    Args:
        shape (List[int]): Input shape [rows, cols] or [out_features, in_features]
        rank (int): Rank for low-rank decomposition of rectangular transformations. Default: 2.
        **kwargs: Additional arguments for KSLinearTriton constructor
    
    Returns:
        KSLinearTriton: Module with butterfly + low-rank pattern chain
        
    Example:
        # For a 128x64 matrix (128 output, 64 input features) with rank=4
        layer = create_butterfly_chain([128, 64], rank=4)
        # Creates: 64 -> 4 -> 128 (low-rank expansion) -> 128 (butterfly) -> 128
    """
    if len(shape) != 2:
        raise ValueError("Shape must be a list of exactly 2 integers [rows, cols]")
    
    out_features, in_features = shape
    
    if out_features <= 0 or in_features <= 0:
        raise ValueError("Both dimensions must be positive")
    
    if rank <= 0:
        raise ValueError("Rank must be a positive integer")
    
    # Find the larger dimension and check if it's a power of 2
    larger_dim = max(out_features, in_features)
    smaller_dim = min(out_features, in_features)
    
    # Check if larger_dim is a power of 2
    if larger_dim & (larger_dim - 1) != 0:
        # Round up to next power of 2
        n = (larger_dim - 1).bit_length()
        power_of_2_dim = 2 ** n
        print(f"Warning: Larger dimension {larger_dim} is not a power of 2. "
              f"Using {power_of_2_dim} (2^{n}) for butterfly decomposition.")
    else:
        power_of_2_dim = larger_dim
        n = power_of_2_dim.bit_length() - 1
    
    # Generate butterfly patterns for the power-of-2 dimension
    butterfly_patterns = create_butterfly_patterns(n)
    
    patterns = []
    
    if out_features >= in_features:
        # Case 1: out_features >= in_features
        # Chain: in_features -> power_of_2_dim (low-rank expansion) -> power_of_2_dim (butterfly) -> out_features (low-rank compression)
        
        if in_features != power_of_2_dim:
            # Need an initial expansion pattern: in_features -> power_of_2_dim
            # Use low-rank decomposition: [1, rank, in_features, 1] -> [1, power_of_2_dim, rank, 1]
            # This gives us in_features -> rank -> power_of_2_dim with specified rank
            expansion_pattern_1 = (1, rank, in_features, 1)  # in_features -> rank
            expansion_pattern_2 = (1, power_of_2_dim, rank, 1)  # rank -> power_of_2_dim
            patterns.extend([expansion_pattern_1, expansion_pattern_2])
        
        # Add butterfly patterns (power_of_2_dim -> power_of_2_dim)
        patterns.extend(butterfly_patterns)
        
        if power_of_2_dim != out_features:
            # Need a final compression pattern: power_of_2_dim -> out_features
            # Use low-rank decomposition: [1, rank, power_of_2_dim, 1] -> [1, out_features, rank, 1]
            # This gives us power_of_2_dim -> rank -> out_features with specified rank
            compression_pattern_1 = (1, rank, power_of_2_dim, 1)  # power_of_2_dim -> rank
            compression_pattern_2 = (1, out_features, rank, 1)  # rank -> out_features
            patterns.extend([compression_pattern_1, compression_pattern_2])
    
    else:
        # Case 2: in_features > out_features
        # Chain: in_features -> power_of_2_dim (low-rank compression) -> power_of_2_dim (butterfly) -> out_features (low-rank compression)
        
        if in_features != power_of_2_dim:
            # Need initial compression: in_features -> power_of_2_dim
            # Use low-rank decomposition: [1, rank, in_features, 1] -> [1, power_of_2_dim, rank, 1]
            compression_pattern_1 = (1, rank, in_features, 1)  # in_features -> rank
            compression_pattern_2 = (1, power_of_2_dim, rank, 1)  # rank -> power_of_2_dim
            patterns.extend([compression_pattern_1, compression_pattern_2])
        
        # Add butterfly patterns (power_of_2_dim -> power_of_2_dim)
        patterns.extend(butterfly_patterns)
        
        if power_of_2_dim != out_features:
            # Need final compression: power_of_2_dim -> out_features
            # Use low-rank decomposition: [1, rank, power_of_2_dim, 1] -> [1, out_features, rank, 1]
            final_pattern_1 = (1, rank, power_of_2_dim, 1)  # power_of_2_dim -> rank
            final_pattern_2 = (1, out_features, rank, 1)  # rank -> out_features
            patterns.extend([final_pattern_1, final_pattern_2])
    
    # Validate the pattern chain
    if patterns:
        # Check first pattern input matches in_features
        first_pattern = patterns[0]
        expected_input = first_pattern[0] * first_pattern[2] * first_pattern[3]
        if expected_input != in_features:
            raise ValueError(f"Pattern chain input {expected_input} doesn't match in_features {in_features}")
        
        # Check last pattern output matches out_features
        last_pattern = patterns[-1]
        expected_output = last_pattern[0] * last_pattern[1] * last_pattern[3]
        if expected_output != out_features:
            raise ValueError(f"Pattern chain output {expected_output} doesn't match out_features {out_features}")
    
    return KSLinearTriton(patterns=patterns, **kwargs)


def create_simple_ks_layer(
    in_features: int,
    out_features: int, 
    pattern: Tuple[int, int, int, int],
    **kwargs
) -> KSLinearTriton:
    """
    Create a simple single-layer KSLinearTriton module.
    
    Args:
        in_features (int): Input feature dimension
        out_features (int): Output feature dimension  
        pattern (Tuple[int, int, int, int]): Single (a, b, c, d) pattern
        **kwargs: Additional arguments for KSLinearTriton constructor
    
    Returns:
        KSLinearTriton: Single-layer module
    """
    a, b, c, d = pattern
    
    # Validate dimensions match
    expected_in = a * c * d
    expected_out = a * b * d
    
    if in_features != expected_in:
        raise ValueError(f"in_features {in_features} doesn't match pattern input size {expected_in}")
    if out_features != expected_out:
        raise ValueError(f"out_features {out_features} doesn't match pattern output size {expected_out}")
    
    return KSLinearTriton(patterns=[pattern], **kwargs)
