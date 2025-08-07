"""
KSLinearTriton: Kronecker-Sparse Linear Layers with Triton

This package provides efficient PyTorch modules for Kronecker-sparse linear transformations
using Triton kernels for high performance on CUDA devices.
"""

from .ksmm_module import (
    KSLinearTriton,
    create_simple_ks_layer,
    create_chain_from_dense_decomposition,
    create_butterfly_patterns,
    create_butterfly_chain,
)

from .ksmm_triton_tc import (
    ks_triton,
    KSTritonFunction,
    ks_triton_forward_impl,
    ks_triton_backward,
    kronecker_bmm_reference,
)

__version__ = "1.0.0"

__all__ = [
    # Main module classes
    "KSLinearTriton",
    "create_simple_ks_layer", 
    "create_chain_from_dense_decomposition",
    "create_butterfly_patterns",
    "create_butterfly_chain",
    
    # Low-level functions
    "ks_triton",
    "KSTritonFunction",
    "ks_triton_forward_impl",
    "ks_triton_backward",
    "kronecker_bmm_reference",
]
