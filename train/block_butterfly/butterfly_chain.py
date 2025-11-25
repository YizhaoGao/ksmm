import torch
import torch.nn as nn
import math
from butterfly_layer import BlockwiseButterfly


class ButterflyChain(nn.Module):
    """
    Complete butterfly chain module that applies all levels of butterfly transformations.
    
    For an input dimension D, the butterfly chain consists of log2(D) levels (when block_size=1),
    or log2(D/block_size) levels (when block_size > 1).
    
    Args:
        data_dim: input dimension D
        block_size: block size C (each non-zero block is CxC)
        share_weights: whether to share weights across positions within each level
        weight_init: 'identity' or 'random' - initialization method for all levels
    """
    def __init__(self, data_dim, block_size=1, share_weights=False, weight_init='identity'):
        super().__init__()
        self.data_dim = data_dim
        self.block_size = block_size
        self.share_weights = share_weights
        self.weight_init = weight_init
        
        # Calculate the number of levels needed
        # For butterfly to work, we need D to be divisible by 2^max_level * block_size
        # The maximum useful level is log2(D/block_size)
        if data_dim < block_size:
            raise ValueError(f"data_dim ({data_dim}) must be >= block_size ({block_size})")
        
        # Calculate padded dimension to make it a power of 2 times block_size
        effective_dim = data_dim // block_size
        if effective_dim == 0:
            self.max_level = 0
        else:
            self.max_level = int(math.ceil(math.log2(effective_dim)))
        
        # Ensure we have at least one level
        if self.max_level == 0:
            self.max_level = 1
            
        self.num_levels = self.max_level
        
        # Create butterfly layers for each level
        self.butterfly_layers = nn.ModuleList()
        
        for level in range(self.num_levels):
            layer = BlockwiseButterfly(
                data_dim=data_dim,
                level=level,
                block_size=block_size,
                share_weights=share_weights,
                weight_init=weight_init
            )
            self.butterfly_layers.append(layer)
    
    def forward(self, x):
        """
        Apply the complete butterfly chain transformation.
        
        Args:
            x: input tensor of shape (batch_size, data_dim)
            
        Returns:
            output tensor of shape (batch_size, data_dim)
        """
        batch_size, input_dim = x.shape
        assert input_dim == self.data_dim, f"Input dimension {input_dim} doesn't match expected {self.data_dim}"
        
        # Apply each butterfly level sequentially
        out = x
        for level, butterfly_layer in enumerate(self.butterfly_layers):
            out = butterfly_layer(out)
            
        return out
    
    def get_num_parameters(self):
        """Get the total number of parameters in the butterfly chain."""
        return sum(p.numel() for p in self.parameters())
    
    def get_parameter_info(self):
        """Get detailed information about parameters in each level."""
        info = {
            'total_params': self.get_num_parameters(),
            'num_levels': self.num_levels,
            'data_dim': self.data_dim,
            'block_size': self.block_size,
            'share_weights': self.share_weights,
            'levels': []
        }
        
        for level, layer in enumerate(self.butterfly_layers):
            level_params = sum(p.numel() for p in layer.parameters())
            level_info = {
                'level': level,
                'params': level_params,
                'weight_shape': layer.w.shape,
                'groups': layer.groups,
                'nb': layer.nb,
                'padded_dim': layer.padded_dim,
                'needs_padding': layer.needs_padding
            }
            info['levels'].append(level_info)
        
        return info
    
    def __repr__(self):
        return (f"ButterflyChain(data_dim={self.data_dim}, block_size={self.block_size}, "
                f"num_levels={self.num_levels}, share_weights={self.share_weights}, "
                f"weight_init='{self.weight_init}')")


class EfficientButterflyChain(ButterflyChain):
    """
    Memory-efficient version of ButterflyChain that computes the optimal number of levels
    based on the input dimension and reduces unnecessary computation.
    """
    def __init__(self, data_dim, block_size=1, share_weights=False, weight_init='identity'):
        # Override the level calculation for efficiency
        self.data_dim = data_dim
        self.block_size = block_size
        
        # Calculate the minimum number of levels needed for meaningful transformation
        effective_dim = data_dim
        min_levels_needed = 0
        
        # Find minimum levels where the transformation actually does something useful
        temp_dim = effective_dim
        while temp_dim > block_size:
            denom = (2 ** (min_levels_needed + 1)) * block_size
            if temp_dim >= denom:
                min_levels_needed += 1
                temp_dim = temp_dim // 2
            else:
                break
        
        # Ensure at least one level
        min_levels_needed = max(1, min_levels_needed)
        
        # Initialize parent with calculated levels
        nn.Module.__init__(self)
        self.share_weights = share_weights
        self.weight_init = weight_init
        self.num_levels = min_levels_needed
        self.max_level = min_levels_needed
        
        # Create butterfly layers
        self.butterfly_layers = nn.ModuleList()
        
        for level in range(self.num_levels):
            layer = BlockwiseButterfly(
                data_dim=data_dim,
                level=level,
                block_size=block_size,
                share_weights=share_weights,
                weight_init=weight_init
            )
            self.butterfly_layers.append(layer)