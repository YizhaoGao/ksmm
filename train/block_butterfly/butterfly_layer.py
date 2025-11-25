import torch
import torch.nn as nn

class BlockwiseButterfly(nn.Module):
    """
    Forward-only block-wise butterfly layer.
    Each butterfly factor mixes pairs of block channels with 2x2 block matrices.

    Args:
        data_dim: input dimension D
        level: butterfly level (0-based)
        block_size: block size C (each non-zero is CxC)
        share_weights: share weights across positions within each group
        weight_init: 'identity' or 'random'
    """
    def __init__(self, data_dim, level, block_size=1, share_weights=False, weight_init='identity'):
        super().__init__()
        self.data_dim = data_dim
        self.level = level
        self.block_size = block_size
        self.share_weights = share_weights

        # number of butterfly groups
        self.groups = 2 ** level
        denom = (2 ** (level + 1)) * block_size
        
        # Calculate padded dimension if needed
        if data_dim % denom != 0:
            self.padded_dim = ((data_dim + denom - 1) // denom) * denom
            self.needs_padding = True
        else:
            self.padded_dim = data_dim
            self.needs_padding = False
        
        self.nb = self.padded_dim // denom  # block positions per half-channel

        # parameters: shape (groups, nb, 2, 2, C, C)
        if share_weights:
            w_shape = (self.groups, 1, 2, 2, block_size, block_size)
        else:
            w_shape = (self.groups, self.nb, 2, 2, block_size, block_size)

        w = torch.empty(w_shape)
        if weight_init == 'identity':
            w.zero_()
            for g in range(w.shape[0]):
                for n in range(w.shape[1]):
                    w[g, n, 0, 0] = torch.eye(block_size)
                    w[g, n, 1, 1] = torch.eye(block_size)
        elif weight_init == 'random':
            torch.nn.init.normal_(w, std=0.02)
        else:
            raise NotImplementedError

        self.w = nn.Parameter(w)

    def forward(self, x):
        """
        x: (batch, data_dim)
        returns: (batch, data_dim)
        """
        b, D = x.shape
        assert D == self.data_dim, f"Input dimension {D} doesn't match expected {self.data_dim}"
        
        # Pad input if necessary
        if self.needs_padding:
            pad_size = self.padded_dim - D
            x_padded = torch.cat([x, torch.zeros(b, pad_size, device=x.device, dtype=x.dtype)], dim=1)
        else:
            x_padded = x
        
        g, C, nb = self.groups, self.block_size, self.nb
        x_padded = x_padded.view(b, g, 2, nb, C)

        w = self.w
        if self.share_weights:
            w = w.expand(g, nb, 2, 2, C, C)

        # Apply 2x2 block mixing: y = W * x
        # x: b,g,c,n,q; w: g,n,r,c,p,q -> y: b,g,r,n,p
        y = torch.einsum('bgcnq,gnrcpq->bgrnp', x_padded, w)
        y = y.reshape(b, self.padded_dim)
        
        # Remove padding from output if it was added
        if self.needs_padding:
            y = y[:, :D]
        
        return y