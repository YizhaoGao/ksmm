# KS Linear Chai```bash
python train_with_random.py \
    --weight_path ckpt/gate_proj_weights0.pth \
    --rank 4 \
    --impl bmm \
    --batch_size 64 \
    --num_epochs 50 \
    --lr 1e-3 \
    --output_dir ./trained_models
```g

This directory contains scripts for training Kronecker-Sparse (KS) linear chains to approximate dense weight matrices.

## Files

- `train_with_random.py` - Main training script
- `example_usage.py` - Example demonstrating how to use the training script
- `test_module.ipynb` - Jupyter notebook for interactive testing

## Quick Start

### 1. Basic Training

Train a KS chain to approximate an existing weight file:

```bash
python train_with_random.py \
    --weight_path ckpt/gate_proj_weights0.pth \
    --rank 4 \
    --batch_size 64 \
    --num_epochs 10 \
    --lr 1e-3 \
    --output_dir ./trained_models
```

### 2. Advanced Training with Custom Parameters

```bash
python train_with_random.py \
    --weight_path ckpt/gate_proj_weights0.pth \
    --rank 8 \
    --impl triton \
    --batch_size 32 \
    --num_epochs 100 \
    --lr 5e-4 \
    --weight_decay 1e-5 \
    --optimizer adamw \
    --scheduler cosine \
    --max_grad_norm 1.0 \
    --log_grad_norm \
    --num_samples 10000 \
    --val_split 0.2 \
    --dtype float16 \
    --device cuda \
    --save_interval 10 \
    --verbose
```

### 3. Run Example

```bash
python example_usage.py
```

## Command Line Arguments

### Required Arguments
- `--weight_path`: Path to the .pth file containing the dense weight matrix

### Model Parameters
- `--rank`: Rank for low-rank decomposition in butterfly chain (default: 4)
- `--bs_last`: Use batch-size-last layout instead of batch-size-first
- `--impl`: Implementation choice - triton or bmm (default: bmm)

### Training Parameters
- `--batch_size`: Batch size for training (default: 32)
- `--num_epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 1e-3)
- `--weight_decay`: Weight decay for optimizer (default: 1e-4)
- `--optimizer`: Optimizer type - adam, sgd, adamw (default: adamw)

### Learning Rate Scheduler Parameters
- `--scheduler`: Learning rate scheduler type - plateau, cosine, step, none (default: plateau)
- `--scheduler_patience`: Patience for ReduceLROnPlateau scheduler (default: 10)
- `--scheduler_factor`: Factor for ReduceLROnPlateau scheduler (default: 0.5)
- `--step_size`: Step size for StepLR scheduler (default: 30)
- `--gamma`: Gamma for StepLR scheduler (default: 0.1)

### Gradient Monitoring
- `--max_grad_norm`: Maximum gradient norm for clipping (default: 1.0)
- `--log_grad_norm`: Log gradient norms during training

### Data Generation Parameters
- `--num_samples`: Number of random samples to generate for training (default: 10000)
- `--val_split`: Validation split ratio (default: 0.2)

### Output Parameters
- `--output_dir`: Directory to save trained models (default: ./trained_models)
- `--save_prefix`: Prefix for saved model files (default: ks_chain)
- `--save_interval`: Save model every N epochs (default: 10)

### Device and Precision
- `--device`: Device to use for training (default: cuda)
- `--dtype`: Data type for training - float16, float32 (default: float16)

### Logging
- `--log_interval`: Log training progress every N batches (default: 100)
- `--verbose`: Enable verbose logging

## How It Works

1. **Load Dense Weight**: The script loads a dense weight matrix from a .pth file
2. **Create KS Chain**: Uses `create_butterfly_chain()` to create a KS linear chain that can approximate the dense matrix
3. **Generate Training Data**: Creates random input data and computes target outputs using the original dense weight
4. **Training Loop**: Trains the KS chain using MSE loss to match the target outputs
5. **Save Results**: Saves trained models, checkpoints, and compression statistics

## Output Files

The training script saves:
- `{save_prefix}_best.pth` - Best model (lowest validation loss)
- `{save_prefix}_final.pth` - Final model after all epochs
- `{save_prefix}_epoch_{N}.pth` - Periodic checkpoints
- `{save_prefix}_training_log.json` - Training progress log

Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Scheduler state dict (if using scheduler)
- Training and validation losses
- Gradient norm (if logged)
- Learning rate at that epoch
- Patterns used in the KS chain
- Implementation choice (triton or bmm)
- Compression statistics

## Compression Statistics

The script automatically calculates and reports:
- Number of parameters in dense vs sparse representation
- Compression ratio (how many times smaller)
- Memory savings percentage

## Example Results

For a typical 1024x512 dense matrix with rank=4:
- Dense parameters: ~524k
- Sparse parameters: ~65k  
- Compression ratio: ~8x
- Memory savings: ~87.5%

## Tips for Better Results

1. **Rank Selection**: Higher rank = better approximation but less compression
2. **Implementation Choice**: Use `bmm` for better compatibility, `triton` for potential performance gains
3. **Learning Rate**: Start with 1e-3, reduce if loss oscillates
4. **Batch Size**: Larger batches can be more stable but use more memory
5. **Number of Samples**: More training samples generally improve results
6. **Epochs**: Monitor validation loss to avoid overfitting
7. **Scheduler**: Use `plateau` scheduler for adaptive LR based on validation loss
8. **Gradient Clipping**: Use `--max_grad_norm 1.0` to prevent gradient explosion
9. **Gradient Monitoring**: Enable `--log_grad_norm` to track training stability

## Loading Trained Models

```python
import torch
from ksmm_triton.ksmm_module import KSLinearTriton

# Load checkpoint
checkpoint = torch.load('trained_models/ks_chain_best.pth')

# Recreate model
ks_chain = KSLinearTriton(
    patterns=checkpoint['patterns'],
    dtype=torch.float16,
    device='cuda',
    impl=checkpoint.get('impl', 'bmm')  # Use saved implementation or default to bmm
)
ks_chain.load_state_dict(checkpoint['model_state_dict'])

# Use for inference
output = ks_chain(input_tensor)
```
