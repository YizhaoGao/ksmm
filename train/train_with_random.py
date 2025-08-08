#!/usr/bin/env python3
"""
Training script for KS Linear Chain to approximate a given weight matrix.

This script trains a Kronecker-Sparse (KS) linear chain to approximate a dense weight matrix
using random input data and MSE loss. The target outputs are generated using the original
dense weight matrix.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import json
import time
from pathlib import Path

# Import KS modules
from ksmm_triton.ksmm_module import create_butterfly_chain, KSLinearTriton, create_simple_ks_layer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train KS Linear Chain to approximate dense weights')
    
    # Model parameters
    parser.add_argument('--weight_path', type=str, required=True,
                        help='Path to the .pth file containing the dense weight matrix')
    parser.add_argument('--rank', type=int, default=4,
                        help='Rank for low-rank decomposition in butterfly chain (default: 4)')
    parser.add_argument('--bs_last', action='store_true',
                        help='Use batch-size-last layout instead of batch-size-first')
    parser.add_argument('--impl', type=str, default='bmm', choices=['triton', 'bmm'],
                        help='Implementation choice - triton or bmm (default: bmm)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer (default: 1e-4)')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'sgd', 'adamw'],
                        help='Optimizer type (default: adamw)')
    
    # Learning rate scheduler parameters
    parser.add_argument('--scheduler', type=str, default='plateau', 
                        choices=['plateau', 'cosine', 'step', 'none'],
                        help='Learning rate scheduler type (default: plateau)')
    parser.add_argument('--scheduler_patience', type=int, default=10,
                        help='Patience for ReduceLROnPlateau scheduler (default: 10)')
    parser.add_argument('--scheduler_factor', type=float, default=0.5,
                        help='Factor for ReduceLROnPlateau scheduler (default: 0.5)')
    parser.add_argument('--step_size', type=int, default=30,
                        help='Step size for StepLR scheduler (default: 30)')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Gamma for StepLR scheduler (default: 0.1)')
    
    # Gradient monitoring
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping (default: 1.0)')
    parser.add_argument('--log_grad_norm', action='store_true',
                        help='Log gradient norms during training')
    
    # Data generation parameters
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Number of random samples to generate for training (default: 10000)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./trained_models',
                        help='Directory to save trained models (default: ./trained_models)')
    parser.add_argument('--save_prefix', type=str, default='ks_chain',
                        help='Prefix for saved model files (default: ks_chain)')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save model every N epochs (default: 10)')
    
    # Device and precision
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training (default: cuda)')
    parser.add_argument('--dtype', type=str, default='float32', choices=['float16', 'float32'],
                        help='Data type for training (default: float32)')
    
    # Logging
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Log training progress every N batches (default: 100)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    return parser.parse_args()


def load_dense_weight(weight_path: str, device: str, dtype: torch.dtype):
    """Load dense weight matrix from file."""
    print(f"Loading dense weight from: {weight_path}")
    
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")
    
    weight = torch.load(weight_path, map_location=device)
    
    # Convert to specified dtype
    weight = weight.to(dtype=dtype, device=device)
    
    print(f"Loaded weight shape: {weight.shape}")
    print(f"Weight dtype: {weight.dtype}, device: {weight.device}")
    
    return weight


def generate_random_data(input_size: int, num_samples: int, batch_size: int, 
                        dtype: torch.dtype, device: str, bs_last: bool = False):
    """Generate random input data for training."""
    print(f"Generating {num_samples} random samples with input size {input_size}")
    
    if bs_last:
        # BSL layout: (input_size, batch_size)
        inputs = torch.randn(num_samples, input_size, dtype=dtype, device=device) * 2
    else:
        # BSF layout: (batch_size, input_size)
        inputs = torch.randn(num_samples, input_size, dtype=dtype, device=device) * 2

    dataset = TensorDataset(inputs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


def compute_target_outputs(dense_weight: torch.Tensor, input_batch: torch.Tensor, bs_last: bool = False):
    """Compute target outputs using the original dense weight."""
    if bs_last:
        # BSL layout: input is (input_size, batch_size), weight is (out_size, in_size)
        # We need to transpose input for matrix multiplication
        targets = torch.matmul(dense_weight, input_batch)
    else:
        # BSF layout: input is (batch_size, input_size), weight is (out_size, in_size)
        # Use F.linear which expects weight as (out_features, in_features)
        targets = torch.nn.functional.linear(input_batch, dense_weight)
    
    return targets


def create_optimizer(model: nn.Module, optimizer_type: str, lr: float, weight_decay: float):
    """Create optimizer for the model."""
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    return optimizer


def create_scheduler(optimizer: optim.Optimizer, scheduler_type: str, **kwargs):
    """Create learning rate scheduler."""
    if scheduler_type.lower() == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 10),
            verbose=True
        )
    elif scheduler_type.lower() == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 50),
            eta_min=kwargs.get('eta_min', 1e-6)
        )
    elif scheduler_type.lower() == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_type.lower() == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_type}")
    
    return scheduler


def compute_grad_norm(model: nn.Module):
    """Compute the gradient norm of the model parameters."""
    total_norm = 0.0
    weight_grad_norms = []
    weight_names = []
    
    # For KS chain, compute individual weight gradient norms with names
    if hasattr(model, 'weights'):
        for i, weight in enumerate(model.weights):
            weight_name = f"weights.{i}"
            weight_names.append(weight_name)
            if weight.grad is not None:
                weight_norm = weight.grad.data.norm(2).item()
                weight_grad_norms.append(weight_norm)
                total_norm += weight_norm ** 2
            else:
                weight_grad_norms.append(0.0)
    else:
        # Fallback for other models using named_parameters
        for name, param in model.named_parameters():
            weight_names.append(name)
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                weight_grad_norms.append(param_norm)
                total_norm += param_norm ** 2
            else:
                weight_grad_norms.append(0.0)
    
    total_norm = total_norm ** (1. / 2)
    return total_norm, weight_grad_norms, weight_names


def calculate_compression_stats(dense_weight: torch.Tensor, ks_chain: KSLinearTriton):
    """Calculate compression statistics."""
    dense_params = dense_weight.numel()
    sparse_params = ks_chain.get_weights_size()
    compression_ratio = dense_params / sparse_params
    
    return {
        'dense_parameters': dense_params,
        'sparse_parameters': sparse_params,
        'compression_ratio': compression_ratio,
        'memory_savings_percent': (1 - sparse_params / dense_params) * 100
    }


def evaluate_model(model: KSLinearTriton, val_loader: DataLoader, dense_weight: torch.Tensor, 
                  criterion: nn.Module, device: str, bs_last: bool = False):
    """Evaluate the model on validation data."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (inputs,) in enumerate(val_loader):
            inputs = inputs.to(device)
            
            # Generate targets using dense weight
            targets = compute_target_outputs(dense_weight, inputs, bs_last)
            
            # Forward pass through KS chain
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def save_model_and_stats(model: KSLinearTriton, optimizer: optim.Optimizer, scheduler, epoch: int, 
                        train_loss: float, val_loss: float, compression_stats: dict,
                        output_dir: str, save_prefix: str, grad_norm: float = None, 
                        weight_grad_norms: list = None, weight_names: list = None):
    """Save model checkpoint and training statistics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'grad_norm': grad_norm,
        'weight_grad_norms': weight_grad_norms,
        'weight_names': weight_names,
        'patterns': model.patterns,
        'impl': model.impl,
        'compression_stats': compression_stats,
    }
    
    checkpoint_path = os.path.join(output_dir, f"{save_prefix}_epoch_{epoch}.pth")
    torch.save(checkpoint, checkpoint_path)
    
    # Save training log
    log_data = {
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'grad_norm': grad_norm,
        'weight_grad_norms': weight_grad_norms,
        'weight_names': weight_names,
        'learning_rate': optimizer.param_groups[0]['lr'],
        'compression_stats': compression_stats
    }
    
    log_path = os.path.join(output_dir, f"{save_prefix}_training_log.json")
    
    # Append to existing log or create new
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            logs = json.load(f)
        logs.append(log_data)
    else:
        logs = [log_data]
    
    with open(log_path, 'w') as f:
        json.dump(logs, f, indent=2)
    
    print(f"Saved checkpoint: {checkpoint_path}")


def save_loss_curves(train_losses: list, val_losses: list, epochs: list, rank: int, output_dir: str, save_prefix: str):
    """Save loss curves to JSON and CSV files for analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to JSON
    loss_data = {
        'rank': rank,
        'epochs': epochs,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    
    json_path = os.path.join(output_dir, f"{save_prefix}_rank_{rank}_loss_curves.json")
    with open(json_path, 'w') as f:
        json.dump(loss_data, f, indent=2)
    
    # Save to CSV for easy plotting
    csv_path = os.path.join(output_dir, f"{save_prefix}_rank_{rank}_loss_curves.csv")
    import csv
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'rank'])
        for epoch, train_loss, val_loss in zip(epochs, train_losses, val_losses):
            writer.writerow([epoch, train_loss, val_loss, rank])
    
    print(f"Saved loss curves: {json_path} and {csv_path}")


def main():
    args = parse_args()
    
    # Set up device and dtype
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16 if args.dtype == 'float16' else torch.float32
    
    print(f"Using device: {device}, dtype: {dtype}")
    print(f"Arguments: {vars(args)}")
    
    # Load dense weight
    dense_weight = load_dense_weight(args.weight_path, device, dtype)
    out_features, in_features = dense_weight.shape
    
    # # Create KS linear chain
    # print(f"\nCreating KS linear chain for shape [{out_features}, {in_features}] with rank {args.rank}")
    # print(f"Using implementation: {args.impl}")
    ks_chain = create_butterfly_chain(
        shape=[out_features, in_features],
        rank=args.rank,
        dtype=dtype,
        device=device,
        bs_last=args.bs_last,
        impl=args.impl
    )
    print(ks_chain)


    # # Use a dense layer to verify the training script
    # # Results shows that the loss goes to 0 very quickly
    # ks_chain = create_simple_ks_layer(
    #     in_features=in_features,
    #     out_features=out_features,
    #     pattern=[1, out_features, in_features, 1], 
    #     dtype=dtype,
    #     device=device,
    #     bs_last=args.bs_last,
    #     impl=args.impl
    # )


    
    print(f"Created KS chain with patterns: {ks_chain.patterns}")
    
    # Calculate compression statistics
    compression_stats = calculate_compression_stats(dense_weight, ks_chain)
    print(f"\nCompression Statistics:")
    print(f"  Dense parameters: {compression_stats['dense_parameters']:,}")
    print(f"  Sparse parameters: {compression_stats['sparse_parameters']:,}")
    print(f"  Compression ratio: {compression_stats['compression_ratio']:.2f}x")
    print(f"  Memory savings: {compression_stats['memory_savings_percent']:.1f}%")
    
    # Generate training data
    print(f"\nGenerating training data...")
    num_train_samples = int(args.num_samples * (1 - args.val_split))
    num_val_samples = args.num_samples - num_train_samples
    
    train_loader = generate_random_data(
        in_features, num_train_samples, args.batch_size, dtype, device, args.bs_last
    )
    val_loader = generate_random_data(
        in_features, num_val_samples, args.batch_size, dtype, device, args.bs_last
    )
    
    print(f"Training samples: {num_train_samples}, Validation samples: {num_val_samples}")
    
    # Set up training
    criterion = nn.MSELoss()
    optimizer = create_optimizer(ks_chain, args.optimizer, args.lr, args.weight_decay)
    
    # Create learning rate scheduler
    scheduler = create_scheduler(
        optimizer, 
        args.scheduler,
        patience=args.scheduler_patience,
        factor=args.scheduler_factor,
        step_size=args.step_size,
        gamma=args.gamma,
        T_max=args.num_epochs
    )
    
    print(f"\nStarting training for {args.num_epochs} epochs...")
    print(f"Optimizer: {args.optimizer}, LR: {args.lr}, Weight decay: {args.weight_decay}")
    print(f"Scheduler: {args.scheduler}")
    print(f"Implementation: {args.impl}")
    if args.log_grad_norm or args.max_grad_norm < float('inf'):
        print(f"Gradient monitoring: max_norm={args.max_grad_norm}, log_norm={args.log_grad_norm}")
    
    # Training loop
    best_val_loss = float('inf')
    
    # Track loss curves for analysis
    train_losses = []
    val_losses = []
    epochs = []
    
    for epoch in range(1, args.num_epochs + 1):
        ks_chain.train()
        total_train_loss = 0.0
        total_grad_norm = 0.0
        total_weight_grad_norms = []
        weight_names = []
        num_train_batches = 0
        
        # Training phase
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}")
        
        for batch_idx, (inputs,) in enumerate(progress_bar):
            inputs = inputs.to(device)
            
            # Generate targets using dense weight
            targets = compute_target_outputs(dense_weight, inputs, args.bs_last)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = ks_chain(inputs)
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Compute gradient norm before clipping
            grad_norm, weight_grad_norms, batch_weight_names = compute_grad_norm(ks_chain)
            
            # Store weight names on first batch
            if len(weight_names) == 0:
                weight_names = batch_weight_names
            
            # Gradient clipping
            if args.max_grad_norm < float('inf'):
                torch.nn.utils.clip_grad_norm_(ks_chain.parameters(), args.max_grad_norm)
            
            optimizer.step()
            
            # Accumulate metrics
            total_train_loss += loss.item()
            total_grad_norm += grad_norm
            
            # Accumulate weight gradient norms
            if len(total_weight_grad_norms) == 0:
                total_weight_grad_norms = [0.0] * len(weight_grad_norms)
            for i, weight_grad_norm in enumerate(weight_grad_norms):
                total_weight_grad_norms[i] += weight_grad_norm
            
            num_train_batches += 1
            
            # Update progress bar
            if batch_idx % args.log_interval == 0:
                avg_loss = total_train_loss / num_train_batches
                avg_grad_norm = total_grad_norm / num_train_batches
                avg_weight_grad_norms = [wgn / num_train_batches for wgn in total_weight_grad_norms]
                current_lr = optimizer.param_groups[0]['lr']
                
                postfix = {'loss': f'{avg_loss:.6f}', 'lr': f'{current_lr:.2e}'}
                if args.log_grad_norm:
                    postfix['grad_norm'] = f'{avg_grad_norm:.4f}'
                    # Create compact weight grad norm display
                    weight_grad_strs = [f"{name}:{wgn:.4f}" for name, wgn in zip(weight_names, avg_weight_grad_norms)]
                    postfix['weight_grads'] = f"[{', '.join(weight_grad_strs)}]"
                progress_bar.set_postfix(postfix)
        
        # Calculate average training metrics
        avg_train_loss = total_train_loss / num_train_batches
        avg_grad_norm = total_grad_norm / num_train_batches
        avg_weight_grad_norms = [wgn / num_train_batches for wgn in total_weight_grad_norms]
        current_lr = optimizer.param_groups[0]['lr']
        
        # Validation phase
        avg_val_loss = evaluate_model(ks_chain, val_loader, dense_weight, criterion, device, args.bs_last)
        
        # Store loss curves
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        epochs.append(epoch)
        
        # Update learning rate scheduler
        if scheduler:
            if args.scheduler.lower() == 'plateau':
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        
        # Print epoch results
        log_str = f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}"
        if args.log_grad_norm:
            log_str += f" | Grad Norm: {avg_grad_norm:.4f}"
            # Create readable weight grad norm display
            weight_grad_pairs = [f"{name}:{wgn:.4f}" for name, wgn in zip(weight_names, avg_weight_grad_norms)]
            log_str += f" | Weight Grad Norms: [{', '.join(weight_grad_pairs)}]"
        print(log_str)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_model_and_stats(
                ks_chain, optimizer, scheduler, epoch, avg_train_loss, avg_val_loss,
                compression_stats, args.output_dir, f"{args.save_prefix}_best", 
                avg_grad_norm, avg_weight_grad_norms, weight_names
            )
        
        # Save periodic checkpoint
        if epoch % args.save_interval == 0:
            save_model_and_stats(
                ks_chain, optimizer, scheduler, epoch, avg_train_loss, avg_val_loss,
                compression_stats, args.output_dir, args.save_prefix, 
                avg_grad_norm, avg_weight_grad_norms, weight_names
            )
    
    # Save final model
    save_model_and_stats(
        ks_chain, optimizer, scheduler, args.num_epochs, avg_train_loss, avg_val_loss,
        compression_stats, args.output_dir, f"{args.save_prefix}_final", 
        avg_grad_norm, avg_weight_grad_norms, weight_names
    )
    
    # Save loss curves for rank comparison study
    save_loss_curves(train_losses, val_losses, epochs, args.rank, args.output_dir, args.save_prefix)
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final training loss: {avg_train_loss:.6f}")
    print(f"Final validation loss: {avg_val_loss:.6f}")
    print(f"Final learning rate: {optimizer.param_groups[0]['lr']:.2e}")
    if args.log_grad_norm:
        print(f"Final gradient norm: {avg_grad_norm:.4f}")
        # Print final weight gradient norms with names
        final_weight_grad_pairs = [f"{name}:{wgn:.4f}" for name, wgn in zip(weight_names, avg_weight_grad_norms)]
        print(f"Final weight gradient norms: [{', '.join(final_weight_grad_pairs)}]")
    print(f"Models saved to: {args.output_dir}")


if __name__ == "__main__":
    main()