#!/usr/bin/env python3
"""
Example usage of the KS Linear Chain training script.

This script demonstrates how to use train_with_random.py to train a KS linear chain
to approximate a dense weight matrix.
"""

import os
import torch
from ksmm_triton.ksmm_module import create_butterfly_chain, KSLinearTriton

def create_example_weight(shape, save_path):
    """Create an example dense weight matrix and save it."""
    print(f"Creating example weight with shape {shape}")
    
    # Create a random dense weight matrix
    weight = torch.randn(shape, dtype=torch.float16)
    
    # Save the weight
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(weight, save_path)
    
    print(f"Saved example weight to: {save_path}")
    return weight

def run_training_example():
    """Run a complete training example."""
    # Define the weight shape (output_features, input_features)
    weight_shape = (1024, 512)  # Example: 1024 output features, 512 input features
    
    # Create example weight
    weight_path = "ckpt/example_weight.pth"
    example_weight = create_example_weight(weight_shape, weight_path)
    
    print(f"\nExample weight statistics:")
    print(f"  Shape: {example_weight.shape}")
    print(f"  Total parameters: {example_weight.numel():,}")
    print(f"  Memory usage: {example_weight.numel() * 2 / 1024**2:.2f} MB (fp16)")
    
    # Show what compression we can expect
    print(f"\nTesting KS chain creation...")
    test_chain = create_butterfly_chain(
        shape=list(weight_shape), 
        rank=4, 
        dtype=torch.float16, 
        device='cpu',  # Use CPU for this test
        impl='bmm'  # Use BMM implementation for testing
    )
    
    sparse_params = test_chain.get_weights_size()
    compression_ratio = example_weight.numel() / sparse_params
    
    print(f"  KS chain patterns: {test_chain.patterns}")
    print(f"  KS chain parameters: {sparse_params:,}")
    print(f"  Expected compression ratio: {compression_ratio:.2f}x")
    print(f"  Expected memory savings: {(1 - sparse_params / example_weight.numel()) * 100:.1f}%")
    
    # Print training command
    print(f"\nTo train the KS chain to approximate this weight, run:")
    print(f"python train_with_random.py \\")
    print(f"    --weight_path {weight_path} \\")
    print(f"    --rank 4 \\")
    print(f"    --impl bmm \\")
    print(f"    --batch_size 64 \\")
    print(f"    --num_epochs 50 \\")
    print(f"    --lr 1e-3 \\")
    print(f"    --num_samples 5000 \\")
    print(f"    --output_dir ./trained_models \\")
    print(f"    --save_prefix example_ks_chain")
    
    print(f"\nAlternatively, with different parameters:")
    print(f"python train_with_random.py \\")
    print(f"    --weight_path {weight_path} \\")
    print(f"    --rank 8 \\")
    print(f"    --impl triton \\")
    print(f"    --batch_size 32 \\")
    print(f"    --num_epochs 100 \\")
    print(f"    --lr 5e-4 \\")
    print(f"    --weight_decay 1e-5 \\")
    print(f"    --optimizer adamw \\")
    print(f"    --num_samples 10000 \\")
    print(f"    --val_split 0.2 \\")
    print(f"    --dtype float16 \\")
    print(f"    --device cuda \\")
    print(f"    --verbose")

def load_and_test_trained_model():
    """Example of how to load and test a trained model."""
    print(f"\n" + "="*60)
    print(f"LOADING AND TESTING TRAINED MODEL")
    print(f"="*60)
    
    # This assumes you have a trained model
    model_path = "./trained_models/example_ks_chain_best.pth"
    
    if not os.path.exists(model_path):
        print(f"Trained model not found at: {model_path}")
        print(f"Run the training script first to create a trained model.")
        return
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Training loss: {checkpoint['train_loss']:.6f}")
    print(f"Validation loss: {checkpoint['val_loss']:.6f}")
    print(f"Patterns: {checkpoint['patterns']}")
    print(f"Compression stats: {checkpoint['compression_stats']}")
    
    # Recreate the model
    # You would need to know the original weight shape and rank used
    # This information could be saved in the checkpoint for convenience
    patterns = checkpoint['patterns']
    impl = checkpoint.get('impl', 'bmm')  # Default to bmm for backward compatibility
    ks_chain = KSLinearTriton(patterns=patterns, dtype=torch.float16, device='cpu', impl=impl)
    ks_chain.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Successfully loaded trained KS chain!")
    
    # Test the model with some random input
    batch_size = 8
    in_features = patterns[0][0] * patterns[0][2] * patterns[0][3]  # Calculate input size
    out_features = patterns[-1][0] * patterns[-1][1] * patterns[-1][3]  # Calculate output size
    
    test_input = torch.randn(batch_size, in_features, dtype=torch.float16)
    
    with torch.no_grad():
        output = ks_chain(test_input)
    
    print(f"Test inference:")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Model successfully processes input!")

if __name__ == "__main__":
    print("KS Linear Chain Training Example")
    print("=" * 50)
    
    # Run the example
    run_training_example()
    
    # Show how to load trained models
    load_and_test_trained_model()
    
    print(f"\n" + "="*60)
    print(f"EXAMPLE COMPLETE")
    print(f"="*60)
    print(f"Next steps:")
    print(f"1. Run the training command shown above")
    print(f"2. Monitor the training progress")
    print(f"3. Use the trained model for inference")
    print(f"4. Experiment with different ranks and parameters")
