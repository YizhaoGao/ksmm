#!/usr/bin/env python3
"""
Script to extract specific weights from SafeTensors format and save to PyTorch .pth file.

Usage:
    python export_weights.py --input model.safetensors --layer model._orig_mod.vlm_with_expert.lm_expert.layers.0.mlp.gate_proj.weight --output weights.pth

Example:
    python export_weights.py --input ckpt/smolvla/model.safetensors --layer model._orig_mod.vlm_with_expert.lm_expert.layers.0.mlp.gate_proj.weight --output ckpt/gate_proj_weights0.pth
"""

import argparse
import torch
import os
from safetensors import safe_open
from typing import Optional


def extract_weights_from_safetensors(
    safetensors_path: str, 
    layer_name: str, 
    output_path: str,
    device: str = "cpu"
) -> None:
    """
    Extract specific weights from a SafeTensors file and save to PyTorch .pth format.
    
    Args:
        safetensors_path: Path to the input SafeTensors file
        layer_name: Name of the layer/weight to extract (e.g., "model._orig_mod.vlm_with_expert.lm_expert.layers.0.mlp.gate_proj.weight")
        output_path: Path to save the extracted weights as .pth file
        device: Device to load the tensor on (default: "cpu")
    """
    
    # Check if input file exists
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"SafeTensors file not found: {safetensors_path}")
    
    # Load the SafeTensors file
    print(f"Loading SafeTensors file: {safetensors_path}")
    
    try:
        with safe_open(safetensors_path, framework="pt", device=device) as f:
            # List all available keys for debugging
            keys = f.keys()
            print(f"Total number of tensors in file: {len(keys)}")
            
            # Check if the requested layer exists
            if layer_name not in keys:
                print(f"Layer '{layer_name}' not found in the SafeTensors file.")
                print("\nAvailable layers (showing first 20):")
                for i, key in enumerate(sorted(keys)):
                    if i < 20:
                        print(f"  {key}")
                    else:
                        print(f"  ... and {len(keys) - 20} more layers")
                        break
                
                # Try to find similar layer names
                similar_layers = [key for key in keys if any(part in key for part in layer_name.split('.'))]
                if similar_layers:
                    print(f"\nSimilar layer names found:")
                    for layer in similar_layers[:10]:  # Show first 10 similar layers
                        print(f"  {layer}")
                
                raise KeyError(f"Layer '{layer_name}' not found")
            
            # Extract the specific tensor
            print(f"Extracting layer: {layer_name}")
            tensor = f.get_tensor(layer_name)
            print(f"Tensor shape: {tensor.shape}")
            print(f"Tensor dtype: {tensor.dtype}")
            print(f"Tensor device: {tensor.device}")
            
    except Exception as e:
        print(f"Error loading SafeTensors file: {e}")
        raise
    
    # Save to PyTorch .pth format
    print(f"Saving weights to: {output_path}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the tensor
    torch.save(tensor, output_path)
    
    # Verify the saved file
    loaded_tensor = torch.load(output_path, map_location=device)
    print(f"Verification: Loaded tensor shape: {loaded_tensor.shape}")
    print(f"Successfully saved weights to {output_path}")


def list_layers(safetensors_path: str, filter_pattern: Optional[str] = None) -> None:
    """
    List all layers in a SafeTensors file, optionally filtering by pattern.
    
    Args:
        safetensors_path: Path to the SafeTensors file
        filter_pattern: Optional pattern to filter layer names
    """
    
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"SafeTensors file not found: {safetensors_path}")
    
    print(f"Listing layers in: {safetensors_path}")
    
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        keys = f.keys()
        
        if filter_pattern:
            keys = [key for key in keys if filter_pattern in key]
            print(f"Found {len(keys)} layers matching pattern '{filter_pattern}':")
        else:
            print(f"Found {len(keys)} layers:")
        
        for key in sorted(keys):
            tensor_info = f.get_tensor(key)
            print(f"  {key}: {tensor_info.shape} ({tensor_info.dtype})")


def main():
    parser = argparse.ArgumentParser(
        description="Extract specific weights from SafeTensors format and save to PyTorch .pth file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract specific layer weights
  python export_weights.py --input model.safetensors --layer model._orig_mod.vlm_with_expert.lm_expert.layers.0.mlp.gate_proj.weight --output gate_proj.pth
  
  # List all layers in the file
  python export_weights.py --input model.safetensors --list
  
  # List layers matching a pattern
  python export_weights.py --input model.safetensors --list --filter gate_proj
        """
    )
    
    parser.add_argument(
        "--input", "-i", 
        required=True,
        help="Path to input SafeTensors file"
    )
    
    parser.add_argument(
        "--layer", "-l",
        help="Name of the layer/weight to extract"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Path to output .pth file"
    )
    
    parser.add_argument(
        "--device", "-d",
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Device to load tensors on (default: cpu)"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available layers in the SafeTensors file"
    )
    
    parser.add_argument(
        "--filter", "-f",
        help="Filter pattern for listing layers (use with --list)"
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    try:
        if args.list:
            # List layers mode
            list_layers(args.input, args.filter)
        else:
            # Extract weights mode
            if not args.layer:
                print("Error: --layer argument is required when not using --list")
                parser.print_help()
                return 1
            
            if not args.output:
                # Generate default output filename based on layer name
                layer_basename = args.layer.split('.')[-1]  # Get the last part of the layer name
                args.output = f"{layer_basename}.pth"
                print(f"No output path specified, using: {args.output}")
            
            extract_weights_from_safetensors(
                args.input, 
                args.layer, 
                args.output,
                device
            )
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
