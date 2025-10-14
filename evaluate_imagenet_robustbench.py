#!/usr/bin/env python3
"""
Custom model evaluation script for RobustBench
Evaluates a custom checkpoint against ImageNet using L2 threat model
"""

import torch
from robustbench.eval import benchmark
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from robustbench.model_zoo.architectures.utils_architectures import normalize_model
import argparse
import os
from timm.models import create_model
from timm.models.resnet import ResNet, Bottleneck, _create_resnet

def load_custom_model(checkpoint_path, architecture='resnet50'):
    """Load custom model from checkpoint"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create base model architecture
    if architecture == 'resnet50':
        model = create_model('resnet50', pretrained=False, num_classes=1000)
    elif architecture == 'wide_resnet50_4':
        # Create wide_resnet50_4 using timm's _create_resnet (following model_factory_224.py)
        model_args = dict(block=Bottleneck, layers=(3, 4, 6, 3), base_width=256)
        model = _create_resnet('wide_resnet50_4', pretrained=False, num_classes=1000, **model_args)
    else:
        raise ValueError(f"Architecture {architecture} not supported yet")
    
    # Load state dict - handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("Found 'state_dict' key in checkpoint")
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
        print("Found 'model' key in checkpoint")
    else:
        state_dict = checkpoint
        print("Using checkpoint directly as state_dict")
    
    # Remove module prefix if present
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.' prefix
        else:
            new_key = key
        new_state_dict[new_key] = value
    
    # Load the state dict
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("Loaded checkpoint with strict=True")
    except RuntimeError as e:
        print(f"Warning: Strict loading failed: {e}")
        print("Trying with strict=False...")
        model.load_state_dict(new_state_dict, strict=False)
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Evaluate custom model with RobustBench')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to ImageNet validation dataset')
    parser.add_argument('--threat_model', type=str, default='L2',
                       choices=['L2', 'Linf', 'corruptions'],
                       help='Threat model to evaluate')
    parser.add_argument('--eps', type=float, default=3.0,
                       help='Perturbation budget')
    parser.add_argument('--n_examples', type=int, default=1000,
                       help='Number of examples to evaluate')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--architecture', type=str, default='resnet50',
                       help='Model architecture')
    parser.add_argument('--preprocessing', type=str, default='Res256Crop224',
                       help='Preprocessing to use (Res256Crop224, Res224, etc.)')
    
    args = parser.parse_args()
    
    # Load custom model
    model = load_custom_model(args.checkpoint, args.architecture)
    model = model.eval()
    
    # Add ImageNet normalization
    mu = (0.485, 0.456, 0.406)
    sigma = (0.229, 0.224, 0.225)
    model = normalize_model(model, mu, sigma)
    
    model.eval()
    print(f"Model loaded and set to eval mode")
    
    # Set device and enable multi-GPU if available
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Move model to device first
    model = model.to(device)
    
    # Enable multi-GPU if available
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    else:
        print(f"Using single device: {device}")
    
    # Convert threat model string to enum
    threat_model = ThreatModel(args.threat_model)
    
    print(f"\nStarting evaluation:")
    print(f"  Dataset: ImageNet")
    print(f"  Threat model: {args.threat_model}")
    print(f"  Epsilon: {args.eps}")
    print(f"  Number of examples: {args.n_examples}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Data directory: {args.data_dir}")
    
    # Run evaluation with preprocessing specified
    clean_acc, robust_acc = benchmark(
        model=model,
        dataset=BenchmarkDataset.imagenet,
        threat_model=threat_model,
        eps=args.eps,
        n_examples=args.n_examples,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        device=device,
        preprocessing=args.preprocessing  # Use specified preprocessing
    )
    
    print(f"\n" + "="*50)
    print(f"EVALUATION RESULTS")
    print(f"="*50)
    print(f"Clean accuracy: {clean_acc:.4f} ({clean_acc:.2%})")
    print(f"Robust accuracy ({args.threat_model}, eps={args.eps}): {robust_acc:.4f} ({robust_acc:.2%})")
    print(f"="*50)

if __name__ == '__main__':
    main()
