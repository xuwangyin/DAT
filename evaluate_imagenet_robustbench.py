#!/usr/bin/env python3
"""
Custom model evaluation script for RobustBench
Evaluates a custom checkpoint against ImageNet using L2 threat model
"""

import argparse
import math
import os
import sys

# Use the local timm fork that adds normalize_input + layernorm knobs.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pytorch-image-models"))

import robustbench
import torch
from robustbench.eval import benchmark
from robustbench.model_zoo.architectures.utils_architectures import (
    normalize_model,
)
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from timm.models import create_model
from timm.models.resnet import Bottleneck, _create_resnet
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from rebm.training.modeling import load_checkpoint
from rebm.training.utils_architecture import create_convnext_model


def get_preprocessing_function(img_size=224, crop_pct=0.875):
    """Create preprocessing function for arbitrary image size.

    Args:
        img_size: Target image size (default: 224)
        crop_pct: Crop percentage (default: 0.875)

    Returns:
        Preprocessing function compatible with RobustBench
    """
    scale_size = int(math.floor(img_size / crop_pct))

    def preprocess(x):
        """Preprocessing that resizes, center crops, and converts to tensor."""
        return transforms.Compose([
            transforms.Resize(scale_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ])(x)

    return preprocess


def load_custom_model(checkpoint_path, architecture="resnet50"):
    """Load custom model from checkpoint"""
    print(f"Loading checkpoint from: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Create base model architecture
    if architecture == "resnet50":
        model = create_model("resnet50", pretrained=False, num_classes=1000)
    elif architecture == "wide_resnet50_4":
        # Create wide_resnet50_4 using timm's _create_resnet (following model_factory_224.py)
        model_args = dict(block=Bottleneck, layers=(3, 4, 6, 3), base_width=256)
        model = _create_resnet(
            "wide_resnet50_4", pretrained=False, num_classes=1000, **model_args
        )
    elif architecture == "convnext_large":
        # Use shared ConvNeXt creation function for consistency with training
        model = create_convnext_model(
            model_type="convnext_large",
            num_classes=1000,
            normalize_input=False,
            use_layernorm=True,
            use_convstem=True,  # AT ConvNeXt models use ConvStem
        )
    else:
        raise ValueError(f"Architecture {architecture} not supported yet")

    # Load checkpoint using shared utility (handles prefixes + DDP)
    return load_checkpoint(model, checkpoint_path, weights_only=False)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate custom model with RobustBench"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to ImageNet validation dataset",
    )
    parser.add_argument(
        "--threat_model",
        type=str,
        default="L2",
        choices=["L2", "Linf", "corruptions"],
        help="Threat model to evaluate",
    )
    parser.add_argument(
        "--eps", type=float, default=3.0, help="Perturbation budget"
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=1000,
        help="Number of examples to evaluate",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="resnet50",
        help="Model architecture",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Image size for preprocessing (e.g., 224 for 224x224, 256 for 256x256). Default: 224",
    )
    parser.add_argument(
        "--input_normalization",
        action="store_true",
        default=False,
        help="Add ImageNet normalization wrapper (default: False, should never be used for DAT models)",
    )

    args = parser.parse_args()

    # Load custom model
    model = load_custom_model(args.checkpoint, args.architecture)
    model = model.eval()

    # Verify normalization settings match model type expectations
    if args.architecture in ["resnet50", "wide_resnet50_4"]:
        # ResNet/WideResNet models should have internal normalization
        assert hasattr(model, 'normalize_input') and model.normalize_input == True, \
            f"{args.architecture} should have normalize_input=True (found: {getattr(model, 'normalize_input', 'N/A')})"
    elif args.architecture == "convnext_large":
        # ConvNeXt models should NOT have internal normalization
        assert hasattr(model, 'normalize_input') and model.normalize_input == False, \
            f"{args.architecture} should have normalize_input=False (found: {getattr(model, 'normalize_input', 'N/A')})"

    # All DAT models should NOT use input normalization wrapper
    assert args.input_normalization == False, \
        "External input normalization must be disabled (--input_normalization should always be False)"

    # Add ImageNet normalization if requested (should never happen for DAT models)
    if args.input_normalization:
        mu = (0.485, 0.456, 0.406)
        sigma = (0.229, 0.224, 0.225)
        model = normalize_model(model, mu, sigma)
        print("Added ImageNet normalization layer")
    else:
        print("Skipping external normalization (models handle normalization appropriately)")

    model.eval()
    print("Model loaded and set to eval mode")

    # Set device and enable multi-GPU if available
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

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

    # Create preprocessing function
    preprocessing = get_preprocessing_function(args.img_size)
    preprocessing_name = f"Resize{int(math.floor(args.img_size/0.875))}Crop{args.img_size}"
    print(f"Using preprocessing: {preprocessing_name}")
    trans = preprocessing

    # Compute clean accuracy on full validation set
    print("\nComputing clean accuracy on full validation set...")
    val_dir = os.path.join(args.data_dir, 'val')
    val_dataset = datasets.ImageFolder(val_dir, transform=trans)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    clean_acc_full = correct / total
    print(f"Clean accuracy (full validation set): {clean_acc_full:.2%}")

    print("\nStarting adversarial evaluation:")
    print("  Dataset: ImageNet")
    print(f"  Threat model: {args.threat_model}")
    print(f"  Epsilon: {args.eps}")
    print(f"  Number of examples: {args.n_examples}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Preprocessing: {preprocessing_name}")

    # Run evaluation with preprocessing specified
    _, robust_acc = benchmark(
        model=model,
        dataset=BenchmarkDataset.imagenet,
        threat_model=threat_model,
        eps=args.eps,
        n_examples=args.n_examples,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        device=device,
        preprocessing=preprocessing,  # Use specified preprocessing (string or function)
    )

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Clean accuracy (full validation set): {clean_acc_full:.4f} ({clean_acc_full:.2%})")
    print(
        f"Robust accuracy ({args.threat_model}, eps={args.eps}): {robust_acc:.4f} ({robust_acc:.2%})"
    )
    print("=" * 50)


if __name__ == "__main__":
    main()
