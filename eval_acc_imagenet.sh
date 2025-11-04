#!/bin/bash
# Evaluate robust accuracy for all ImageNet models using run_acc_eval.py

# Parse partition argument (default: mi2508x)
PARTITION=${1:-mi2508x}

echo "Using partition: $PARTITION"
echo "========================================"

# Array of all main ImageNet configs
# Organized by: method, architecture, resolution
CONFIGS=(
    # # Standard Adversarial Training (AT) - 224x224
    # "model_configs/imagenet-at-ResNet50ImageNet.yaml"
    # "model_configs/imagenet-at-WideResNet50x4ImageNet.yaml"
    # "model_configs/imagenet-at-ConvNeXtLarge-convst.yaml"

    # # Dual Adversarial Training (DAT) - 224x224
    # "model_configs/imagenet-dat-ResNet50ImageNet-T15-300K.yaml"
    # "model_configs/imagenet-dat-ResNet50ImageNet-T30.yaml"
    # "model_configs/imagenet-dat-WideResNet50x4ImageNet-T30.yaml"
    # "model_configs/imagenet-dat-WideResNet50x4ImageNet-T65.yaml"
    # "model_configs/imagenet-dat-ConvNeXtSmall-convst.yaml"
    # "model_configs/imagenet-dat-ConvNeXtBase-convst.yaml"
    # "model_configs/imagenet-dat-ConvNeXtLarge-convst.yaml"

    # Dual Adversarial Training (DAT) - 256x256
    "model_configs/imagenet-dat-ResNet50ImageNet-T15-256x256.yaml"
    "model_configs/imagenet-dat-ResNet50ImageNet-T30-256x256.yaml"
    "model_configs/imagenet-dat-WideResNet50x4ImageNet-T30-256x256.yaml"
    "model_configs/imagenet-dat-WideResNet50x4ImageNet-T65-256x256.yaml"
    "model_configs/imagenet-dat-ConvNeXtLarge-convst-256x256-stepsize3.yaml"
    # "model_configs/imagenet-dat-ConvNeXtLarge-convst-256x256-stepsize4.yaml"

    # Standard Adversarial Training (AT) - 256x256 evaluation (resolution robustness test)
    # These models were trained at 224x224 but evaluated at 256x256 to test resolution robustness
    "model_configs/imagenet-at-ResNet50ImageNet-256x256.yaml"
    "model_configs/imagenet-at-WideResNet50x4ImageNet-256x256.yaml"
    "model_configs/imagenet-at-ConvNeXtLarge-convst-256x256.yaml"
)

# Submit jobs for each config
for config in "${CONFIGS[@]}"; do
    echo "Submitting evaluation for: $config"
    python run_acc_eval.py "$config" --partition $PARTITION --batch-size 1000
    echo ""
done

echo "========================================"
echo "All jobs submitted!"
echo "Check job status with: squeue -u $USER"
