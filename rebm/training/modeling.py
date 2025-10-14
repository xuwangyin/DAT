"""Model creation, loading, and checkpoint management."""

import logging
import os
import sys
from collections import OrderedDict

# Add pytorch-image-models to path for timm imports
sys.path.insert(0, "pytorch-image-models")

import torch
import wandb
from torch import nn

import rebm.models.wide_resnet_innoutrobustness
import rebm.training.misc
from rebm.training.config_classes import BaseModelConfig

# TODO: Fix import order issue with timm
# utils_architecture.py imports timm without sys.path.insert, which causes
# system timm (v1.0.15) to be cached in sys.modules before our local
# pytorch-image-models timm can be loaded. This breaks imports like
# wide_resnet50_4 which don't exist in system timm.
# Workaround: Commented out until utils_architecture.py adds sys.path.insert
# from rebm.training.utils_architecture import replace_convstem

# Import model classes that need to be available globally
from timm.models.resnet import resnet50 as ResNet50ImageNet
from timm.models.resnet import wide_resnet50_2 as WideResNet50x2ImageNet
from timm.models.resnet import wide_resnet50_4 as WideResNet50x4ImageNet

LOGGER = logging.getLogger(__name__)


def load_checkpoint(model: nn.Module, ckpt_path: str, weights_only: bool = True) -> nn.Module:
    """Load model checkpoint with automatic DataParallel handling."""
    state_dict = torch.load(ckpt_path, weights_only=weights_only, map_location="cpu")

    # EMA model
    if any(k.startswith("module.n_averaged") for k in state_dict.keys()):
        del state_dict["module.n_averaged"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[14:] if k.startswith("module.module.") else k
            new_state_dict[name] = v
        state_dict = new_state_dict

    # Check if the state_dict has 'module.' prefix (saved from DataParallel)
    # but the current model is not a DataParallel model
    is_state_dict_data_parallel = any(k.startswith("module.") for k in state_dict.keys())
    is_model_data_parallel = isinstance(model, nn.DataParallel)

    if is_state_dict_data_parallel and not is_model_data_parallel:
        # Remove 'module.' prefix for loading into non-DataParallel model
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k  # remove 'module.' prefix
            new_state_dict[name] = v
        state_dict = new_state_dict
    elif not is_state_dict_data_parallel and is_model_data_parallel:
        # Add 'module.' prefix for loading into DataParallel model
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = f"module.{k}" if not k.startswith("module.") else k
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    LOGGER.info(f"Loaded pretrained checkpoint from: {ckpt_path}")
    return model


def get_model(
    model_config: BaseModelConfig,
    device: str,
    num_classes: int,
    indist_dataset: str,
) -> nn.Module:
    """Create and initialize a model based on configuration.

    Args:
        model_config: Model configuration (type, checkpoint path, architecture settings)
        device: Device to place model on
        num_classes: Number of output classes
        indist_dataset: Name of in-distribution dataset for normalization

    Returns:
        Initialized model (wrapped in DataParallel)
    """
    model_type = model_config.model_type

    # ImageNet ResNet/WideResNet models
    if "resnet" in model_type.lower() and model_type.lower().endswith("imagenet"):
        model_class = globals().get(model_type)

        # Instantiate the model
        model = model_class(
            num_classes=num_classes,
            normalize_input=model_config.normalize_input,
            use_batchnorm=model_config.use_batchnorm,
        ).to(device)

        model = nn.DataParallel(model)

        # Load checkpoint if specified
        if model_config.ckpt_path is not None:
            model = load_checkpoint(model, model_config.ckpt_path)

        return model

    # ConvNeXt models from timm
    elif model_type.startswith("convnext_"):
        from timm.models.convnext import convnext_tiny, convnext_base, convnext_small, convnext_large

        model_class = locals().get(model_type)

        # Instantiate the model with custom parameters
        model = model_class(
            num_classes=num_classes,
            normalize_input=model_config.normalize_input,
            use_layernorm=model_config.use_layernorm,
        ).to(device)

        if model_config.use_convstem:
            model = replace_convstem(model, model_type)

        model = nn.DataParallel(model)

        # Load checkpoint if specified
        if model_config.ckpt_path is not None:
            model = load_checkpoint(model, model_config.ckpt_path)

        return model

    # CIFAR WideResNet models
    elif model_type.lower().startswith("wideresnet"):
        model_class = getattr(rebm.models.wide_resnet_innoutrobustness, model_type)

        # Instantiate the model
        model = model_class(
            num_classes=num_classes,
            normalize_input=model_config.normalize_input,
            use_batchnorm=model_config.use_batchnorm,
        ).to(device)

        model = nn.DataParallel(model)

        # Load checkpoint if specified
        if model_config.ckpt_path is not None:
            model = load_checkpoint(model, model_config.ckpt_path)

        return model

    else:
        raise ValueError(f"Unknown model: {model_type}")


def get_optimizer(
    model: nn.Module,
    optimizer_name: str,
    lr: float,
    wd: float,
) -> torch.optim.Optimizer:
    """Create optimizer for model parameters.

    Args:
        model: Model to optimize
        optimizer_name: Optimizer type ("sgd", "adam", "adamw")
        lr: Learning rate
        wd: Weight decay

    Returns:
        Initialized optimizer
    """
    match optimizer_name:
        case "sgd":
            # Copy https://github.com/locuslab/robust_overfitting/blob/c47a25c5e00c8b2bb35488d962c04dd771b7e9af/train_cifar.py#L230
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=wd,
                nesterov=True,
            )

        case "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                betas=(0.0, 0.99),
                weight_decay=wd,
            )

        case "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                betas=(0.9, 0.95),
                weight_decay=wd,
            )

        case _:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizer


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, step: int):
    """Save model and optimizer checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer to save
        step: Current training step
    """
    rebm.training.misc.save_model(
        model, os.path.join(wandb.run.dir, f"model_{step}.pth")
    )
    rebm.training.misc.save_model(
        optimizer, os.path.join(wandb.run.dir, f"optimizer_{step}.pth")
    )


def save_best_fid_model(model: nn.Module):
    """Save the model when a new best FID is achieved."""
    best_model_path = os.path.join(wandb.run.dir, "model_bestfid.pth")
    rebm.training.misc.save_model(model, best_model_path)


def save_best_accuracy_model(model: nn.Module):
    """Save the model when a new best test accuracy is achieved."""
    best_model_path = os.path.join(wandb.run.dir, "model_bestacc.pth")
    rebm.training.misc.save_model(model, best_model_path)
