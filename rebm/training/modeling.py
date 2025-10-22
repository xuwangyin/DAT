"""Model creation, loading, and checkpoint management."""

import dataclasses
import logging
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Optional

# Add pytorch-image-models to path for timm imports
sys.path.insert(0, "pytorch-image-models")

import torch
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP

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
from torch import nn

import rebm.models.wide_resnet_innoutrobustness
import rebm.training.misc
from rebm.training.average_model import AveragedModel
from rebm.training.config_classes import BaseModelConfig, TrainConfig
from rebm.training.metrics import ClassificationMetrics, ImageGenerationMetrics

LOGGER = logging.getLogger(__name__)


def load_checkpoint(
    model: nn.Module, ckpt_path: str, weights_only: bool = True
) -> nn.Module:
    """Load model checkpoint with automatic DataParallel handling."""
    state_dict = torch.load(
        ckpt_path, weights_only=weights_only, map_location="cpu"
    )

    # EMA model
    if any(k.startswith("module.n_averaged") for k in state_dict.keys()):
        del state_dict["module.n_averaged"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[14:] if k.startswith("module.module.") else k
            new_state_dict[name] = v
        state_dict = new_state_dict

    # Check if the state_dict has 'module.' prefix (saved from DataParallel/DDP)
    # but the current model is not wrapped
    is_state_dict_wrapped = any(
        k.startswith("module.") for k in state_dict.keys()
    )
    is_model_wrapped = isinstance(model, (nn.DataParallel, DDP))

    if is_state_dict_wrapped and not is_model_wrapped:
        # Remove 'module.' prefix for loading into non-wrapped model
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = (
                k[7:] if k.startswith("module.") else k
            )  # remove 'module.' prefix
            new_state_dict[name] = v
        state_dict = new_state_dict
    elif not is_state_dict_wrapped and is_model_wrapped:
        # Add 'module.' prefix for loading into wrapped model
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
    device: torch.device,
    num_classes: int,
    indist_dataset: str,
    use_ddp: bool = False,
    rank: int = 0,
) -> nn.Module:
    """Create and initialize a model based on configuration.

    Args:
        model_config: Model configuration (type, checkpoint path, architecture settings)
        device: Device to place model on
        num_classes: Number of output classes
        indist_dataset: Name of in-distribution dataset for normalization
        use_ddp: Whether to use DistributedDataParallel
        rank: Rank for DDP

    Returns:
        Initialized model (wrapped in DataParallel or DDP)
    """
    model_type = model_config.model_type

    # ImageNet ResNet/WideResNet models
    if "resnet" in model_type.lower() and model_type.lower().endswith(
        "imagenet"
    ):
        model_class = globals().get(model_type)

        # Instantiate the model
        model = model_class(
            num_classes=num_classes,
            normalize_input=model_config.normalize_input,
            use_batchnorm=model_config.use_batchnorm,
        ).to(device)

        # Wrap model with DDP or DataParallel
        if use_ddp:
            # broadcast_buffers=False prevents BatchNorm buffer sync issues that cause
            # "modified by an inplace operation" errors in autograd
            model = DDP(model, device_ids=[rank], broadcast_buffers=False)
        else:
            model = nn.DataParallel(model)

        # Load checkpoint if specified
        if model_config.ckpt_path is not None:
            model = load_checkpoint(model, model_config.ckpt_path)

        return model

    # ConvNeXt models from timm
    elif model_type.startswith("convnext_"):
        model_class = locals().get(model_type)

        # Instantiate the model with custom parameters
        model = model_class(
            num_classes=num_classes,
            normalize_input=model_config.normalize_input,
            use_layernorm=model_config.use_layernorm,
        ).to(device)

        # TODO: Uncomment when utils_architecture.py is fixed
        # if model_config.use_convstem:
        #     model = replace_convstem(model, model_type)

        # Wrap model with DDP or DataParallel
        if use_ddp:
            # broadcast_buffers=False prevents BatchNorm buffer sync issues that cause
            # "modified by an inplace operation" errors in autograd
            model = DDP(model, device_ids=[rank], broadcast_buffers=False)
        else:
            model = nn.DataParallel(model)

        # Load checkpoint if specified
        if model_config.ckpt_path is not None:
            model = load_checkpoint(model, model_config.ckpt_path)

        return model

    # CIFAR WideResNet models
    elif model_type.lower().startswith("wideresnet"):
        model_class = getattr(
            rebm.models.wide_resnet_innoutrobustness, model_type
        )

        # Instantiate the model
        model = model_class(
            num_classes=num_classes,
            normalize_input=model_config.normalize_input,
            use_batchnorm=model_config.use_batchnorm,
        ).to(device)

        # Wrap model with DDP or DataParallel
        if use_ddp:
            # broadcast_buffers=False prevents BatchNorm buffer sync issues that cause
            # "modified by an inplace operation" errors in autograd
            model = DDP(model, device_ids=[rank], broadcast_buffers=False)
        else:
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


def save_checkpoint(
    model: nn.Module, optimizer: torch.optim.Optimizer, step: int
):
    """Save model and optimizer checkpoint (legacy format).

    Args:
        model: Model to save
        optimizer: Optimizer to save
        step: Current training step

    Note: This saves model and optimizer separately. For full training resumption,
    use save_training_state() instead.
    """
    rebm.training.misc.save_model(
        model, os.path.join(wandb.run.dir, f"model_{step}.pth")
    )
    rebm.training.misc.save_model(
        optimizer, os.path.join(wandb.run.dir, f"optimizer_{step}.pth")
    )


def save_training_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    cur_outdist_steps: int,
    image_generation_metrics: ImageGenerationMetrics,
    classification_metrics: ClassificationMetrics,
    cfg: TrainConfig,
    ema_model: Optional[AveragedModel] = None,
    wandb_run_id: Optional[str] = None,
) -> str:
    """Save complete training state to checkpoint for resumption.

    Args:
        model: Model to save (can be wrapped in DDP/DataParallel)
        optimizer: Optimizer to save
        global_step: Current global step (1-indexed)
        cur_outdist_steps: Current out-distribution attack steps
        image_generation_metrics: Image generation metrics tracker
        classification_metrics: Classification metrics tracker
        cfg: Training configuration
        ema_model: EMA model (if using EMA)
        wandb_run_id: Wandb run ID for resumption

    Returns:
        Path to saved state file
    """
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_state': {
            'global_step': global_step,
            'cur_outdist_steps': cur_outdist_steps,
        },
        'metrics': {
            'image_generation_metrics': dataclasses.asdict(image_generation_metrics),
            'classification_metrics': dataclasses.asdict(classification_metrics),
        },
        'config': dataclasses.asdict(cfg),
        'wandb_run_id': wandb_run_id,
    }

    # Save EMA model state if using EMA
    if ema_model is not None:
        state['ema_state_dict'] = ema_model.state_dict()

    # Save state (overwrites previous)
    save_path = os.path.join(wandb.run.dir, "training_state_latest.pth")
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    torch.save(state, save_path)

    LOGGER.info(f"Saved training state to {save_path}")
    return save_path


def load_training_state(
    state_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    ema_model: Optional[AveragedModel] = None,
) -> dict:
    """Load complete training state and restore.

    Args:
        state_path: Path to training state file
        model: Model to load state into (can be wrapped in DDP/DataParallel)
        optimizer: Optimizer to load state into
        device: Device for loading
        ema_model: EMA model to load state into (if using EMA)

    Returns:
        Dictionary containing training state:
            - global_step: Current global step
            - cur_outdist_steps: Current out-distribution attack steps
            - metrics: Dict with image_generation_metrics and classification_metrics
            - config: Training configuration from saved state
            - wandb_run_id: Wandb run ID (if available)
    """
    LOGGER.info(f"Loading training state from {state_path}")

    state = torch.load(state_path, map_location=device, weights_only=False)

    # Load model state
    model.load_state_dict(state['model_state_dict'])
    LOGGER.info("Loaded model state")

    # Load optimizer state
    optimizer.load_state_dict(state['optimizer_state_dict'])
    LOGGER.info("Loaded optimizer state")

    # Load EMA model state if available
    if ema_model is not None and 'ema_state_dict' in state:
        ema_model.load_state_dict(state['ema_state_dict'])
        LOGGER.info("Loaded EMA model state")
    elif ema_model is not None and 'ema_state_dict' not in state:
        LOGGER.warning(
            "EMA model requested but no EMA state found in saved state. "
            "EMA model will start from current model weights."
        )

    # Extract training state
    training_state = state['training_state']
    training_state['metrics'] = state.get('metrics', {})
    training_state['config'] = state.get('config', {})
    training_state['wandb_run_id'] = state.get('wandb_run_id', None)

    LOGGER.info(
        f"Resumed from step {training_state['global_step']}, "
        f"outdist_steps {training_state['cur_outdist_steps']}"
    )

    return training_state


def save_best_fid_model(model: nn.Module):
    """Save the model when a new best FID is achieved."""
    best_model_path = os.path.join(wandb.run.dir, "model_bestfid.pth")
    rebm.training.misc.save_model(model, best_model_path)


def save_best_accuracy_model(model: nn.Module):
    """Save the model when a new best test accuracy is achieved."""
    best_model_path = os.path.join(wandb.run.dir, "model_bestacc.pth")
    rebm.training.misc.save_model(model, best_model_path)
