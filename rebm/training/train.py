import sys

sys.path.insert(0, "pytorch-image-models")
from timm.models.resnet import resnet50 as ResNet50ImageNet
from timm.models.resnet import wide_resnet50_2 as WideResNet50x2ImageNet
from timm.models.resnet import wide_resnet50_4 as WideResNet50x4ImageNet
from timm.models.convnext import convnext_tiny, convnext_base, convnext_small, convnext_large

import collections
import dataclasses
import hashlib
import json
import logging
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import uuid
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Literal, Optional, Tuple

import einops
# import kornia.augmentation as K
import numpy as np
import simple_parsing
import torch
import torch.utils.data
import torchvision.utils
import wandb
import yaml
from timm.models.layers import trunc_normal_
from torch import nn
from torchvision import datasets

import InNOutRobustness.utils.datasets as dl
from InNOutRobustness.utils.datasets.augmentations.imagenet_augmentation import get_imageNet_augmentation

# import rebm.models.bat
# import rebm.models.convnextv2 as convnextv2
import rebm.models.preactresnet
import rebm.models.robustness_resnet_cifar10
import rebm.models.wide_resnet_innoutrobustness
import rebm.models.resnet
import rebm.training.adv_attacks
import rebm.training.data
import rebm.training.misc
from rebm.training.adv_attacks import pgd_attack, pgd_attack_xent
from rebm.training.average_model import AveragedModel
from rebm.training.calibration import eval_calibration
from rebm.training.config_classes import (
    AttackConfig,
    BaseModelConfig,
    DataConfig,
    ImageLogConfig,
    create_model_config,
)
from rebm.training.eval_utils import (
    compute_fid,
    compute_img_diff,
    eval_acc,
    eval_robust_acc,
    generate_images,
    get_auc,
    log_generate_images,
    generate_counterfactuals,
    ood_detection,
)
from rebm.utils import assert_no_grad, load_state_dict, remap_checkpoint_keys
from rebm.training.utils_architecture import replace_convstem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def infinite_iter(iterable: Iterable):
    while True:
        for x in iterable:
            yield x


def get_lr_for_epoch(
    base_lr: float, epoch: int, total_epochs: int, dataset=None
) -> float:
    """
    Implements a stepwise learning rate decay with three phases:
    1. For the first 50% of training epochs, the learning rate remains at its maximum
    2. Between 50% and 75% of epochs, it decreases by a factor of 10
    3. In the final 25% of training, it further drops by a factor of 100

    Args:
        base_lr: The initial (maximum) learning rate
        epoch: Current epoch (0-indexed)
        total_epochs: Total number of epochs for training

    Returns:
        The learning rate for the current epoch
    """
    if dataset in ["RestrictedImageNet", "ImageNet"]:
        assert total_epochs == 75
        if epoch < 30:
            return base_lr
        elif epoch < 60:
            return base_lr / 10.0
        elif epoch < 75:
            return base_lr / 100.0
        else:
            return base_lr / 1000.0
    else:
        if epoch > 200:
            # Adversarial Robustness on In- and Out-Distribution Improves Explainability
            return base_lr / 1000.0
        if epoch < total_epochs * 0.5:
            return base_lr
        elif epoch < total_epochs * 0.75:
            return base_lr / 10.0
        else:
            return base_lr / 100.0


def dict_append_label(d: dict, label: str) -> dict:
    return {label + k: v for k, v in d.items()}


def recursive_asdict(obj):
    """Recursively converts dataclass instances to dictionaries."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            k: recursive_asdict(v) for k, v in dataclasses.asdict(obj).items()
        }
    elif isinstance(obj, list):
        return [recursive_asdict(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: recursive_asdict(v) for k, v in obj.items()}
    else:
        return obj


@dataclasses.dataclass
class TrainConfig:
    """Default parameters are for training on lsun-bird. See defaults in baselines/*.yaml"""

    # Required parameters
    data: DataConfig
    attack: AttackConfig
    model: BaseModelConfig
    image_log: ImageLogConfig
    config_path: str

    # Optimization parameters
    resume_path: Optional[str]  # Path to resume training from checkpoint
    optimizer: str
    wd: float
    lr: float
    r1reg: float
    xent_indist_weight: float
    xent_outdist_weight: float
    xent_adv_weight: float  # These 3 will be normalized to sum to 1
    r1_indist_weight: float
    r1_outdist_weight: float
    r1_adv_weight: float  # These 3 will be normalized to sum to 1

    # Training scheduling
    batch_size: int  # openai/clip-vit-base-patch32: 160, stargan: 62
    min_imgs_per_threshold: int  # Minimum steps before checking AUC
    AUC_th: float  # When AUC reaches threshold, move to next epoch
    rand_seed: int

    # Logging
    n_imgs_per_metrics_log: int
    n_imgs_per_image_log: int
    n_imgs_per_ckpt_save: int

    # WandB
    wandb_project: str
    wandb_dir: str  # Default: ./
    wandb_disabled: bool
    tags: Tuple[str, ...]

    # Optional parameters
    indist_attack: AttackConfig | None = None
    indist_attack_only: bool = False
    indist_attack_xent: AttackConfig | None = None
    indist_clean_extra: bool = (
        False  # When true, sample additional clean data and compute xent
    )
    fp16: bool = False
    samples_per_attack_step: int | None = None
    n_imgs_per_classification_log: int | None = None
    use_ema: bool = (
        False  # Whether to use Exponential Moving Average for model weights
    )

    # Evaluation parameters
    robust_eval: bool = True  # Whether to perform robust evaluation
    indist_perturb: bool = False
    indist_perturb_steps: int = 10
    indist_perturb_eps: float = 0.5
    augm_type_classification: str = "autoaugment_cutout"
    augm_type_generation: str = "original"
    mixup_alpha: int = 5
    mixup_beta: int = 1
    tinyimages_loader: Literal["GOOD", "innout"] = "GOOD"
    use_batchnorm: bool = False
    use_layernorm: bool = True
    use_convstem: bool = True
    indist_train_only: bool = False
    fixed_lr: bool = False
    logsumexp: bool = True
    logsumexp_sampling: bool = False
    bce_weight: float = 1.0
    xent_lr_multiplier: float = 1.0
    eval_only: bool = False  # When enabled, quit after FID score is computed
    use_counterfactuals: bool = False
    evaluate_ood_detection: bool = False  # When enabled, perform OOD detection evaluation
    ood_detection_logsumexp: bool = False  # When enabled, perform OOD detection evaluation with logsumexp
    outdist_dataset_ood_detection: str = "noise"  # Dataset to use for OOD detection evaluation, options: "noise", "svhn", "cifar100", "cifar10", "imagenet"
    openimages_max_samples: int | None = None  # Maximum number of samples to use from OpenImages dataset (default: use all ~330K samples)
    openimages_augm: str | None = None  # Augmentation type for OpenImages dataset

    total_epochs: int | None = None  # Total number of epochs for lr scheduling

    @property
    def dtype(self) -> torch.dtype:
        return torch.float16 if self.fp16 else torch.float32

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainConfig":
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        config_dict["config_path"] = yaml_path
        # Create nested configs
        data = DataConfig(**config_dict.get("data", {}))
        attack = AttackConfig(**config_dict.get("attack", {}))
        indist_attack = None
        if "indist_attack" in config_dict:
            indist_attack = AttackConfig(**config_dict.get("indist_attack", {}))
        indist_attack_xent = None
        if "indist_attack_xent" in config_dict:
            indist_attack_xent = AttackConfig(
                **config_dict.get("indist_attack_xent", {})
            )
        model = create_model_config(
            config_dict.get("model", {})
        )  # Use factory function
        image_log = ImageLogConfig(**config_dict.get("image_log", {}))

        # Remove nested configs from main dict
        for key in [
            "data",
            "attack",
            "indist_attack",
            "indist_attack_xent",
            "model",
            "image_log",
        ]:
            config_dict.pop(key, None)

        return cls(
            data=data,
            attack=attack,
            indist_attack=indist_attack,
            indist_attack_xent=indist_attack_xent,
            model=model,
            image_log=image_log,
            **config_dict,
        )

    def should_trigger_event(
        self,
        global_step_one_indexed: int,
        interval_in_imgs: int,
        at_end: bool = False,
    ):
        global_step0 = global_step_one_indexed - 1
        global_images0 = global_step0 * self.batch_size
        next_images = (global_step0 + 1) * self.batch_size

        # Special case for first step
        if global_step0 == 0 and not at_end:
            return True

        if at_end:
            # Check if we're approaching the end of an interval
            current_interval = global_images0 // interval_in_imgs
            next_interval = next_images // interval_in_imgs
            return (current_interval < next_interval) or (
                global_images0 % interval_in_imgs
                >= interval_in_imgs - self.batch_size
            )
        else:
            # For start triggers, check if we're crossing into a new interval
            prev_interval = global_images0 // interval_in_imgs
            next_interval = next_images // interval_in_imgs
            return prev_interval < next_interval

    def __post_init__(self):
        if self.resume_path is not None:
            # Set a random seed for resumption
            self.seed = int(datetime.now().timestamp())

            summary_path = Path(self.resume_path) / "wandb-summary.json"
            with open(summary_path, "r") as f:
                summary_data = json.load(f)
                self.attack.start_step = summary_data.get(
                    "cur_outdist_steps", 0
                )

        if self.wandb_dir is None:
            self.wandb_dir = "./"
        if self.image_log.save_dir is None:
            # Create short directory names to avoid filesystem limits
            # Use hash of the full config path to ensure uniqueness
            random_suffix = uuid.uuid4().hex  # 32 random hex chars
            config_hash = hashlib.md5(
                (str(self.config_path) + random_suffix).encode("utf-8")
            ).hexdigest()[:8]
            self.image_log.save_dir = f"{self.wandb_dir}/eval_fid/{config_hash}"
        if (
            self.optimizer == "sgd"
            and self.total_epochs is None
            and not self.fixed_lr
        ):
            raise ValueError("total_epochs must be set for SGD optimizer")
        # if (
        #     self.indist_attack_xent is not None
        #     and self.indist_attack is not None
        # ):
        #     raise ValueError(
        #         "indist_attack and indist_attack_xent cannot both be set"
        #     )
        if self.indist_train_only and self.indist_clean_extra:
            raise ValueError(
                "indist_clean_extra cannot be True when indist_train_only is True"
            )
        # if self.indist_attack_xent is not None and self.indist_perturb:
        #     raise ValueError(
        #         "indist_attack_xent and indist_perturb cannot both be set"
        #     )

    @property
    def device(self) -> torch.device:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_model(self) -> nn.Module:
        cfg = self.model

        def load_checkpoint(model, ckpt_path, weights_only=True):
            state_dict = torch.load(
                ckpt_path, weights_only=weights_only, map_location="cpu"
            )

            # EMA model
            if any(
                k.startswith("module.n_averaged") for k in state_dict.keys()
            ):
                del state_dict["module.n_averaged"]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[14:] if k.startswith("module.module.") else k
                    new_state_dict[name] = v
                state_dict = new_state_dict

            # Check if the state_dict has 'module.' prefix (saved from DataParallel)
            # but the current model is not a DataParallel model
            is_state_dict_data_parallel = any(
                k.startswith("module.") for k in state_dict.keys()
            )
            is_model_data_parallel = isinstance(model, nn.DataParallel)

            if is_state_dict_data_parallel and not is_model_data_parallel:
                # Remove 'module.' prefix for loading into non-DataParallel model

                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = (
                        k[7:] if k.startswith("module.") else k
                    )  # remove 'module.' prefix
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

        def load_resume_checkpoint(model):
            ckpt_path = os.path.join(self.resume_path, "model.pth")
            model = load_checkpoint(model, ckpt_path, weights_only=True)
            LOGGER.info(f"Resuming model from {self.resume_path}")
            return model

        match cfg.model_type:
            case model_type if "resnet" in model_type.lower() and model_type.lower().endswith("imagenet"):
                model_class = globals().get(cfg.model_type)

                # Instantiate the model
                model = model_class(
                    num_classes=self.data.num_classes,
                    normalize_input=cfg.normalize_input,
                    normalization_type=self.data.indist_dataset.lower(),
                    use_batchnorm=self.use_batchnorm,
                ).to(self.device)

                model = nn.DataParallel(model)

                # Load checkpoint if specified
                if cfg.ckpt_path is not None:
                    model = load_checkpoint(model, cfg.ckpt_path)

                # Resume from checkpoint if specified
                if self.resume_path is not None:
                    model = load_resume_checkpoint(model)

                return model

            case model_type if model_type.startswith("convnext_"):
                # Handle ConvNeXt models from timm
                model_class = globals().get(cfg.model_type)

                # Instantiate the model with custom parameters
                model = model_class(
                    num_classes=self.data.num_classes,
                    normalize_input=cfg.normalize_input,
                    use_layernorm=self.use_layernorm,
                ).to(self.device)

                if self.use_convstem:
                    model = replace_convstem(model, cfg.model_type)

                model = nn.DataParallel(model)

                # Load checkpoint if specified
                if cfg.ckpt_path is not None:
                    model = load_checkpoint(model, cfg.ckpt_path)

                # Resume from checkpoint if specified
                if self.resume_path is not None:
                    model = load_resume_checkpoint(model)

                return model

            case model_type if (
                model_type.lower().startswith(("resnet", "preactresnet", "wideresnet")) and
                not model_type.lower().endswith("imagenet")
            ):
                # Handle different types of ResNet models
                if cfg.model_type.lower().startswith("preact"):
                    # For PreActResNet models, use the preactresnet module
                    model_class = getattr(
                        rebm.models.preactresnet, cfg.model_type
                    )
                elif cfg.model_type.lower().startswith("wideresnet"):
                    # For WideResNet models, use the wide_resnet module
                    model_class = getattr(
                        rebm.models.wide_resnet_innoutrobustness, cfg.model_type
                    )
                else:
                    # For standard ResNet models, use the robustness_resnet_cifar10 module
                    model_class = getattr(
                        rebm.models.robustness_resnet_cifar10, cfg.model_type
                    )

                # Instantiate the model
                model = model_class(
                    num_classes=self.data.num_classes,
                    normalize_input=cfg.normalize_input,
                    use_batchnorm=self.use_batchnorm,
                ).to(self.device)

                model = nn.DataParallel(model)

                # Load checkpoint if specified
                if cfg.ckpt_path is not None:
                    model = load_checkpoint(model, cfg.ckpt_path)

                # Resume from checkpoint if specified
                if self.resume_path is not None:
                    model = load_resume_checkpoint(model)

                return model

            case _:
                raise ValueError(f"Unknown model: {cfg.model_type}")

    def _get_base_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        match self.optimizer:
            case "sgd":
                # Copy https://github.com/locuslab/robust_overfitting/blob/c47a25c5e00c8b2bb35488d962c04dd771b7e9af/train_cifar.py#L230
                return torch.optim.SGD(
                    model.parameters(),
                    lr=self.lr,
                    momentum=0.9,
                    weight_decay=self.wd,
                    nesterov=True,
                )

            case "adam":
                return torch.optim.Adam(
                    model.parameters(),
                    lr=self.lr,
                    betas=(0.0, 0.99),
                    weight_decay=self.wd,
                )

            case "adamw":
                return torch.optim.AdamW(
                    model.parameters(),
                    lr=self.lr,
                    betas=(0.9, 0.95),
                    weight_decay=self.wd,
                )

            case _:
                raise ValueError(f"Unknown optimizer: {self.optimizer}")

    def get_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        optimizer = self._get_base_optimizer(model)
        if self.resume_path is not None:
            optimizer.load_state_dict(
                torch.load(os.path.join(self.resume_path, "optimizer.pth"))
            )
            LOGGER.info(f"Resuming optimizer from {self.resume_path}")

        return optimizer

    def save_state(
        self, model: nn.Module, optimizer: torch.optim.Optimizer, step: int
    ):
        rebm.training.misc.save_model(
            model, os.path.join(wandb.run.dir, f"model_{step}.pth")
        )
        rebm.training.misc.save_model(
            optimizer, os.path.join(wandb.run.dir, f"optimizer_{step}.pth")
        )

    def save_best_fid_model(self, model: nn.Module):
        """Save the model when a new best FID is achieved."""
        best_model_path = os.path.join(wandb.run.dir, "model_bestfid.pth")
        rebm.training.misc.save_model(model, best_model_path)

    def save_best_accuracy_model(self, model: nn.Module):
        """Save the model when a new best test accuracy is achieved."""
        best_model_path = os.path.join(wandb.run.dir, "model_bestacc.pth")
        rebm.training.misc.save_model(model, best_model_path)

    def get_indist_dataset(
        self,
        split: str = "train",
        attack: bool = False,
        augm_type: str = "autoaugment_cutout",
    ):
        cfg = self.data
        match cfg.indist_dataset:
            case "cifar10-conditional":
                indist_dataset = rebm.training.data.get_cifar10_dataset(
                    data_dir=cfg.indist_ds_dir,
                    split=split,
                    conditional=True,
                    augm_type=augm_type,
                )
            case "cifar100-conditional":
                indist_dataset = rebm.training.data.get_cifar100_dataset(
                    data_dir=cfg.indist_ds_dir,
                    split=split,
                    conditional=True,
                    augm_type=augm_type,
                )
            case "ImageNet":
                LOGGER.info("Using ImageNet dataset")

                # Validate augmentation type for ImageNet
                is_train = split == "train"
                if is_train:
                    assert augm_type in ["madry", "generation_id", "generation_id_randomcrop", "none", "default"]
                else:
                    assert augm_type in ["none", "test"]

                transform = get_imageNet_augmentation(type=augm_type, out_size=224)
                dataset_split = 'train' if split == 'train' else 'val'
                indist_dataset = datasets.ImageNet(
                    cfg.indist_ds_dir,
                    split=dataset_split,
                    transform=transform
                )
            case _:
                raise ValueError(f"Unknown dataset: {cfg.indist_dataset}")
        assert len(indist_dataset.classes) == self.data.num_classes
        return indist_dataset

    def get_indist_dataloader(
        self,
        split: str = "train",
        attack: bool = False,
        shuffle: bool = True,
        augm_type: str = "autoaugment_cutout",
        balanced=True,
    ):
        dataset = self.get_indist_dataset(
            split=split, attack=attack, augm_type=augm_type
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=True,
            pin_memory=True,
            num_workers=self.data.num_workers,
            persistent_workers=True,
        )

    def get_outdist_dataset(self, split="train"):
        cfg = self.data
        match cfg.outdist_dataset:
            case "OpenImageO":
                LOGGER.info("Using OpenImageO outdist dataset")

                # Determine augmentation type
                if self.openimages_augm is not None:
                    augm_type = self.openimages_augm
                    LOGGER.info(f"Using custom OpenImageO augmentation: {augm_type}")
                else:
                    augm_type = "generation_od_randomcrop" if self.augm_type_generation == "generation_id_randomcrop" else "generation_od"

                transform = get_imageNet_augmentation(type=augm_type, out_size=224)
                dataset = datasets.ImageFolder(cfg.outdist_std_dir, transform=transform)
                original_size = len(dataset)

                # Create random subset if max_samples is specified
                if self.openimages_max_samples is not None and self.openimages_max_samples < len(dataset):
                    generator = torch.Generator()
                    generator.manual_seed(42)
                    indices = torch.randperm(len(dataset), generator=generator)[:self.openimages_max_samples].tolist()
                    outdist_dataset = torch.utils.data.Subset(dataset, indices)
                    LOGGER.info(f"OpenImageO dataset: Using {len(outdist_dataset)} samples out of {original_size} total samples")
                else:
                    outdist_dataset = dataset
                    LOGGER.info(f"OpenImageO dataset: Using all {original_size} samples")

            case "tinyimages":
                if split != "train":
                    LOGGER.warning(
                        f"TinyImages don't have a {split} split, using the train split instead"
                    )
                outdist_dataset = rebm.training.data.get_tinyimages_dataset(
                    data_dir=cfg.outdist_std_dir,
                    augm_type=self.augm_type_generation,
                    tinyimages_loader=self.tinyimages_loader,
                )
            case _:
                raise ValueError(
                    f"Unknown outdist dataset: {cfg.outdist_dataset}"
                )
        return outdist_dataset

    def get_outdist_dataloader(self, split="train", shuffle=True):
        outdist_dataset = self.get_outdist_dataset(split=split)
        return torch.utils.data.DataLoader(
            outdist_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=True,
            pin_memory=True,
            num_workers=self.data.num_workers,
            persistent_workers=True,
        )


@dataclasses.dataclass
class Metrics:
    clean_auc: float
    adv_auc: float
    indist_adv_auc: float | None = None
    indist_clean_auc: float | None = None

    def to_simple_dict(self) -> dict[str, float]:
        ret_dict = dict()
        for field in dataclasses.fields(self):
            key = field.name
            val = getattr(self, key)
            if val is None:
                continue

            if torch.is_tensor(val):
                if val.numel() == 1:
                    ret_dict[key] = val.item()
                continue

            ret_dict[key] = val

        return ret_dict


@dataclasses.dataclass
class TrainingMetrics(Metrics):
    xent: torch.Tensor | None = None
    loss: torch.Tensor | None = None
    r1: torch.Tensor | None = None

    indist_imgs: torch.Tensor | None = None
    outdist_imgs_clean: torch.Tensor | None = None
    adv_imgs: torch.Tensor | None = None
    outdist_imgs_error: torch.Tensor | None = None

    xent_indist: torch.Tensor | None = None
    xent_outdist: torch.Tensor | None = None
    xent_adv: torch.Tensor | None = None

    l2_dist_relative: float | None = None


@dataclasses.dataclass
class ImageGenerationMetrics:
    """Class to track image generation metrics including FID scores and generated images."""

    fid: float | None = None
    gen_imgs: torch.Tensor | None = None
    best_fid: float = float("inf")

    def update(self, fid: float | None, gen_imgs: torch.Tensor | None) -> bool:
        self.fid = fid
        self.gen_imgs = gen_imgs

        # Check if we have a new best FID
        new_best = False
        if fid is not None and fid < self.best_fid:
            self.best_fid = fid
            new_best = True

        return new_best


@dataclasses.dataclass
class ClassificationMetrics:
    """Class to track classification metrics including training and test accuracy before and after calibration."""

    # Standard (pre-calibration) metrics
    train_acc: float | None = None
    test_acc: float | None = None

    # Post-calibration metrics
    train_acc_calib: float | None = None
    test_acc_calib: float | None = None

    # Robust accuracy metrics
    robust_train_acc: float | None = None
    robust_test_acc: float | None = None

    # Best metrics tracking
    best_test_acc: float | None = None
    best_test_acc_calib: float | None = None
    best_robust_test_acc: float | None = None

    def update(
        self,
        *,  # Force keyword arguments
        train_acc: float,
        test_acc: float,
        train_acc_calib: float | None = None,
        test_acc_calib: float | None = None,
        robust_train_acc: float | None = None,
        robust_test_acc: float | None = None,
    ) -> bool:
        # Update standard (pre-calibration) metrics
        self.train_acc = train_acc
        self.test_acc = test_acc

        # Update post-calibration metrics
        self.train_acc_calib = train_acc_calib
        self.test_acc_calib = test_acc_calib

        # Update robust accuracy metrics
        if robust_train_acc is not None:
            self.robust_train_acc = robust_train_acc
        if robust_test_acc is not None:
            self.robust_test_acc = robust_test_acc

        # Update best standard accuracy
        new_best = False
        # if self.best_test_acc is None or test_acc > self.best_test_acc:
        #     self.best_test_acc = test_acc
        #     new_best = True

        # Check if we have a new best robust test accuracy
        if robust_test_acc is not None and (
            self.best_robust_test_acc is None
            or robust_test_acc > self.best_robust_test_acc
        ):
            self.best_robust_test_acc = robust_test_acc
            new_best = True

        # Check if we have a new best calibrated test accuracy
        if test_acc_calib is not None and (
            self.best_test_acc_calib is None
            or test_acc_calib > self.best_test_acc_calib
        ):
            self.best_test_acc_calib = test_acc_calib

        return new_best


def log_generation(model, cfg):
    assert_no_grad(model)
    fid, gen_imgs = None, None
    model.eval()
    if cfg.image_log.log_fid:
        if cfg.image_log.adaptive_steps:
            from rebm.training.eval_utils import find_optimal_steps
            optimal_steps = find_optimal_steps(cfg, model)
            fid = compute_fid(
                model=model,
                cfg=cfg,
                override_fid_cfg={'num_steps': optimal_steps}
            )
        else:
            fid = compute_fid(
                model=model,
                cfg=cfg,
            )
        
        # Free CUDA memory after FID computation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    gen_imgs = log_generate_images(
        cfg=cfg,
        model=model,
        samples=10,
    )

    return fid, gen_imgs


def generate_outdist_adv_images(
    model, outdist_imgs, cfg, outdist_step, indist_labels=None
):
    assert_no_grad(model)
    assert not model.training
    # Generate adversarial images from out-of-distribution samples
    if indist_labels is not None:
        attack_labels = indist_labels
    else:
        assert False
        attack_labels = torch.randint(
            0,
            cfg.data.num_classes,
            (outdist_imgs.size(0),),
        )

    adv_imgs = generate_images(
        num_steps=outdist_step,
        model=model,
        x=outdist_imgs,
        attack_labels=attack_labels,
        logsumexp=cfg.logsumexp_sampling,
        **vars(cfg.attack),
    )
    assert not adv_imgs.requires_grad
    assert_no_grad(model)
    return adv_imgs, attack_labels


def compute_metrics(
    model,
    indist_imgs,
    indist_labels,
    adv_imgs,
    outdist_imgs,
    attack_labels,
    indist_adv_imgs=None,
    indist_attack_labels=None,
):
    """
    Compute AUC metrics with no gradient tracking.
    """
    assert_no_grad(model)
    assert not model.training

    with torch.no_grad():
        # Compute logits for mandatory inputs
        inputs = torch.cat([indist_imgs, adv_imgs, outdist_imgs])
        labels = torch.cat([indist_labels, attack_labels, attack_labels])

        batch_logits = model(inputs, labels)
        indist_logits, adv_logits, outdist_logits = torch.chunk(batch_logits, 3)

        # Compute basic AUC metrics
        auc_metrics = {
            "adv_auc": get_auc(
                pos=indist_logits.cpu().numpy(), neg=adv_logits.cpu().numpy()
            ),
            "clean_auc": get_auc(
                pos=indist_logits.cpu().numpy(),
                neg=outdist_logits.cpu().numpy(),
            ),
        }

        # Compute optional in-distribution adversarial metrics if provided
        if indist_adv_imgs is not None and indist_attack_labels is not None:
            # Process additional inputs separately for clarity
            inputs = torch.cat([indist_adv_imgs, indist_imgs])
            labels = torch.cat([indist_attack_labels, indist_attack_labels])

            batch_logits = model(inputs, labels)
            indist_adv_logits, indist_abstain_logits = torch.chunk(
                batch_logits, 2
            )

            # Add in-distribution adversarial AUC metrics
            auc_metrics.update(
                {
                    "indist_adv_auc": get_auc(
                        pos=indist_logits.cpu().numpy(),
                        neg=indist_adv_logits.cpu().numpy(),
                    ),
                    "indist_clean_auc": get_auc(
                        pos=indist_logits.cpu().numpy(),
                        neg=indist_abstain_logits.cpu().numpy(),
                    ),
                }
            )

    # Ensure no gradients were accumulated
    assert_no_grad(model)
    return Metrics(**auc_metrics)


def compute_training_metrics(
    *,
    indist_imgs: torch.Tensor,
    indist_labels: torch.Tensor,
    outdist_imgs: torch.Tensor,
    outdist_step: int,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    model: nn.Module,
    cfg: TrainConfig,
) -> Metrics:
    model.eval()
    adv_imgs, attack_labels = generate_outdist_adv_images(
        model, outdist_imgs, cfg, outdist_step, indist_labels=indist_labels
    )

    indist_adv_imgs, indist_attack_labels = None, None
    if cfg.indist_attack is not None:
        indist_adv_imgs, indist_attack_labels = generate_indist_adv_images(
            model, indist_imgs, indist_labels, cfg
        )

    if cfg.indist_perturb:
        indist_imgs = pgd_attack(
            model,
            indist_imgs,
            norm="L2",
            eps=cfg.indist_perturb_eps,
            step_size=cfg.attack.step_size,
            steps=cfg.indist_perturb_steps,
            attack_labels=indist_labels,
            descent=True,
        )

    l2_dist = (
        torch.norm(
            einops.rearrange(adv_imgs - outdist_imgs, "b ... -> b (...)"), dim=1
        )
        .mean()
        .item()
    )
    l2_dist_relative = (
        1
        if outdist_step == 0
        else l2_dist / (outdist_step * cfg.attack.step_size)
    )

    metrics = compute_metrics(
        model,
        indist_imgs,
        indist_labels,
        adv_imgs,
        outdist_imgs,
        attack_labels,
        indist_adv_imgs,
        indist_attack_labels,
    )

    indist_target = torch.ones(indist_imgs.shape[0]).to(indist_imgs.device)
    adv_target = torch.zeros(indist_imgs.shape[0]).to(indist_imgs.device)

    model.train()
    if cfg.r1reg > 0:
        indist_imgs.requires_grad_()
        if cfg.logsumexp:
            indist_logits = torch.logsumexp(model(x=indist_imgs, y=None), dim=1)
        else:
            indist_logits = model(x=indist_imgs, y=indist_labels)
        r1 = rebm.training.misc.r1_reg(indist_logits, indist_imgs)
    else:
        r1 = 0
        if cfg.logsumexp:
            indist_logits = torch.logsumexp(model(x=indist_imgs, y=None), dim=1)
        else:
            indist_logits = model(x=indist_imgs, y=indist_labels)

    if cfg.indist_attack is not None and cfg.indist_attack_only:
        # Only use in-distribution adversarial examples
        indist_adv_logits = model(indist_adv_imgs, indist_attack_labels)
        logits = torch.cat([indist_logits, indist_adv_logits])
        targets = torch.cat([indist_target, adv_target])
    elif cfg.indist_attack is not None:
        # Use both in-distribution and out-of-distribution adversarial examples
        adv_input = torch.cat([adv_imgs, indist_adv_imgs])
        adv_labels = torch.cat([attack_labels, indist_attack_labels])
        adv_output = model(adv_input, adv_labels)
        adv_logits, indist_adv_logits = torch.chunk(adv_output, 2)
        logits = torch.cat([indist_logits, adv_logits, indist_adv_logits])
        targets = torch.cat([indist_target, adv_target, adv_target])
    else:
        # Only use out-of-distribution adversarial examples
        if cfg.logsumexp:
            adv_logits = torch.logsumexp(model(adv_imgs, y=None), dim=1)
        else:
            adv_logits = model(adv_imgs, attack_labels)
        logits = torch.cat([indist_logits, adv_logits])
        targets = torch.cat([indist_target, adv_target])

    xent = criterion(logits, targets)
    loss = xent + cfg.r1reg * r1

    ret_metrics_dict = dict(
        loss=loss,
        xent=xent.detach().item(),
        r1=r1.detach().item() if isinstance(r1, torch.Tensor) else r1,
        l2_dist_relative=l2_dist_relative,
        indist_imgs=indist_imgs.detach(),
        outdist_imgs_clean=outdist_imgs,
        adv_imgs=adv_imgs.detach(),
        outdist_imgs_error=compute_img_diff(adv_imgs, outdist_imgs).detach(),
        **metrics.to_simple_dict(),
    )

    return TrainingMetrics(**ret_metrics_dict)


def compute_training_metrics_xent(
    *,
    indist_imgs: torch.Tensor,
    indist_labels: torch.Tensor,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    model: nn.Module,
    cfg: TrainConfig,
    indist_samples_extra: torch.Tensor = None,
    indist_labels_extra: torch.Tensor = None,
) -> Metrics:
    assert (
        cfg.indist_attack_xent.max_steps
        == cfg.indist_attack_xent.fixed_steps
        == cfg.indist_attack_xent.start_step
    )
    indist_adv_imgs = pgd_attack_xent(
        model,
        indist_imgs,
        indist_labels,
        norm="L2",
        eps=cfg.indist_attack_xent.eps,
        step_size=cfg.indist_attack_xent.step_size,
        steps=cfg.indist_attack_xent.max_steps,
    )
    indist_adv_logits = model(indist_adv_imgs)

    # Base loss from adversarial samples
    loss = criterion(indist_adv_logits, indist_labels)

    if indist_samples_extra is not None and indist_labels_extra is not None:
        indist_clean_logits = model(indist_samples_extra)
        loss_clean = criterion(indist_clean_logits, indist_labels_extra)
        loss = loss + loss_clean * 0.1

    return loss


def compute_testing_metrics(
    *,
    indist_imgs: torch.Tensor,
    indist_labels: torch.Tensor,
    outdist_imgs: torch.Tensor,
    outdist_step: int,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    model: nn.Module,
    cfg: TrainConfig,
) -> Metrics:
    assert not model.training
    assert indist_labels.dtype == torch.long

    adv_imgs, attack_labels = generate_outdist_adv_images(
        model, outdist_imgs, cfg, outdist_step, indist_labels=indist_labels
    )

    indist_adv_imgs, indist_attack_labels = None, None
    if cfg.indist_attack is not None:
        indist_adv_imgs, indist_attack_labels = generate_indist_adv_images(
            model, indist_imgs, indist_labels, cfg
        )

    return compute_metrics(
        model,
        indist_imgs,
        indist_labels,
        adv_imgs,
        outdist_imgs,
        attack_labels,
        indist_adv_imgs,
        indist_attack_labels,
    )


def train(cfg: TrainConfig):
    np.random.seed(cfg.rand_seed)
    torch.manual_seed(cfg.rand_seed)
    global_step_one_indexed: int = 0

    # Generate counterfactuals if requested
    if cfg.use_counterfactuals:
        LOGGER.info(
            "Counterfactual generation requested. Initializing model..."
        )
        model = cfg.get_model().to(cfg.device)
        model.eval()  # Ensure model is in evaluation mode

        # Create data loader specifically for counterfactual generation
        train_loader_for_counterfactuals = cfg.get_indist_dataloader(
            split='train',
            shuffle=False,
            augm_type="none",  # No augmentation for clean reference images
        )

        LOGGER.info("Starting counterfactual generation process...")
        generate_counterfactuals(model, train_loader_for_counterfactuals, cfg)
        LOGGER.info("Counterfactual generation completed. Exiting.")
        return
        
    # Perform OOD detection evaluation if requested
    if cfg.evaluate_ood_detection:
        LOGGER.info(
            "OOD detection evaluation requested. Initializing model..."
        )
        model = cfg.get_model().to(cfg.device)
        model.eval()  # Ensure model is in evaluation mode

        # Create data loaders for in-distribution and out-of-distribution data
        indist_loader = cfg.get_indist_dataloader(
            split='test',
            shuffle=False,
            augm_type="none",  # No augmentation for clean evaluation
            balanced=True,
        )

        # Use specified dataset as the out-of-distribution dataset for detection
        LOGGER.info(f"Using {cfg.outdist_dataset_ood_detection} as OOD dataset for detection")
        
        # Set image size based on in-distribution dataset
        if cfg.data.indist_dataset in ["cifar10-conditional", "cifar100-conditional"]:
            size = 32
        elif cfg.data.indist_dataset in ["RestrictedImageNet", "ImageNet"]:
            size = 224
        else:
            # Default size for other datasets
            size = 32
            
        LOGGER.info(f"Using image size {size} for OOD detection based on in-distribution dataset: {cfg.data.indist_dataset}")
        
        match cfg.outdist_dataset_ood_detection:
            case "noise":
                outdist_loader = dl.get_noise_dataset(
                    type="uniform",
                    length=1024,
                    size=size,
                    augm_type="none",
                    batch_size=cfg.batch_size,
                )
            case "svhn":
                outdist_loader = dl.get_SVHN(
                    split='train',
                    batch_size=cfg.batch_size,
                    shuffle=True,
                    augm_type="none",
                    size=size,
                )
            case "cifar100":
                outdist_loader = dl.get_CIFAR100(
                    train=True,
                    batch_size=cfg.batch_size,
                    shuffle=True,
                    augm_type="none",
                    size=size,
                )
            case "cifar10":
                outdist_loader = dl.get_CIFAR10(
                    train=True,
                    batch_size=cfg.batch_size,
                    shuffle=True,
                    augm_type="none",
                    size=size,
                )
            case "imagenet":
                outdist_loader = dl.get_restrictedImageNetOD(
                    train=False,
                    batch_size=cfg.batch_size,
                    shuffle=True,
                    augm_type="none",
                    size=size,
                    path='./data/ImageNet',
                )
            case _:
                raise ValueError(f"Unknown outdist_dataset_ood_detection: {cfg.outdist_dataset_ood_detection}")

        LOGGER.info("Starting OOD detection evaluation...")
        clean_auroc, adv_auroc = ood_detection(model, indist_loader, outdist_loader, cfg)
        LOGGER.info(f"OOD detection evaluation completed. ID: {cfg.data.indist_dataset}, OD: {cfg.outdist_dataset_ood_detection}, Clean AUROC: {clean_auroc:.4f}, Adversarial AUROC: {adv_auroc:.4f}")
        return

    image_generation_metrics = ImageGenerationMetrics()
    classification_metrics = ClassificationMetrics()

    # Create data loaders
    train_indist_loader = cfg.get_indist_dataloader(
        shuffle=True, augm_type=cfg.augm_type_generation
    )
    train_indist_loader_xent = cfg.get_indist_dataloader(
        shuffle=True, augm_type=cfg.augm_type_classification
    )
    train_indist_iter = infinite_iter(train_indist_loader)
    train_indist_iter_xent = infinite_iter(train_indist_loader_xent)
    train_outdist_iter = infinite_iter(cfg.get_outdist_dataloader(shuffle=True))
    # test_indist_iter = infinite_iter(
    #     cfg.get_indist_dataloader(split="val", shuffle=True)
    # )
    # test_outdist_iter = infinite_iter(
    #     cfg.get_outdist_dataloader(split="val", shuffle=True)
    # )

    # Initialize model, criterion, optimizer
    model = cfg.get_model().to(cfg.device)
    if cfg.use_ema:
        non_parallel_avg_model = AveragedModel(
            model.module,
            avg_type="ema",
            ema_decay=0.999,
            avg_batchnorm=True,
            device=cfg.device,
        )

    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    criterion_xent = nn.CrossEntropyLoss(reduction="mean")
    optimizer = cfg.get_optimizer(model)

    get_metrics_shared_kwargs = dict(
        model=model,
        cfg=cfg,
        criterion=criterion,
    )

    get_metrics_shared_kwargs_xent = dict(
        model=model,
        cfg=cfg,
        criterion=criterion_xent,
    )

    # Create evaluation dataloaders outside the training loop
    train_loader_for_eval = cfg.get_indist_dataloader(
        split="train", shuffle=False, augm_type="none"
    )
    if 'cifar' in cfg.data.indist_dataset:
        test_loader_for_eval = cfg.get_indist_dataloader(
            split="val", shuffle=False, augm_type="none"
        )
    else:
        test_loader_for_eval = cfg.get_indist_dataloader(
            split="val", shuffle=False, augm_type="test"
        )

    LOGGER.info(
        f"indist dataset classes: {train_indist_loader.dataset.classes}"
    )

    # Training loop
    # We iteratively increase strength of out-distribution attack
    indist_epoch = 0
    max_epochs_reached = False
    for cur_outdist_steps in range(
        cfg.attack.start_step, cfg.attack.max_steps + 1
    ):
        train_adv_auc_deque = collections.deque(
            maxlen=ceil_div(cfg.min_imgs_per_threshold, cfg.batch_size)
        )
        train_clean_auc_deque = collections.deque(
            maxlen=ceil_div(cfg.min_imgs_per_threshold, cfg.batch_size)
        )
        for local_step, train_indist_batch in enumerate(train_indist_iter):
            global_step_one_indexed += 1
            indist_epoch = (global_step_one_indexed - 1) // len(
                train_indist_loader
            )

            # Interrupt training if indist_epoch exceeds total_epochs (if set)
            if (
                cfg.total_epochs is not None
                and indist_epoch >= cfg.total_epochs
            ):
                LOGGER.info(
                    f"Reached maximum number of epochs ({cfg.total_epochs}). Stopping training."
                )
                max_epochs_reached = True
                break

            # Update learning rate based on current epoch only for SGD optimizer
            if cfg.optimizer == "sgd" and not cfg.fixed_lr:
                current_lr = get_lr_for_epoch(
                    cfg.lr,
                    indist_epoch,
                    cfg.total_epochs,
                    cfg.data.indist_dataset,
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

            # Always log the current learning rate
            current_lr = optimizer.param_groups[0]["lr"]

            wandb.log(
                {
                    "cur_outdist_steps": cur_outdist_steps,
                    "indist_epoch": indist_epoch,
                    "learning_rate": current_lr,
                },
                step=global_step_one_indexed * cfg.batch_size,
            )
            is_metric_logging_step = cfg.should_trigger_event(
                global_step_one_indexed=global_step_one_indexed,
                interval_in_imgs=cfg.n_imgs_per_metrics_log,
            )
            is_image_logging_step = cfg.should_trigger_event(
                global_step_one_indexed=global_step_one_indexed,
                interval_in_imgs=cfg.n_imgs_per_image_log * 100,
            )
            is_fid_logging_step = cfg.should_trigger_event(
                global_step_one_indexed=global_step_one_indexed,
                interval_in_imgs=cfg.n_imgs_per_image_log,
            )
            is_classification_logging_step = (
                cfg.data.num_classes > 1
                and cfg.n_imgs_per_classification_log is not None
                and cfg.should_trigger_event(
                    global_step_one_indexed=global_step_one_indexed,
                    interval_in_imgs=cfg.n_imgs_per_classification_log,
                )
            )

            if cfg.should_trigger_event(
                global_step_one_indexed=global_step_one_indexed,
                interval_in_imgs=cfg.n_imgs_per_ckpt_save,
                at_end=True,
            ):
                if cfg.use_ema:
                    cfg.save_state(
                        model=non_parallel_avg_model.module,
                        optimizer=optimizer,
                        step=global_step_one_indexed,
                    )
                else:
                    cfg.save_state(
                        model=model.module,
                        optimizer=optimizer,
                        step=global_step_one_indexed,
                    )

            if is_fid_logging_step and not cfg.indist_train_only:
                model.zero_grad()

                eval_model = (
                    nn.DataParallel(non_parallel_avg_model)
                    if cfg.use_ema
                    else model
                )
                eval_model.eval()
                fid, gen_imgs = log_generation(eval_model, cfg)
                LOGGER.info(
                    f"FID: {fid}, step: {global_step_one_indexed}, n_imgs: {global_step_one_indexed * cfg.batch_size}"
                )
                if cfg.eval_only:
                    return
                is_new_best_fid = image_generation_metrics.update(fid, gen_imgs)
                if is_new_best_fid:
                    if cfg.use_ema:
                        print('saving EMA model')
                        cfg.save_best_fid_model(eval_model.module.module)
                    else:
                        print('saving regular model')
                        cfg.save_best_fid_model(eval_model.module)
                    LOGGER.info(f"New best FID: {fid}")
                wandb.log(
                    dataclasses.asdict(image_generation_metrics),
                    step=global_step_one_indexed * cfg.batch_size,
                )

            if is_classification_logging_step:
                model.eval()
                model.zero_grad()

                eval_model = (
                    nn.DataParallel(non_parallel_avg_model)
                    if cfg.use_ema
                    else model
                )
                eval_model.eval()

                # Create a dictionary to store all classification metrics
                metrics_dict = {
                    "train_acc": None,
                    "test_acc": None,
                    "train_acc_calib": None,
                    "test_acc_calib": None,
                    "robust_train_acc": None,
                    "robust_test_acc": None,
                }

                # Always evaluate standard (pre-calibration) accuracy
                LOGGER.info("Evaluating standard accuracy...")
                # metrics_dict["train_acc"] = eval_acc(
                #     model=eval_model,
                #     dataloader=train_loader_for_eval,
                #     device=cfg.device,
                # )
                metrics_dict["test_acc"] = eval_acc(
                    model=eval_model,
                    dataloader=test_loader_for_eval,
                    device=cfg.device,
                )
                
                # Free CUDA memory after accuracy evaluation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                LOGGER.info(
                    f"Standard Acc - Train: {metrics_dict['train_acc']}, Test: {metrics_dict['test_acc']:.4f}"
                )

                # Optionally evaluate robust accuracy
                if cfg.robust_eval:
                    LOGGER.info("Evaluating robust accuracy...")
                    # Prepare attack kwargs from indist_attack_xent config if available
                    attack_kwargs = None
                    if cfg.indist_attack_xent is not None:
                        attack_kwargs = {
                            "norm": "L2",
                            "eps": cfg.indist_attack_xent.eps,
                            "step_size": cfg.indist_attack_xent.step_size,
                            "steps": cfg.indist_attack_xent.max_steps,
                            "random_start": False,
                        }

                    # metrics_dict["robust_train_acc"] = eval_robust_acc(
                    #     model=eval_model,
                    #     dataloader=train_loader_for_eval,
                    #     device=cfg.device,
                    #     percentage=20,
                    #     attack_kwargs=attack_kwargs,
                    # )

                    metrics_dict["robust_test_acc"] = eval_robust_acc(
                        model=eval_model,
                        dataloader=test_loader_for_eval,
                        device=cfg.device,
                        percentage=100,
                        attack_kwargs=attack_kwargs,
                    )
                    
                    # Free CUDA memory after robust accuracy evaluation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    LOGGER.info(
                        f"Robust Acc - Train: {metrics_dict['robust_train_acc']}, Test: {metrics_dict['robust_test_acc']:.4f}"
                    )

                # Update all metrics using dictionary unpacking
                is_new_best_acc = classification_metrics.update(**metrics_dict)
                if is_new_best_acc:
                    if cfg.use_ema:
                        cfg.save_best_accuracy_model(eval_model.module.module)
                    else:
                        cfg.save_best_accuracy_model(eval_model.module)

                wandb.log(
                    dataclasses.asdict(classification_metrics),
                    step=global_step_one_indexed * cfg.batch_size,
                )

            train_outdist_imgs = (
                None
                if cfg.indist_train_only
                else next(train_outdist_iter)[0].to(cfg.device)
            )
            train_indist_imgs = train_indist_batch[0].to(cfg.device)
            train_indist_labels = train_indist_batch[1]

            optimizer.zero_grad()
            if cfg.indist_train_only:
                # Use batch normalization if training only on in-distribution data
                model.train()
                train_indist_imgs_xent, train_indist_labels_xent = next(
                    train_indist_iter_xent
                )
                train_indist_imgs_xent = train_indist_imgs_xent.to(cfg.device)
                train_indist_labels_xent = train_indist_labels_xent.to(
                    cfg.device
                )
                xent_loss = compute_training_metrics_xent(
                    indist_imgs=train_indist_imgs_xent,
                    indist_labels=train_indist_labels_xent,
                    **get_metrics_shared_kwargs_xent,
                )
                xent_loss.backward()
                optimizer.step()

                if cfg.use_ema:
                    with torch.no_grad():
                        non_parallel_avg_model.update_parameters(model.module)
                
                # print(f"Step {global_step_one_indexed:04d} - Training Loss (indist_only): {xent_loss.item():.6f}")
                
                if global_step_one_indexed % 20 == 0:
                    LOGGER.info(
                        f"Step {global_step_one_indexed:04d} - "
                        f"xent_loss: {xent_loss.item():.5f}"
                    )
                if is_image_logging_step:
                    for label, imgs in [
                        ("train_indist_imgs_xent", train_indist_imgs_xent),
                    ]:
                        # Use white padding for CIFAR10
                        padding = (
                            0 if "cifar10" in cfg.data.indist_dataset else 2
                        )
                        image_grid = torchvision.utils.make_grid(
                            imgs[:10], nrow=10, padding=padding
                        )
                        wandb.log(
                            {label: wandb.Image(image_grid)},
                            step=global_step_one_indexed * cfg.batch_size,
                        )
                continue

            # Hybrid training
            train_metrics = compute_training_metrics(
                indist_imgs=train_indist_imgs,
                indist_labels=train_indist_labels,
                outdist_imgs=train_outdist_imgs,
                outdist_step=cur_outdist_steps,
                **get_metrics_shared_kwargs,
            )
            # TODO: train_indist_imgs.requires_grad is true at this point
            if cfg.indist_attack_xent is not None:
                train_indist_imgs_xent, train_indist_labels_xent = next(
                    train_indist_iter_xent
                )
                train_indist_imgs_xent = train_indist_imgs_xent.to(cfg.device)
                train_indist_labels_xent = train_indist_labels_xent.to(
                    cfg.device
                )

                # If indist_clean_extra flag is set, get additional clean samples
                indist_samples_extra = None
                indist_labels_extra = None
                if cfg.indist_clean_extra:
                    indist_batch_extra = next(train_indist_iter_xent)
                    indist_samples_extra = indist_batch_extra[0].to(cfg.device)
                    indist_labels_extra = indist_batch_extra[1].to(cfg.device)

                xent_loss = compute_training_metrics_xent(
                    indist_imgs=train_indist_imgs_xent,
                    indist_labels=train_indist_labels_xent,
                    indist_samples_extra=indist_samples_extra,
                    indist_labels_extra=indist_labels_extra,
                    **get_metrics_shared_kwargs_xent,
                )
            else:
                xent_loss = 0.0
            train_adv_auc_deque.append(train_metrics.adv_auc)
            train_clean_auc_deque.append(train_metrics.clean_auc)
            (train_metrics.loss * cfg.bce_weight + xent_loss * cfg.xent_lr_multiplier).backward()
            optimizer.step()

            if cfg.use_ema:
                with torch.no_grad():
                    non_parallel_avg_model.update_parameters(model.module)

            # print(f"Step {global_step_one_indexed:04d} - Training Loss: {train_metrics.loss.item():.6f}")

            if global_step_one_indexed % 20 == 0:
                train_metrics_dict = train_metrics.to_simple_dict()
                metrics_str = ", ".join(
                    [
                        f"{k}: {float(v):.5f}"
                        for k, v in train_metrics_dict.items()
                    ]
                )
                LOGGER.info(
                    f"Step {global_step_one_indexed:04d} - "
                    f"cur_outdist_steps: {cur_outdist_steps}, "
                    f"train_adv_auc_mean: {np.mean(train_adv_auc_deque):.2f}, "
                    f"train_clean_auc_mean: {np.mean(train_clean_auc_deque):.2f}, "
                    f"{metrics_str}"
                )

            # Define the log_interval in the config
            if is_metric_logging_step:
                wandb.log(
                    dict_append_label(train_metrics.to_simple_dict(), "train_"),
                    step=global_step_one_indexed * cfg.batch_size,
                )

                # model.zero_grad()
                # model.eval()
                # test_indist_batch = next(test_indist_iter)
                # test_outdist_imgs = cfg.get_outdist_images(
                #     next(test_outdist_iter)
                # )

                # test_metrics = compute_testing_metrics(
                #     indist_imgs=test_indist_batch[0].to(cfg.device),
                #     indist_labels=test_indist_batch[1],
                #     outdist_imgs=test_outdist_imgs,
                #     outdist_step=cur_outdist_steps,
                #     **get_metrics_shared_kwargs,
                # )
                # test_fixed_metrics = compute_testing_metrics(
                #     indist_imgs=test_indist_batch[0].to(cfg.device),
                #     indist_labels=test_indist_batch[1],
                #     outdist_imgs=test_outdist_imgs,
                #     outdist_step=cfg.attack.fixed_steps,
                #     **get_metrics_shared_kwargs,
                # )

                # wandb.log(
                #     dict_append_label(train_metrics.to_simple_dict(), "train_")
                #     | dict_append_label(test_metrics.to_simple_dict(), "test_")
                #     | dict_append_label(
                #         test_fixed_metrics.to_simple_dict(), "test_fixed_"
                #     )
                #     | dict(
                #         local_imgs=(local_step + 1) * cfg.batch_size,
                #     ),
                #     step=global_step_one_indexed * cfg.batch_size,
                # )

            # Log images infrequently
            if is_image_logging_step:
                for label, imgs in [
                    ("train_indist_imgs_xent", train_indist_imgs_xent),
                    ("train_indist_imgs", train_metrics.indist_imgs),
                    ("train_outdist_imgs", train_metrics.outdist_imgs_clean),
                    ("train_error_imgs", train_metrics.outdist_imgs_error),
                    ("train_adv_imgs", train_metrics.adv_imgs),
                    ("train_gen_imgs", image_generation_metrics.gen_imgs),
                ]:
                    # Use white padding for CIFAR10
                    padding = 0 if "cifar10" in cfg.data.indist_dataset else 2
                    image_grid = torchvision.utils.make_grid(
                        imgs[:10], nrow=10, padding=padding
                    )
                    wandb.log(
                        {label: wandb.Image(image_grid)},
                        step=global_step_one_indexed * cfg.batch_size,
                    )

            if (
                cur_outdist_steps < cfg.attack.max_steps
                and cfg.samples_per_attack_step is not None
                and (local_step + 1) * cfg.batch_size
                >= cfg.samples_per_attack_step
            ):
                LOGGER.info(
                    f"Outdist step {cur_outdist_steps} reached max samples {cfg.samples_per_attack_step}"
                )
                break  # breaks the iteration loop

            # Break if we have reached the auc threshold
            if (
                cur_outdist_steps < cfg.attack.max_steps
                and np.mean(train_adv_auc_deque) >= cfg.AUC_th
                and len(train_adv_auc_deque) == train_adv_auc_deque.maxlen
            ):
                LOGGER.info(
                    f"Adv AUC reached threshold {cfg.AUC_th} on local step {local_step} outdist step {cur_outdist_steps}"
                )
                break

        # Check if max epochs was reached in the inner loop
        if max_epochs_reached:
            break


# Example: python rebm/training/train.py --config="rebm/training/lsun-baselines/stargan.yaml" --image_log_num_steps=10
if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--image_log_num_steps", type=int, help="Override image_log.num_steps"
    )
    parser.add_argument(
        "--image_log_num_samples",
        type=int,
        help="Override image_log.num_samples",
    )
    parser.add_argument(
        "--image_log_ood_data_dir", type=str, help="Override image_log.ood_data_dir"
    )
    parser.add_argument(
        "--image_log_target_class", type=int, help="Override image_log.target_class"
    )
    parser.add_argument(
        "--ckpt_path", type=str, help="Override model.ckpt_path"
    )
    parser.add_argument(
        "--model_type", type=str, help="Override model.model_type"
    )
    parser.add_argument(
        "--indist_attack_eps", type=float, help="Override indist_attack.eps"
    )
    parser.add_argument(
        "--indist_attack_steps", type=int, help="Override indist_attack.fixed_steps"
    )
    parser.add_argument(
        "--generate_counterfactuals",
        action="store_true",
        help="Generate counterfactual examples instead of training",
    )
    parser.add_argument(
        "--evaluate_ood_detection",
        action="store_true", 
        help="Evaluate OOD detection instead of training",
    )
    parser.add_argument(
        "--ood_detection_logsumexp",
        action="store_true", 
        help="Evaluate OOD detection instead of training",
    )
    parser.add_argument(
        "--outdist_dataset_ood_detection",
        type=str,
        choices=["noise", "svhn", "cifar100", "cifar10", "imagenet"],
        help="Dataset to use for OOD detection evaluation"
    )
    parser.add_argument(
        "--logsumexp_sampling",
        action="store_true",
        help="Override cfg.logsumexp_sampling to use logsumexp for sampling"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Override trainconfig batch_size"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["sgd", "adam", "adamw"],
        help="Override optimizer"
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Override learning rate"
    )
    parser.add_argument(
        "--wd",
        type=float,
        help="Override weight decay"
    )
    parser.add_argument(
        "--auc_th",
        type=float,
        help="Override AUC_th threshold"
    )
    args = parser.parse_args()
    config_file = args.config
    cfg = TrainConfig.from_yaml(config_file)

    # Override image_log.num_steps if specified
    if args.image_log_num_steps is not None:
        LOGGER.info(
            f"Overriding image_log.num_steps from {cfg.image_log.num_steps} to {args.image_log_num_steps}"
        )
        cfg.image_log.num_steps = args.image_log_num_steps

    # Override image_log.num_samples if specified
    if args.image_log_num_samples is not None:
        LOGGER.info(
            f"Overriding image_log.num_samples from {cfg.image_log.num_samples} to {args.image_log_num_samples}"
        )
        cfg.image_log.num_samples = args.image_log_num_samples

    # Override image_log.ood_data_dir if specified
    if args.image_log_ood_data_dir is not None:
        LOGGER.info(
            f"Overriding image_log.ood_data_dir from {cfg.image_log.ood_data_dir} to {args.image_log_ood_data_dir}"
        )
        cfg.image_log.ood_data_dir = args.image_log_ood_data_dir

    # Override image_log.target_class if specified
    if args.image_log_target_class is not None:
        LOGGER.info(
            f"Overriding image_log.target_class from {cfg.image_log.target_class} to {args.image_log_target_class}"
        )
        cfg.image_log.target_class = args.image_log_target_class

    # Override model.ckpt_path if specified
    if args.ckpt_path is not None:
        LOGGER.info(
            f"Overriding model.ckpt_path from {cfg.model.ckpt_path} to {args.ckpt_path}"
        )
        cfg.model.ckpt_path = args.ckpt_path

    # Override model.model_type if specified
    if args.model_type is not None:
        LOGGER.info(
            f"Overriding model.model_type from {cfg.model.model_type} to {args.model_type}"
        )
        cfg.model.model_type = args.model_type

    # Override indist_attack.eps if specified
    if args.indist_attack_eps is not None:
        old_eps = cfg.indist_attack.eps
        cfg.indist_attack.eps = args.indist_attack_eps
        LOGGER.info(
            f"Overriding indist_attack.eps from {old_eps} to {args.indist_attack_eps}"
        )
        
    # Override indist_attack.fixed_steps if specified
    if args.indist_attack_steps is not None:
        old_steps = cfg.indist_attack.fixed_steps
        cfg.indist_attack.fixed_steps = args.indist_attack_steps
        LOGGER.info(
            f"Overriding indist_attack.fixed_steps from {old_steps} to {args.indist_attack_steps}"
        )

    # Set use_counterfactuals flag if specified
    if args.generate_counterfactuals:
        LOGGER.info(
            "Setting use_counterfactuals to True based on command-line argument"
        )
        cfg.use_counterfactuals = True
        
    # Set evaluate_ood_detection flag if specified
    if args.evaluate_ood_detection:
        LOGGER.info(
            "Setting evaluate_ood_detection to True based on command-line argument"
        )
        cfg.evaluate_ood_detection = True
    if args.ood_detection_logsumexp:
        LOGGER.info(
            "Setting ood_detection_logsumexp to True based on command-line argument"
        )
        cfg.ood_detection_logsumexp = True
        
    # Override outdist_dataset_ood_detection if specified
    if args.outdist_dataset_ood_detection is not None:
        LOGGER.info(
            f"Overriding outdist_dataset_ood_detection from {cfg.outdist_dataset_ood_detection} to {args.outdist_dataset_ood_detection}"
        )
        cfg.outdist_dataset_ood_detection = args.outdist_dataset_ood_detection
        
    # Override logsumexp_sampling if specified
    if args.logsumexp_sampling:
        LOGGER.info(
            f"Overriding logsumexp_sampling from {cfg.logsumexp_sampling} to True"
        )
        cfg.logsumexp_sampling = True

    # Override batch_size if specified
    if args.batch_size is not None:
        LOGGER.info(
            f"Overriding batch_size from {cfg.batch_size} to {args.batch_size}"
        )
        cfg.batch_size = args.batch_size

    # Override optimizer if specified
    if args.optimizer is not None:
        LOGGER.info(
            f"Overriding optimizer from {cfg.optimizer} to {args.optimizer}"
        )
        cfg.optimizer = args.optimizer

    # Override lr if specified
    if args.lr is not None:
        LOGGER.info(
            f"Overriding lr from {cfg.lr} to {args.lr}"
        )
        cfg.lr = args.lr

    # Override wd if specified
    if args.wd is not None:
        LOGGER.info(
            f"Overriding wd from {cfg.wd} to {args.wd}"
        )
        cfg.wd = args.wd

    # Override AUC_th if specified
    if args.auc_th is not None:
        LOGGER.info(
            f"Overriding AUC_th from {cfg.AUC_th} to {args.auc_th}"
        )
        cfg.AUC_th = args.auc_th

    # Don't upload .pth files to wandb, since they are big
    # We just save them on disk for now.
    os.environ["WANDB_IGNORE_GLOBS"] = "*.pth"

    # Initialize wandb
    # Use WANDB_NAME environment variable if set, otherwise use config file stem
    run_name = os.environ.get("WANDB_NAME", Path(config_file).stem)
    wandb.init(
        project=cfg.wandb_project,
        tags=cfg.tags,
        dir=cfg.wandb_dir,
        save_code=True,
        mode="disabled" if cfg.wandb_disabled else "online",
        name=run_name,
    )
    print(recursive_asdict(cfg))
    wandb.config.update(recursive_asdict(cfg))
    LOGGER.info(f"Using device: {cfg.device}")

    # setting benchmark to True enables better performance on fixed input sizes.
    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    torch.backends.cudnn.benchmark = True

    train(cfg)
