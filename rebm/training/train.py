import sys

sys.path.insert(0, "pytorch-image-models")
from timm.models.resnet import resnet50 as ResNet50ImageNet
from timm.models.resnet import wide_resnet50_2 as WideResNet50x2ImageNet
from timm.models.resnet import wide_resnet50_4 as WideResNet50x4ImageNet
from timm.models.convnext import convnext_tiny, convnext_base, convnext_small, convnext_large

import collections
import copy
import dataclasses
import hashlib
import json
import logging
import math
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import uuid
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Literal, Optional, Tuple

import einops
# import kornia.augmentation as K
from omegaconf import OmegaConf
import numpy as np
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
import rebm.training.modeling
from rebm.training.adv_attacks import pgd_attack, pgd_attack_xent
from rebm.training.metrics import (
    ClassificationMetrics,
    ImageGenerationMetrics,
    Metrics,
    TrainingMetrics,
    compute_metrics,
    compute_testing_metrics,
    compute_training_metrics,
    compute_training_metrics_xent,
)
from rebm.training.scheduling import get_lr_for_epoch, should_trigger_event
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
    generate_counterfactuals,
    generate_images,
    generate_indist_adv_images,
    generate_outdist_adv_images,
    get_auc,
    log_generate_images,
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



def infinite_iter(iterable: Iterable):
    while True:
        for x in iterable:
            yield x


def dict_append_label(d: dict, label: str) -> dict:
    return {label + k: v for k, v in d.items()}


@dataclasses.dataclass
class TrainConfig:
    """Default parameters are for training on lsun-bird. See defaults in baselines/*.yaml"""

    # Required parameters
    data: DataConfig
    attack: AttackConfig
    model: BaseModelConfig
    image_log: ImageLogConfig

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
    tinyimages_loader: str = "innout"
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
        # Note: image_log.save_dir will be set later after wandb.init() using wandb run ID
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


def train(cfg: TrainConfig):
    np.random.seed(cfg.rand_seed)
    torch.manual_seed(cfg.rand_seed)
    global_step_one_indexed: int = 0

    # Generate counterfactuals if requested
    if cfg.use_counterfactuals:
        LOGGER.info(
            "Counterfactual generation requested. Initializing model..."
        )
        model = rebm.training.modeling.get_model(
            model_config=cfg.model,
            device=cfg.device,
            num_classes=cfg.data.num_classes,
            indist_dataset=cfg.data.indist_dataset,
            resume_path=cfg.resume_path,
        ).to(cfg.device)
        model.eval()  # Ensure model is in evaluation mode

        # Create data loader specifically for counterfactual generation
        train_loader_for_counterfactuals = rebm.training.data.get_indist_dataloader(
            config=cfg.data,
            batch_size=cfg.batch_size,
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
        model = rebm.training.modeling.get_model(
            model_config=cfg.model,
            device=cfg.device,
            num_classes=cfg.data.num_classes,
            indist_dataset=cfg.data.indist_dataset,
            resume_path=cfg.resume_path,
        ).to(cfg.device)
        model.eval()  # Ensure model is in evaluation mode

        # Create data loaders for in-distribution and out-of-distribution data
        indist_loader = rebm.training.data.get_indist_dataloader(
            config=cfg.data,
            batch_size=cfg.batch_size,
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
    train_indist_loader = rebm.training.data.get_indist_dataloader(
        config=cfg.data,
        batch_size=cfg.batch_size,
        shuffle=True,
        augm_type=cfg.augm_type_generation
    )
    train_indist_loader_xent = rebm.training.data.get_indist_dataloader(
        config=cfg.data,
        batch_size=cfg.batch_size,
        shuffle=True,
        augm_type=cfg.augm_type_classification
    )
    train_indist_iter = infinite_iter(train_indist_loader)
    train_indist_iter_xent = infinite_iter(train_indist_loader_xent)
    train_outdist_iter = infinite_iter(rebm.training.data.get_outdist_dataloader(
        config=cfg.data,
        batch_size=cfg.batch_size,
        shuffle=True,
        augm_type_generation=cfg.augm_type_generation,
        tinyimages_loader=cfg.tinyimages_loader,
        openimages_max_samples=cfg.openimages_max_samples,
        openimages_augm=cfg.openimages_augm,
    ))
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
    optimizer = rebm.training.modeling.get_optimizer(
        model=model,
        optimizer_name=cfg.optimizer,
        lr=cfg.lr,
        wd=cfg.wd,
        resume_path=cfg.resume_path,
    )

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
    train_loader_for_eval = rebm.training.data.get_indist_dataloader(
        config=cfg.data,
        batch_size=cfg.batch_size,
        split="train",
        shuffle=False,
        augm_type="none"
    )
    if 'cifar' in cfg.data.indist_dataset:
        test_loader_for_eval = rebm.training.data.get_indist_dataloader(
            config=cfg.data,
            batch_size=cfg.batch_size,
            split="val",
            shuffle=False,
            augm_type="none"
        )
    else:
        test_loader_for_eval = rebm.training.data.get_indist_dataloader(
            config=cfg.data,
            batch_size=cfg.batch_size,
            split="val",
            shuffle=False,
            augm_type="test"
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
            maxlen=math.ceil(cfg.min_imgs_per_threshold / cfg.batch_size)
        )
        train_clean_auc_deque = collections.deque(
            maxlen=math.ceil(cfg.min_imgs_per_threshold / cfg.batch_size)
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
            is_metric_logging_step = should_trigger_event(
                global_step_one_indexed=global_step_one_indexed,
                batch_size=cfg.batch_size,
                interval_in_imgs=cfg.n_imgs_per_metrics_log,
            )
            is_image_logging_step = should_trigger_event(
                global_step_one_indexed=global_step_one_indexed,
                batch_size=cfg.batch_size,
                interval_in_imgs=cfg.n_imgs_per_image_log * 100,
            )
            is_fid_logging_step = should_trigger_event(
                global_step_one_indexed=global_step_one_indexed,
                batch_size=cfg.batch_size,
                interval_in_imgs=cfg.n_imgs_per_image_log,
            )
            is_classification_logging_step = (
                cfg.data.num_classes > 1
                and cfg.n_imgs_per_classification_log is not None
                and should_trigger_event(
                    global_step_one_indexed=global_step_one_indexed,
                    batch_size=cfg.batch_size,
                    interval_in_imgs=cfg.n_imgs_per_classification_log,
                )
            )

            if should_trigger_event(
                global_step_one_indexed=global_step_one_indexed,
                batch_size=cfg.batch_size,
                interval_in_imgs=cfg.n_imgs_per_ckpt_save,
                at_end=True,
            ):
                if cfg.use_ema:
                    rebm.training.modeling.save_checkpoint(
                        model=non_parallel_avg_model.module,
                        optimizer=optimizer,
                        step=global_step_one_indexed,
                    )
                else:
                    rebm.training.modeling.save_checkpoint(
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
                        rebm.training.modeling.save_best_fid_model(eval_model.module.module)
                    else:
                        print('saving regular model')
                        rebm.training.modeling.save_best_fid_model(eval_model.module)
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
                        rebm.training.modeling.save_best_accuracy_model(eval_model.module.module)
                    else:
                        rebm.training.modeling.save_best_accuracy_model(eval_model.module)

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


if __name__ == "__main__":
    import sys
    from rebm.training.config_classes import DataConfig, AttackConfig, ImageLogConfig, create_model_config

    # Parse arguments
    args = sys.argv[1:]

    if not args or args[0] in ['-h', '--help']:
        print("Training script with OmegaConf for configuration management")
        print("\nUsage: python -m rebm.training.train CONFIG_FILE [KEY=VALUE ...]")
        print("\nExamples:")
        print("  # Basic usage")
        print("  python -m rebm.training.train experiments/cifar10/config.yaml")
        print("\n  # Override top-level fields")
        print("  python -m rebm.training.train experiments/cifar10/config.yaml batch_size=256 lr=0.001")
        print("\n  # Override nested fields (use dot notation)")
        print("  python -m rebm.training.train experiments/cifar10/config.yaml model.ckpt_path=/path/to/model.pth")
        print("  python -m rebm.training.train experiments/cifar10/config.yaml image_log.num_steps=50 attack.max_steps=100")
        sys.exit(0)

    # First argument is the config file
    config_file = args[0]

    # Remaining arguments are overrides
    overrides = args[1:]

    # Load YAML config with OmegaConf
    omega_cfg = OmegaConf.load(config_file)

    # Apply overrides using OmegaConf's dotlist
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        omega_cfg = OmegaConf.merge(omega_cfg, override_cfg)

    # Convert configs to plain containers for logging and dataclass instantiation
    config_for_wandb = OmegaConf.to_container(omega_cfg, resolve=True)
    config_dict = copy.deepcopy(config_for_wandb)

    # Create nested configs manually to ensure proper types
    config_dict['data'] = DataConfig(**config_dict.get('data', {}))
    config_dict['attack'] = AttackConfig(**config_dict.get('attack', {}))
    config_dict['model'] = create_model_config(config_dict.get('model', {}))
    config_dict['image_log'] = ImageLogConfig(**config_dict.get('image_log', {}))

    # Handle optional attack configs
    if 'indist_attack' in config_dict and config_dict['indist_attack'] is not None:
        config_dict['indist_attack'] = AttackConfig(**config_dict['indist_attack'])
    if 'indist_attack_xent' in config_dict and config_dict['indist_attack_xent'] is not None:
        config_dict['indist_attack_xent'] = AttackConfig(**config_dict['indist_attack_xent'])

    cfg = TrainConfig(**config_dict)

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
        config=config_for_wandb,
    )

    # Set image_log.save_dir using wandb run ID if not specified
    if cfg.image_log.save_dir is None:
        cfg.image_log.save_dir = f"{cfg.wandb_dir}/eval_fid/{wandb.run.id}"
    print(config_for_wandb)
    LOGGER.info(f"Using device: {cfg.device}")

    # setting benchmark to True enables better performance on fixed input sizes.
    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    torch.backends.cudnn.benchmark = True

    train(cfg)
