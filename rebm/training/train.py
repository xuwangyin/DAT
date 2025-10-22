import collections
import dataclasses
import logging
import math
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Iterable

os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.insert(0, "pytorch-image-models")

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data
import torchvision.utils
import wandb
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

import rebm.training.data
import rebm.training.misc
import rebm.training.modeling
from rebm.eval.eval_utils import (
    eval_acc,
    eval_robust_acc,
    evaluate_image_generation,
    log_generate_images,
)
from rebm.training.average_model import AveragedModel
from rebm.training.config_classes import TrainConfig, load_train_config
from rebm.training.metrics import (
    ClassificationMetrics,
    ImageGenerationMetrics,
    compute_clf_adv_loss,
    compute_ebm_metrics,
)
from rebm.training.scheduling import should_trigger_event

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class Dataloaders:
    """Container for all dataloaders used during training."""

    indist_loader_ebm: torch.utils.data.DataLoader
    indist_loader_clf: torch.utils.data.DataLoader
    indist_iter_ebm: Iterable
    indist_iter_clf: Iterable
    outdist_iter: Iterable
    test_loader: torch.utils.data.DataLoader


def infinite_iter(iterable: Iterable):
    while True:
        for x in iterable:
            yield x


def dict_append_label(d: dict, label: str) -> dict:
    return {label + k: v for k, v in d.items()}


def average_metric_across_ranks(value: float, device: torch.device, world_size: int) -> float:
    """Average a scalar metric across all DDP ranks."""
    tensor = torch.tensor([value], device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor / world_size).item()


def setup_distributed(cfg: TrainConfig) -> tuple[int, int, torch.device]:
    """Setup distributed training environment.

    Note: When using DDP, process group is initialized in __main__ before this function.

    Returns:
        tuple: (rank, world_size, device)
    """
    if cfg.use_ddp:
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Set device for this process
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        if rank == 0:
            LOGGER.info(f"Initialized DDP with {world_size} processes")
            if world_size == 1:
                LOGGER.warning(
                    "DDP enabled with only 1 GPU. Consider setting use_ddp=False "
                    "for better performance on single GPU."
                )
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        LOGGER.info("Using DataParallel mode (non-distributed)")

    return rank, world_size, device


def create_dataloaders(cfg: TrainConfig, rank: int = 0, world_size: int = 1) -> Dataloaders:
    """Create all dataloaders needed for training.

    Args:
        cfg: Training configuration
        rank: Process rank for DDP
        world_size: Total number of processes for DDP

    Returns:
        Dataloaders instance containing:
            - indist_loader_ebm: Loader for in-distribution training (generation augmentation)
            - indist_loader_clf: Loader for in-distribution training (classification augmentation)
            - indist_iter_ebm: Infinite iterator for indist_loader_ebm
            - indist_iter_clf: Infinite iterator for indist_loader_clf
            - outdist_iter: Infinite iterator for out-distribution data
            - test_loader: Loader for test set evaluation
    """
    # Calculate per-device batch size for DDP
    # Config batch_size is the total/global batch size across all GPUs
    if cfg.batch_size % world_size != 0:
        raise ValueError(
            f"batch_size ({cfg.batch_size}) must be divisible by world_size ({world_size})"
        )
    per_device_batch_size = cfg.batch_size // world_size

    if rank == 0:
        LOGGER.info(
            f"Batch size - Per device: {per_device_batch_size}, "
            f"Total across {world_size} device(s): {cfg.batch_size}"
        )

    train_indist_loader = rebm.training.data.get_indist_dataloader(
        config=cfg.data,
        batch_size=per_device_batch_size,
        shuffle=True,
        augm_type=cfg.augm_type_generation,
        use_ddp=cfg.use_ddp,
        rank=rank,
        world_size=world_size,
    )
    train_indist_loader_clf = rebm.training.data.get_indist_dataloader(
        config=cfg.data,
        batch_size=per_device_batch_size,
        shuffle=True,
        augm_type=cfg.augm_type_classification,
        use_ddp=cfg.use_ddp,
        rank=rank,
        world_size=world_size,
    )
    train_outdist_iter = infinite_iter(
        rebm.training.data.get_outdist_dataloader(
            config=cfg.data,
            batch_size=per_device_batch_size,
            shuffle=True,
            augm_type_generation=cfg.augm_type_generation,
            tinyimages_loader=cfg.tinyimages_loader,
            openimages_max_samples=cfg.openimages_max_samples,
            openimages_augm=cfg.openimages_augm,
            use_ddp=cfg.use_ddp,
            rank=rank,
            world_size=world_size,
        )
    )
    test_loader_for_eval = rebm.training.data.get_indist_dataloader(
        config=cfg.data,
        batch_size=per_device_batch_size,
        split="val",
        shuffle=False,
        augm_type="none" if "cifar" in cfg.data.indist_dataset else "test",
        use_ddp=cfg.use_ddp,
        rank=rank,
        world_size=world_size,
    )

    return Dataloaders(
        indist_loader_ebm=train_indist_loader,
        indist_loader_clf=train_indist_loader_clf,
        indist_iter_ebm=infinite_iter(train_indist_loader),
        indist_iter_clf=infinite_iter(train_indist_loader_clf),
        outdist_iter=train_outdist_iter,
        test_loader=test_loader_for_eval,
    )


def log_image_grid(
    label: str, imgs: torch.Tensor, cfg: TrainConfig, step: int
) -> None:
    """Log image grid to wandb.

    Args:
        label: Label for the wandb log
        imgs: Image tensor to log (will use first 10 images)
        cfg: Training configuration
        step: Global step for logging

    Note: Should only be called on rank 0 in DDP mode.
    """
    padding = 0 if "cifar10" in cfg.data.indist_dataset else 2
    image_grid = torchvision.utils.make_grid(
        imgs[:10], nrow=10, padding=padding
    )
    wandb.log({label: wandb.Image(image_grid)}, step=step)


def evaluate_and_log_fid(
    model_to_eval: nn.Module,
    cfg: TrainConfig,
    image_generation_metrics: ImageGenerationMetrics,
    global_step: int,
) -> None:
    """Evaluate FID and save best model if improved.

    Note: Should only be called on rank 0 in DDP mode.

    Args:
        model_to_eval: Model to evaluate (EMA or training model)
        cfg: Training configuration
        image_generation_metrics: Metrics tracker for image generation
        global_step: Current global step (1-indexed)
    """

    model_to_eval.eval()
    n_imgs_seen = global_step * cfg.batch_size

    fid = evaluate_image_generation(model_to_eval, cfg)
    LOGGER.info(f"FID: {fid}, step: {global_step}, n_imgs: {n_imgs_seen}")

    is_new_best = image_generation_metrics.update(fid)
    if is_new_best:
        model_to_save = (
            model_to_eval.module.module if cfg.use_ema else model_to_eval.module
        )
        LOGGER.info(
            f"Saving {'EMA' if cfg.use_ema else 'regular'} model with best FID"
        )
        rebm.training.modeling.save_best_fid_model(model_to_save)
        LOGGER.info(f"New best FID: {fid}")

    wandb.log(
        dataclasses.asdict(image_generation_metrics),
        step=n_imgs_seen,
    )


def evaluate_and_log_accuracy(
    model_to_eval: nn.Module,
    cfg: TrainConfig,
    test_loader_for_eval,
    classification_metrics: ClassificationMetrics,
    global_step: int,
) -> None:
    """Evaluate classification accuracy and save best model if improved.

    Note: Should only be called on rank 0 in DDP mode.

    Args:
        model_to_eval: Model to evaluate (EMA or training model)
        cfg: Training configuration
        test_loader_for_eval: Test dataloader
        classification_metrics: Metrics tracker for classification
        global_step: Current global step (1-indexed)
    """

    model_to_eval.eval()
    n_imgs_seen = global_step * cfg.batch_size

    LOGGER.info("Evaluating standard accuracy...")
    # Get device from config's device property (which will use cuda:0 by default)
    eval_device = cfg.device
    test_acc = eval_acc(
        model=model_to_eval,
        dataloader=test_loader_for_eval,
        device=eval_device,
    )

    # Free CUDA memory after accuracy evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    LOGGER.info(f"Test Acc: {test_acc:.4f}")

    robust_test_acc = None
    if cfg.robust_eval:
        LOGGER.info("Evaluating robust accuracy...")
        attack_kwargs = {
            "norm": "L2",
            "eps": cfg.indist_attack_clf.eps,
            "step_size": cfg.indist_attack_clf.step_size,
            "steps": cfg.indist_attack_clf.max_steps,
            "random_start": False,
        }

        robust_test_acc = eval_robust_acc(
            model=model_to_eval,
            dataloader=test_loader_for_eval,
            device=eval_device,
            percentage=100,
            attack_kwargs=attack_kwargs,
        )

        # Free CUDA memory after robust accuracy evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        LOGGER.info(f"Robust Test Acc: {robust_test_acc:.4f}")

    is_new_best_acc = classification_metrics.update(
        test_acc=test_acc,
        robust_test_acc=robust_test_acc,
    )
    if is_new_best_acc:
        model_to_save = (
            model_to_eval.module.module if cfg.use_ema else model_to_eval.module
        )
        rebm.training.modeling.save_best_accuracy_model(model_to_save)

    wandb.log(
        dataclasses.asdict(classification_metrics),
        step=n_imgs_seen,
    )


def train(cfg: TrainConfig):
    # Setup distributed training
    rank, world_size, device = setup_distributed(cfg)

    # Set seeds (different for each rank for data augmentation diversity)
    np.random.seed(cfg.rand_seed + rank)
    torch.manual_seed(cfg.rand_seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.rand_seed + rank)

    # Initialize training state (may be overridden by resume)
    global_step_one_indexed: int = 0
    start_outdist_steps: int = cfg.attack.start_step

    image_generation_metrics = ImageGenerationMetrics()
    classification_metrics = ClassificationMetrics()

    # Create all dataloaders
    dataloaders = create_dataloaders(cfg, rank=rank, world_size=world_size)
    indist_loader_ebm = dataloaders.indist_loader_ebm
    indist_iter_ebm = dataloaders.indist_iter_ebm
    indist_iter_clf = dataloaders.indist_iter_clf
    outdist_iter = dataloaders.outdist_iter
    test_loader = dataloaders.test_loader

    model = rebm.training.modeling.get_model(
        model_config=cfg.model,
        device=device,
        num_classes=cfg.data.num_classes,
        indist_dataset=cfg.data.indist_dataset,
        use_ddp=cfg.use_ddp,
        rank=rank,
    )
    if cfg.use_ema:
        # EMA model wraps the underlying model (before DDP wrapper)
        base_model = model.module if cfg.use_ddp else model.module
        ema_model = AveragedModel(
            base_model,
            avg_type="ema",
            ema_decay=0.999,
            avg_batchnorm=True,
            device=device,
        )

    criterion_ebm = nn.BCEWithLogitsLoss(reduction="mean")
    criterion_clf = nn.CrossEntropyLoss(reduction="mean")
    optimizer = rebm.training.modeling.get_optimizer(
        model=model,
        optimizer_name=cfg.optimizer,
        lr=cfg.lr,
        wd=cfg.wd,
    )

    # Resume from saved state if specified
    if cfg.resume_from_state:
        if rank == 0:
            LOGGER.info(f"Resuming training from {cfg.resume_from_state}")
        training_state = rebm.training.modeling.load_training_state(
            state_path=cfg.resume_from_state,
            model=model,
            optimizer=optimizer,
            device=device,
            ema_model=ema_model if cfg.use_ema else None,
        )
        # Restore training state
        global_step_one_indexed = training_state['global_step']
        start_outdist_steps = training_state['cur_outdist_steps']

        # Restore metrics
        metrics_dict = training_state['metrics']
        image_generation_metrics = ImageGenerationMetrics(
            **metrics_dict['image_generation_metrics']
        )
        classification_metrics = ClassificationMetrics(
            **metrics_dict['classification_metrics']
        )

        # Skip batches to resume at correct position in dataloader
        # Since random seed is fixed, dataloader will have same shuffle order
        batches_to_skip = global_step_one_indexed
        if rank == 0:
            LOGGER.info(f"Skipping {batches_to_skip} batches to resume at correct position...")
        for _ in range(batches_to_skip):
            next(indist_iter_ebm)
            next(indist_iter_clf)
            next(outdist_iter)

        if rank == 0:
            LOGGER.info(
                f"Resumed from step {global_step_one_indexed}, "
                f"outdist_steps {start_outdist_steps}"
            )

    max_epochs_reached = False
    for cur_outdist_steps in range(
        start_outdist_steps, cfg.attack.max_steps + 1
    ):
        # Initialize AUC deques
        train_adv_auc_deque = collections.deque(
            maxlen=math.ceil(cfg.min_imgs_per_threshold / cfg.batch_size)
        )
        train_clean_auc_deque = collections.deque(
            maxlen=math.ceil(cfg.min_imgs_per_threshold / cfg.batch_size)
        )
        for local_step, train_indist_batch in enumerate(indist_iter_ebm):
            indist_epoch = global_step_one_indexed // len(indist_loader_ebm)
            global_step_one_indexed += 1
            n_imgs_seen = global_step_one_indexed * cfg.batch_size

            if (
                cfg.total_epochs is not None
                and indist_epoch >= cfg.total_epochs
            ):
                if rank == 0:
                    LOGGER.info(
                        f"Reached maximum number of epochs ({cfg.total_epochs}). Stopping training."
                    )
                max_epochs_reached = True
                break

            if rank == 0:
                wandb.log(
                    {
                        "cur_outdist_steps": cur_outdist_steps,
                        "indist_epoch": indist_epoch,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    },
                    step=n_imgs_seen,
                )
            is_metric_logging_step = should_trigger_event(
                global_step_one_indexed=global_step_one_indexed,
                batch_size=cfg.batch_size,
                interval_in_imgs=cfg.n_imgs_per_metrics_log,
            )
            is_evaluation_step = (
                cfg.n_imgs_per_evaluation_log is not None
                and should_trigger_event(
                    global_step_one_indexed=global_step_one_indexed,
                    batch_size=cfg.batch_size,
                    interval_in_imgs=cfg.n_imgs_per_evaluation_log,
                )
            )

            if is_evaluation_step and rank == 0:
                model.zero_grad()
                # In DDP mode, wrap model with DataParallel using all available GPUs
                # This gives ~8x speedup for FID evaluation vs single GPU
                if cfg.use_ddp and torch.cuda.device_count() > 1:
                    base_model = ema_model if cfg.use_ema else model.module
                    model_to_eval = nn.DataParallel(
                        base_model,
                        device_ids=list(range(torch.cuda.device_count()))
                    )
                else:
                    model_to_eval = nn.DataParallel(ema_model) if cfg.use_ema else model
                evaluate_and_log_fid(
                    model_to_eval,
                    cfg,
                    image_generation_metrics,
                    global_step_one_indexed,
                )

                # Generate and log sample images
                gen_imgs = log_generate_images(
                    cfg=cfg,
                    model=model_to_eval,
                    samples=10,
                )
                log_image_grid("train_gen_imgs", gen_imgs, cfg, n_imgs_seen)

                # Synchronize all ranks after FID evaluation and image logging
                if cfg.use_ddp:
                    dist.barrier()

            if is_evaluation_step and cfg.data.num_classes > 1 and rank == 0:
                model.eval()
                model.zero_grad()
                # In DDP mode, wrap model with DataParallel using all available GPUs
                # This gives ~8x speedup for accuracy evaluation vs single GPU
                if cfg.use_ddp and torch.cuda.device_count() > 1:
                    base_model = ema_model if cfg.use_ema else model.module
                    model_to_eval = nn.DataParallel(
                        base_model,
                        device_ids=list(range(torch.cuda.device_count()))
                    )
                else:
                    model_to_eval = nn.DataParallel(ema_model) if cfg.use_ema else model
                evaluate_and_log_accuracy(
                    model_to_eval,
                    cfg,
                    test_loader,
                    classification_metrics,
                    global_step_one_indexed,
                )
                # Synchronize all ranks after accuracy evaluation
                if cfg.use_ddp:
                    dist.barrier()

            # Save training state after evaluation
            if is_evaluation_step and rank == 0:
                rebm.training.modeling.save_training_state(
                    model=model,
                    optimizer=optimizer,
                    global_step=global_step_one_indexed,
                    cur_outdist_steps=cur_outdist_steps,
                    image_generation_metrics=image_generation_metrics,
                    classification_metrics=classification_metrics,
                    cfg=cfg,
                    ema_model=ema_model if cfg.use_ema else None,
                    wandb_run_id=wandb.run.id if wandb.run else None,
                )

            outdist_imgs = next(outdist_iter)[0].to(device)
            indist_imgs_ebm = train_indist_batch[0].to(device)
            indist_labels_ebm = train_indist_batch[1]

            indist_imgs_clf, indist_labels_clf = next(indist_iter_clf)
            indist_imgs_clf = indist_imgs_clf.to(device)
            indist_labels_clf = indist_labels_clf.to(device)

            optimizer.zero_grad()

            ebm_metrics = compute_ebm_metrics(
                indist_imgs=indist_imgs_ebm,
                indist_labels=indist_labels_ebm,
                outdist_imgs=outdist_imgs,
                outdist_step=cur_outdist_steps,
                model=model,
                cfg=cfg,
                criterion=criterion_ebm,
            )

            clf_loss = compute_clf_adv_loss(
                indist_imgs=indist_imgs_clf,
                indist_labels=indist_labels_clf,
                model=model,
                cfg=cfg,
                criterion=criterion_clf,
            )

            # Backpropagate combined loss and update weights
            total_loss = ebm_metrics.loss + clf_loss
            total_loss.backward()
            optimizer.step()

            # Aggregate AUC metrics across all ranks
            if cfg.use_ddp:
                adv_auc = average_metric_across_ranks(ebm_metrics.adv_auc, device, world_size)
                clean_auc = average_metric_across_ranks(ebm_metrics.clean_auc, device, world_size)
            else:
                adv_auc = ebm_metrics.adv_auc
                clean_auc = ebm_metrics.clean_auc

            train_adv_auc_deque.append(adv_auc)
            train_clean_auc_deque.append(clean_auc)

            if cfg.use_ema:
                with torch.no_grad():
                    ema_model.update_parameters(model.module)

            if global_step_one_indexed % 20 == 0 and rank == 0:
                ebm_metrics_dict = ebm_metrics.to_simple_dict()
                metrics_str = ", ".join(
                    [
                        f"{k}: {float(v):.5f}"
                        for k, v in ebm_metrics_dict.items()
                    ]
                )
                LOGGER.info(
                    f"Step {global_step_one_indexed:04d} - "
                    f"cur_outdist_steps: {cur_outdist_steps}, "
                    f"train_adv_auc_mean: {np.mean(train_adv_auc_deque):.2f}, "
                    f"train_clean_auc_mean: {np.mean(train_clean_auc_deque):.2f}, "
                    f"{metrics_str}"
                )

            if is_metric_logging_step and rank == 0:
                wandb.log(
                    dict_append_label(ebm_metrics.to_simple_dict(), "train_"),
                    step=n_imgs_seen,
                )

            is_image_logging_step = should_trigger_event(
                global_step_one_indexed=global_step_one_indexed,
                batch_size=cfg.batch_size,
                interval_in_imgs=cfg.n_imgs_per_image_log,
            )

            if is_image_logging_step and rank == 0:
                # Log training images
                for label, imgs in [
                    ("train_indist_imgs_clf", indist_imgs_clf),
                    ("train_indist_imgs", ebm_metrics.indist_imgs),
                    ("train_outdist_imgs", ebm_metrics.outdist_imgs_clean),
                    ("train_error_imgs", ebm_metrics.outdist_imgs_error),
                    ("train_adv_imgs", ebm_metrics.adv_imgs),
                ]:
                    log_image_grid(label, imgs, cfg, n_imgs_seen)

            if (
                cur_outdist_steps < cfg.attack.max_steps
                and cfg.samples_per_attack_step is not None
                and (local_step + 1) * cfg.batch_size
                >= cfg.samples_per_attack_step
            ):
                if rank == 0:
                    LOGGER.info(
                        f"Outdist step {cur_outdist_steps} reached max samples {cfg.samples_per_attack_step}"
                    )
                break

            if (
                cur_outdist_steps < cfg.attack.max_steps
                and np.mean(train_adv_auc_deque) >= cfg.AUC_th
                and len(train_adv_auc_deque) == train_adv_auc_deque.maxlen
            ):
                if rank == 0:
                    LOGGER.info(
                        f"Adv AUC reached threshold {cfg.AUC_th} on local step {local_step} outdist step {cur_outdist_steps}"
                    )
                break

        if max_epochs_reached:
            break


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args or "-h" in args or "--help" in args:
        print("Training script with OmegaConf for configuration management")
        print(
            "\nUsage: python -m rebm.training.train CONFIG_FILE [KEY=VALUE ...]"
        )
        print("\nExamples:")
        print("  # Single GPU (DataParallel)")
        print("  python -m rebm.training.train experiments/cifar10/config.yaml use_ddp=False")
        print(
            "  python -m rebm.training.train experiments/cifar10/config.yaml batch_size=256 lr=0.001"
        )
        print("\n  # Multi-GPU (DDP with torchrun)")
        print("  torchrun --nproc_per_node=4 -m rebm.training.train experiments/cifar10/config.yaml")
        print(
            "  torchrun --nproc_per_node=2 -m rebm.training.train experiments/cifar10/config.yaml batch_size=64"
        )
        print(
            "\n  # Other examples"
        )
        print(
            "  python -m rebm.training.train experiments/cifar10/config.yaml model.ckpt_path=/path/to/model.pth"
        )
        print(
            "  python -m rebm.training.train experiments/cifar10/config.yaml image_log.num_steps=50 attack.max_steps=100"
        )
        sys.exit(0)

    config_file = args[0]
    overrides = args[1:]

    cfg = load_train_config(config_file, overrides)

    # Don't upload .pth files to wandb, since they are big
    # We just save them on disk for now.
    os.environ["WANDB_IGNORE_GLOBS"] = "*.pth"

    # Initialize wandb only on rank 0 to avoid multiple runs
    # When using DDP, we need to initialize the process group first
    if cfg.use_ddp:
        # In DDP mode with torchrun, only init wandb on rank 0
        # We need to initialize the process group first to get the rank
        if not dist.is_initialized():
            # Set a longer timeout for operations like FID evaluation and image logging
            # which can take 60+ minutes with step sweeping
            timeout = timedelta(minutes=90)
            dist.init_process_group(backend=cfg.ddp_backend, timeout=timeout)
        rank = dist.get_rank()
    else:
        # Single GPU or DataParallel mode
        rank = 0

    if rank == 0:
        run_name = os.environ.get("WANDB_NAME", Path(config_file).stem)

        # Load wandb run ID from saved state if resuming
        resume_wandb_id = None
        if cfg.resume_from_state:
            state = torch.load(cfg.resume_from_state, map_location='cpu', weights_only=False)
            resume_wandb_id = state['wandb_run_id']
            if resume_wandb_id:
                LOGGER.info(f"Resuming wandb run: {resume_wandb_id}")

        # Initialize wandb
        wandb_init_kwargs = {
            'project': cfg.wandb_project,
            'tags': cfg.tags,
            'dir': cfg.wandb_dir,
            'save_code': True,
            'mode': "disabled" if cfg.wandb_disabled else "online",
            'config': dataclasses.asdict(cfg),
        }

        # Resume existing run if available
        if resume_wandb_id:
            wandb_init_kwargs['id'] = resume_wandb_id
            wandb_init_kwargs['resume'] = 'must'
            # Don't set name when resuming - wandb keeps the original run name
        else:
            wandb_init_kwargs['name'] = run_name

        wandb.init(**wandb_init_kwargs)

    torch.backends.cudnn.benchmark = True

    train(cfg)
