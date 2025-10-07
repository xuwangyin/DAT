import collections
import dataclasses
import logging
import math
import os
import sys
from pathlib import Path
from typing import Iterable

os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.insert(0, "pytorch-image-models")

import numpy as np
import torch
import torch.utils.data
import torchvision.utils
import wandb
from torch import nn

import rebm.training.data
import rebm.training.misc
import rebm.training.modeling
from rebm.training.average_model import AveragedModel
from rebm.training.config_classes import TrainConfig, load_train_config
from rebm.training.eval_utils import eval_acc, eval_robust_acc, evaluate_image_generation
from rebm.training.metrics import (
    ClassificationMetrics,
    ImageGenerationMetrics,
    TrainingMetrics,
    compute_training_metrics,
    compute_training_metrics_clf,
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
    train_indist_loader: torch.utils.data.DataLoader
    train_indist_loader_clf: torch.utils.data.DataLoader
    train_indist_iter: Iterable
    train_indist_iter_clf: Iterable
    train_outdist_iter: Iterable
    test_loader_for_eval: torch.utils.data.DataLoader


def infinite_iter(iterable: Iterable):
    while True:
        for x in iterable:
            yield x


def dict_append_label(d: dict, label: str) -> dict:
    return {label + k: v for k, v in d.items()}


def create_dataloaders(cfg: TrainConfig) -> Dataloaders:
    """Create all dataloaders needed for training.

    Returns:
        Dataloaders instance containing:
            - train_indist_loader: Loader for in-distribution training (generation augmentation)
            - train_indist_loader_clf: Loader for in-distribution training (classification augmentation)
            - train_indist_iter: Infinite iterator for train_indist_loader
            - train_indist_iter_clf: Infinite iterator for train_indist_loader_clf
            - train_outdist_iter: Infinite iterator for out-distribution data
            - test_loader_for_eval: Loader for test set evaluation
    """
    train_indist_loader = rebm.training.data.get_indist_dataloader(
        config=cfg.data,
        batch_size=cfg.batch_size,
        shuffle=True,
        augm_type=cfg.augm_type_generation
    )
    train_indist_loader_clf = rebm.training.data.get_indist_dataloader(
        config=cfg.data,
        batch_size=cfg.batch_size,
        shuffle=True,
        augm_type=cfg.augm_type_classification
    )
    train_outdist_iter = infinite_iter(rebm.training.data.get_outdist_dataloader(
        config=cfg.data,
        batch_size=cfg.batch_size,
        shuffle=True,
        augm_type_generation=cfg.augm_type_generation,
        tinyimages_loader=cfg.tinyimages_loader,
        openimages_max_samples=cfg.openimages_max_samples,
        openimages_augm=cfg.openimages_augm,
    ))
    test_loader_for_eval = rebm.training.data.get_indist_dataloader(
        config=cfg.data,
        batch_size=cfg.batch_size,
        split="val",
        shuffle=False,
        augm_type="none" if 'cifar' in cfg.data.indist_dataset else "test"
    )

    return Dataloaders(
        train_indist_loader=train_indist_loader,
        train_indist_loader_clf=train_indist_loader_clf,
        train_indist_iter=infinite_iter(train_indist_loader),
        train_indist_iter_clf=infinite_iter(train_indist_loader_clf),
        train_outdist_iter=train_outdist_iter,
        test_loader_for_eval=test_loader_for_eval,
    )


def log_image_grid(label: str, imgs: torch.Tensor, cfg: TrainConfig, step: int) -> None:
    """Log image grid to wandb.

    Args:
        label: Label for the wandb log
        imgs: Image tensor to log (will use first 10 images)
        cfg: Training configuration
        step: Global step for logging
    """
    padding = 0 if "cifar10" in cfg.data.indist_dataset else 2
    image_grid = torchvision.utils.make_grid(imgs[:10], nrow=10, padding=padding)
    wandb.log({label: wandb.Image(image_grid)}, step=step)


def evaluate_and_log_fid(
    model_to_eval: nn.Module,
    cfg: TrainConfig,
    image_generation_metrics: ImageGenerationMetrics,
    global_step: int,
) -> None:
    """Evaluate FID and save best model if improved.

    Args:
        model_to_eval: Model to evaluate (EMA or training model)
        cfg: Training configuration
        image_generation_metrics: Metrics tracker for image generation
        global_step: Current global step (1-indexed)
    """
    model_to_eval.eval()
    n_imgs_seen = global_step * cfg.batch_size

    fid, gen_imgs = evaluate_image_generation(model_to_eval, cfg)
    LOGGER.info(
        f"FID: {fid}, step: {global_step}, n_imgs: {n_imgs_seen}"
    )

    is_new_best = image_generation_metrics.update(fid, gen_imgs)
    if is_new_best:
        model_to_save = model_to_eval.module.module if cfg.use_ema else model_to_eval.module
        LOGGER.info(f'Saving {"EMA" if cfg.use_ema else "regular"} model with best FID')
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

    Args:
        model_to_eval: Model to evaluate (EMA or training model)
        cfg: Training configuration
        test_loader_for_eval: Test dataloader
        classification_metrics: Metrics tracker for classification
        global_step: Current global step (1-indexed)
    """
    model_to_eval.eval()
    n_imgs_seen = global_step * cfg.batch_size

    metrics_dict = {
        "train_acc": None,
        "test_acc": None,
        "train_acc_calib": None,
        "test_acc_calib": None,
        "robust_train_acc": None,
        "robust_test_acc": None,
    }

    LOGGER.info("Evaluating standard accuracy...")
    metrics_dict["test_acc"] = eval_acc(
        model=model_to_eval,
        dataloader=test_loader_for_eval,
        device=cfg.device,
    )

    # Free CUDA memory after accuracy evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    LOGGER.info(
        f"Standard Acc - Train: {metrics_dict['train_acc']}, Test: {metrics_dict['test_acc']:.4f}"
    )

    if cfg.robust_eval:
        LOGGER.info("Evaluating robust accuracy...")
        attack_kwargs = {
            "norm": "L2",
            "eps": cfg.indist_attack_clf.eps,
            "step_size": cfg.indist_attack_clf.step_size,
            "steps": cfg.indist_attack_clf.max_steps,
            "random_start": False,
        }

        metrics_dict["robust_test_acc"] = eval_robust_acc(
            model=model_to_eval,
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

    is_new_best_acc = classification_metrics.update(**metrics_dict)
    if is_new_best_acc:
        model_to_save = model_to_eval.module.module if cfg.use_ema else model_to_eval.module
        rebm.training.modeling.save_best_accuracy_model(model_to_save)

    wandb.log(
        dataclasses.asdict(classification_metrics),
        step=n_imgs_seen,
    )


def train(cfg: TrainConfig):
    np.random.seed(cfg.rand_seed)
    torch.manual_seed(cfg.rand_seed)
    global_step_one_indexed: int = 0

    image_generation_metrics = ImageGenerationMetrics()
    classification_metrics = ClassificationMetrics()

    # Create all dataloaders
    dataloaders = create_dataloaders(cfg)
    train_indist_loader = dataloaders.train_indist_loader
    train_indist_loader_clf = dataloaders.train_indist_loader_clf
    train_indist_iter = dataloaders.train_indist_iter
    train_indist_iter_clf = dataloaders.train_indist_iter_clf
    train_outdist_iter = dataloaders.train_outdist_iter
    test_loader_for_eval = dataloaders.test_loader_for_eval

    model = rebm.training.modeling.get_model(
        model_config=cfg.model,
        device=cfg.device,
        num_classes=cfg.data.num_classes,
        indist_dataset=cfg.data.indist_dataset,
    )
    if cfg.use_ema:
        ema_model = AveragedModel(
            model.module,
            avg_type="ema",
            ema_decay=0.999,
            avg_batchnorm=True,
            device=cfg.device,
        )

    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    criterion_clf = nn.CrossEntropyLoss(reduction="mean")
    optimizer = rebm.training.modeling.get_optimizer(
        model=model,
        optimizer_name=cfg.optimizer,
        lr=cfg.lr,
        wd=cfg.wd,
    )

    get_metrics_shared_kwargs = dict(
        model=model,
        cfg=cfg,
        criterion=criterion,
    )

    get_metrics_shared_kwargs_clf = dict(
        model=model,
        cfg=cfg,
        criterion=criterion_clf,
    )

    LOGGER.info(
        f"indist dataset classes: {train_indist_loader.dataset.classes}"
    )

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
            indist_epoch = global_step_one_indexed // len(train_indist_loader)
            global_step_one_indexed += 1
            n_imgs_seen = global_step_one_indexed * cfg.batch_size

            if cfg.total_epochs is not None and indist_epoch >= cfg.total_epochs:
                LOGGER.info(
                    f"Reached maximum number of epochs ({cfg.total_epochs}). Stopping training."
                )
                max_epochs_reached = True
                break

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
            is_image_logging_step = should_trigger_event(
                global_step_one_indexed=global_step_one_indexed,
                batch_size=cfg.batch_size,
                interval_in_imgs=cfg.n_imgs_per_image_log,
            )
            is_evaluation_step = (
                cfg.n_imgs_per_evaluation_log is not None
                and should_trigger_event(
                    global_step_one_indexed=global_step_one_indexed,
                    batch_size=cfg.batch_size,
                    interval_in_imgs=cfg.n_imgs_per_evaluation_log,
                )
            )
            is_checkpoint_save_step = should_trigger_event(
                global_step_one_indexed=global_step_one_indexed,
                batch_size=cfg.batch_size,
                interval_in_imgs=cfg.n_imgs_per_ckpt_save,
                at_end=True,
            )

            if is_checkpoint_save_step:
                model_to_save = ema_model.module if cfg.use_ema else model.module
                rebm.training.modeling.save_checkpoint(
                    model=model_to_save,
                    optimizer=optimizer,
                    step=global_step_one_indexed,
                )

            if is_evaluation_step:
                model.zero_grad()
                model_to_eval = nn.DataParallel(ema_model) if cfg.use_ema else model
                evaluate_and_log_fid(
                    model_to_eval,
                    cfg,
                    image_generation_metrics,
                    global_step_one_indexed,
                )

            if is_evaluation_step and cfg.data.num_classes > 1:
                model.eval()
                model.zero_grad()
                model_to_eval = nn.DataParallel(ema_model) if cfg.use_ema else model
                evaluate_and_log_accuracy(
                    model_to_eval,
                    cfg,
                    test_loader_for_eval,
                    classification_metrics,
                    global_step_one_indexed,
                )

            train_outdist_imgs = next(train_outdist_iter)[0].to(cfg.device)
            train_indist_imgs = train_indist_batch[0].to(cfg.device)
            train_indist_labels = train_indist_batch[1]

            optimizer.zero_grad()
            train_metrics = compute_training_metrics(
                indist_imgs=train_indist_imgs,
                indist_labels=train_indist_labels,
                outdist_imgs=train_outdist_imgs,
                outdist_step=cur_outdist_steps,
                **get_metrics_shared_kwargs,
            )
            train_indist_imgs_clf, train_indist_labels_clf = next(
                train_indist_iter_clf
            )
            train_indist_imgs_clf = train_indist_imgs_clf.to(cfg.device)
            train_indist_labels_clf = train_indist_labels_clf.to(
                cfg.device
            )

            clf_loss = compute_training_metrics_clf(
                indist_imgs=train_indist_imgs_clf,
                indist_labels=train_indist_labels_clf,
                **get_metrics_shared_kwargs_clf,
            )
            train_adv_auc_deque.append(train_metrics.adv_auc)
            train_clean_auc_deque.append(train_metrics.clean_auc)
            (train_metrics.loss * cfg.bce_weight + clf_loss * cfg.clf_lr_multiplier).backward()
            optimizer.step()

            if cfg.use_ema:
                with torch.no_grad():
                    ema_model.update_parameters(model.module)


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

            if is_metric_logging_step:
                wandb.log(
                    dict_append_label(train_metrics.to_simple_dict(), "train_"),
                    step=n_imgs_seen,
                )

            if is_image_logging_step:
                for label, imgs in [
                    ("train_indist_imgs_clf", train_indist_imgs_clf),
                    ("train_indist_imgs", train_metrics.indist_imgs),
                    ("train_outdist_imgs", train_metrics.outdist_imgs_clean),
                    ("train_error_imgs", train_metrics.outdist_imgs_error),
                    ("train_adv_imgs", train_metrics.adv_imgs),
                    ("train_gen_imgs", image_generation_metrics.gen_imgs),
                ]:
                    log_image_grid(label, imgs, cfg, n_imgs_seen)

            if (
                cur_outdist_steps < cfg.attack.max_steps
                and cfg.samples_per_attack_step is not None
                and (local_step + 1) * cfg.batch_size
                >= cfg.samples_per_attack_step
            ):
                LOGGER.info(
                    f"Outdist step {cur_outdist_steps} reached max samples {cfg.samples_per_attack_step}"
                )
                break

            if (
                cur_outdist_steps < cfg.attack.max_steps
                and np.mean(train_adv_auc_deque) >= cfg.AUC_th
                and len(train_adv_auc_deque) == train_adv_auc_deque.maxlen
            ):
                LOGGER.info(
                    f"Adv AUC reached threshold {cfg.AUC_th} on local step {local_step} outdist step {cur_outdist_steps}"
                )
                break

        if max_epochs_reached:
            break


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]

    if not args or "-h" in args or "--help" in args:
        print("Training script with OmegaConf for configuration management")
        print("\nUsage: python -m rebm.training.train CONFIG_FILE [KEY=VALUE ...]")
        print("\nExamples:")
        print("  python -m rebm.training.train experiments/cifar10/config.yaml")
        print(
            "  python -m rebm.training.train experiments/cifar10/config.yaml batch_size=256 lr=0.001"
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

    run_name = os.environ.get("WANDB_NAME", Path(config_file).stem)
    wandb.init(
        project=cfg.wandb_project,
        tags=cfg.tags,
        dir=cfg.wandb_dir,
        save_code=True,
        mode="disabled" if cfg.wandb_disabled else "online",
        name=run_name,
        config=dataclasses.asdict(cfg),
    )
    LOGGER.info(f"Using device: {cfg.device}")

    torch.backends.cudnn.benchmark = True

    train(cfg)
