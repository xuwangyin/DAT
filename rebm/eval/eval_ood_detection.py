"""Utilities and CLI for out-of-distribution detection evaluation."""

from __future__ import annotations

import sys
import logging
from typing import Tuple

import numpy as np
import torch
import InNOutRobustness.utils.datasets as dl
from rebm.training import data as training_data
from rebm.training.config_classes import (
    TrainConfig,
    load_train_config,
)
from rebm.eval.eval_utils import ood_detection
from rebm.training.modeling import get_model

LOGGER = logging.getLogger(__name__)


def run_ood_evaluation(cfg: TrainConfig) -> Tuple[float, float]:
    """Run OOD detection evaluation and return clean/adv AUROC."""
    # Set random seed for reproducibility
    np.random.seed(cfg.rand_seed)
    torch.manual_seed(cfg.rand_seed)

    LOGGER.info("OOD detection evaluation requested. Initializing model...")
    model = get_model(
        model_config=cfg.model,
        device=cfg.device,
        num_classes=cfg.data.num_classes,
        indist_dataset=cfg.data.indist_dataset,
    ).to(cfg.device)
    model.eval()

    indist_loader = training_data.get_indist_dataloader(
        config=cfg.data,
        batch_size=cfg.batch_size,
        split="val",
        shuffle=False,
        augm_type="none",
        balanced=True,
    )

    LOGGER.info(
        "Using %s as OOD dataset for detection",
        cfg.outdist_dataset_ood_detection,
    )

    if cfg.data.indist_dataset in ["cifar10-conditional", "cifar100-conditional"]:
        size = 32
    elif cfg.data.indist_dataset in ["RestrictedImageNet", "ImageNet"]:
        size = 224
    else:
        size = 32

    LOGGER.info(
        "Using image size %d for OOD detection based on in-distribution dataset: %s",
        size,
        cfg.data.indist_dataset,
    )

    outdist_loader = _get_outdist_loader(cfg, size)

    LOGGER.info("Starting OOD detection evaluation...")
    clean_auroc, adv_auroc = ood_detection(model, indist_loader, outdist_loader, cfg)
    LOGGER.info(
        "OOD detection evaluation completed. ID: %s, OD: %s, Clean AUROC: %.4f, Adversarial AUROC: %.4f",
        cfg.data.indist_dataset,
        cfg.outdist_dataset_ood_detection,
        clean_auroc,
        adv_auroc,
    )
    return clean_auroc, adv_auroc


def _get_outdist_loader(cfg: TrainConfig, size: int):
    match cfg.outdist_dataset_ood_detection:
        case "noise":
            return dl.get_noise_dataset(
                type="uniform",
                length=1024,
                size=size,
                augm_type="none",
                batch_size=cfg.batch_size,
            )
        case "svhn":
            return dl.get_SVHN(
                split="train",
                batch_size=cfg.batch_size,
                shuffle=True,
                augm_type="none",
                size=size,
            )
        case "cifar100":
            return dl.get_CIFAR100(
                train=True,
                batch_size=cfg.batch_size,
                shuffle=True,
                augm_type="none",
                size=size,
            )
        case "cifar10":
            return dl.get_CIFAR10(
                train=True,
                batch_size=cfg.batch_size,
                shuffle=True,
                augm_type="none",
                size=size,
            )
        case "imagenet":
            return dl.get_restrictedImageNetOD(
                train=False,
                batch_size=cfg.batch_size,
                shuffle=True,
                augm_type="none",
                size=size,
                path="./data/ImageNet",
            )
        case _:
            raise ValueError(
                f"Unknown outdist_dataset_ood_detection: {cfg.outdist_dataset_ood_detection}"
            )


def main(argv: list[str] | None = None) -> Tuple[float, float]:
    args = sys.argv[1:] if argv is None else argv

    if not args or args[0] in {"-h", "--help"}:
        print("OOD detection evaluation")
        print("\nUsage: python -m rebm.eval.eval_ood_detection CONFIG_FILE [KEY=VALUE ...]")
        sys.exit(0)

    config_file = args[0]
    overrides = args[1:]

    cfg = load_train_config(config_file, overrides)
    clean_auroc, adv_auroc = run_ood_evaluation(cfg)
    print(
        f"Clean AUROC: {clean_auroc:.4f}, Adversarial AUROC: {adv_auroc:.4f}"
    )
    return clean_auroc, adv_auroc


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
