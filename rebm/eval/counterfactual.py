"""Utilities and CLI for counterfactual image generation."""

from __future__ import annotations

import logging
import sys
from typing import Iterable

from rebm.training import data as training_data
from rebm.training.config_classes import (
    TrainConfig,
    load_train_config,
)
from rebm.eval.eval_utils import generate_counterfactuals
from rebm.training.modeling import get_model

LOGGER = logging.getLogger(__name__)


def run_counterfactual_generation(cfg: TrainConfig) -> None:
    """Generate counterfactual images using the given configuration."""
    LOGGER.info("Counterfactual generation requested. Initializing model...")
    model = get_model(
        model_config=cfg.model,
        device=cfg.device,
        num_classes=cfg.data.num_classes,
        indist_dataset=cfg.data.indist_dataset,
    ).to(cfg.device)
    model.eval()

    loader = training_data.get_indist_dataloader(
        config=cfg.data,
        batch_size=cfg.batch_size,
        split="train",
        shuffle=False,
        augm_type="none",
    )

    LOGGER.info("Starting counterfactual generation process...")
    generate_counterfactuals(model, loader, cfg)
    LOGGER.info("Counterfactual generation completed.")


def main(argv: Iterable[str] | None = None) -> None:
    args = list(argv) if argv is not None else sys.argv[1:]

    if not args or args[0] in {"-h", "--help"}:
        print("Counterfactual generation")
        print("\nUsage: python -m rebm.eval.counterfactual CONFIG_FILE [KEY=VALUE ...]")
        sys.exit(0)

    config_file = args[0]
    overrides = args[1:]

    cfg = load_train_config(config_file, overrides)
    run_counterfactual_generation(cfg)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
