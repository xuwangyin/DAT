"""Utilities and CLI for FID evaluation."""

from __future__ import annotations

import dataclasses
import logging
import sys

from rebm.training.config_classes import TrainConfig, load_train_config
from rebm.eval.eval_utils import evaluate_image_generation
from rebm.training.modeling import get_model

LOGGER = logging.getLogger(__name__)


def run_fid_evaluation(cfg: TrainConfig) -> float:
    """Run FID evaluation and return FID score.

    Args:
        cfg: Training configuration with model and image_log settings

    Returns:
        FID score
    """
    LOGGER.info("FID evaluation image_log config:")
    for field in dataclasses.fields(cfg.image_log):
        value = getattr(cfg.image_log, field.name)
        LOGGER.info(f"  {field.name}: {value}")
    LOGGER.info(f"Batch size: {cfg.batch_size}")
    LOGGER.info(f"Logsumexp sampling (unconditional): {cfg.logsumexp_sampling}")
    LOGGER.info("Initializing model...")
    model = get_model(
        model_config=cfg.model,
        device=cfg.device,
        num_classes=cfg.data.num_classes,
        indist_dataset=cfg.data.indist_dataset,
    )
    model.eval()

    LOGGER.info("Starting FID evaluation...")
    fid, gen_imgs = evaluate_image_generation(model, cfg)

    LOGGER.info(f"FID evaluation completed. FID: {fid:.4f}")
    return fid


def main(argv: list[str] | None = None) -> float:
    """CLI entry point for FID evaluation.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:])

    Returns:
        FID score
    """
    args = sys.argv[1:] if argv is None else argv

    if not args or "-h" in args or "--help" in args:
        print("FID evaluation for trained models")
        print("\nUsage: python -m rebm.eval.eval_fid CONFIG_FILE [KEY=VALUE ...]")
        print("\nExamples:")
        print("  python -m rebm.eval.eval_fid experiments/cifar10/eval_fid/config.yaml")
        print("  python -m rebm.eval.eval_fid experiments/cifar10/config.yaml model.ckpt_path=/path/to/model.pth")
        sys.exit(0)

    config_file = args[0]
    overrides = args[1:]

    cfg = load_train_config(config_file, overrides)
    fid = run_fid_evaluation(cfg)
    print(f"FID: {fid:.4f}")
    return fid


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
