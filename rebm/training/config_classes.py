import copy
import os
import uuid
from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Tuple

import torch
from omegaconf import OmegaConf


@dataclass
# Out-dist attack as part of training objective
class AttackConfig:
    attack_type: Literal["adam", "pgd"]
    max_steps: int
    fixed_steps: int
    start_step: int  # Set for resumption purposes

    # Adam-specific attack parameters
    lr: Optional[float] = None  # ~ equivalent to PGD attack step size

    # PGD-specific attack parameters
    step_size: Optional[float] = None
    eps: Optional[float] = None


@dataclass
class ImageLogConfig:
    # Generated images
    attack_type: Literal["adam", "pgd"]
    num_steps: int  # for PGD and Adam

    # FID (computed on generated images)
    log_fid: bool
    data_dir: Optional[str]
    ood_data_dir: Optional[str]
    img_extension: Optional[str]
    num_samples: Optional[int]
    save_dir: Optional[str] = None
    target_class: Optional[int] = None

    # Adaptive FID step selection
    adaptive_steps: bool = False  # Enable adaptive step selection
    step_sweep_range: Optional[List[int]] = (
        None  # Steps to sweep (e.g. [1, 5, 10, 20, 30])
    )
    sweep_num_samples: int = 500  # Smaller sample count for step sweep

    # Dependent generation params
    lr: Optional[float] = None  # only for Adam
    step_size: Optional[float] = None  # only for PGD
    eps: Optional[float | int] = None  # only for PGD

    def __post_init__(self):
        if self.save_dir is None:
            unique_suffix = uuid.uuid4().hex[:8]
            self.save_dir = os.path.join("image_log", unique_suffix)


@dataclass
class DataConfig:
    # In-distribution dataset config
    indist_dataset: str
    indist_ds_dir: str
    mixed_id: bool  # Mixed in-distribution

    # Out-distribution dataset config
    outdist_dataset: str
    outdist_std_dir: str

    num_workers: int

    num_classes: int = 1


@dataclass
class BaseModelConfig:
    """Base configuration shared by all models"""

    model_type: str
    ckpt_path: Optional[str] = None
    which_logit: Literal["first", "all", "bird"] = "first"
    normalize_input: bool = True
    use_batchnorm: bool = False
    use_layernorm: bool = True


@dataclass
class ConvNextConfig(BaseModelConfig):
    """Configuration specific to ConvNext2 B"""

    model_prefix: str = ""
    drop_path_rate: float = 0.1
    head_init_scale: float = 1.0
    model_id: str = "convnextv2_base"
    use_convstem: bool = True


def create_model_config(config_dict: dict) -> BaseModelConfig:
    """Factory function to create the appropriate model config"""
    model_type = config_dict.get("model_type", "").lower()

    if model_type.startswith("convnext"):
        return ConvNextConfig(**config_dict)
    else:
        # Return BaseModelConfig for all other cases (ResNet, WideResNet, etc.)
        return BaseModelConfig(**config_dict)


@dataclass
class TrainConfig:
    """Top-level training configuration."""

    # Required parameters
    data: DataConfig
    attack: AttackConfig
    model: BaseModelConfig
    image_log: ImageLogConfig

    # Optimization parameters
    optimizer: str
    wd: float
    lr: float
    r1reg: float
    clf_indist_weight: float
    clf_outdist_weight: float
    clf_adv_weight: float
    r1_indist_weight: float
    r1_outdist_weight: float
    r1_adv_weight: float

    # Training scheduling
    batch_size: int
    min_imgs_per_threshold: int
    AUC_th: float
    rand_seed: int

    # Logging
    n_imgs_per_metrics_log: int
    n_imgs_per_image_log: int
    n_imgs_per_ckpt_save: int

    # WandB
    wandb_project: str
    wandb_dir: str
    wandb_disabled: bool
    tags: Tuple[str, ...]

    # Optional parameters
    indist_attack_clf: AttackConfig
    indist_clean_extra: bool = False
    fp16: bool = False
    samples_per_attack_step: int | None = None
    n_imgs_per_evaluation_log: int | None = (
        None  # Controls both FID and accuracy evaluation
    )
    use_ema: bool = False

    # Evaluation parameters
    robust_eval: bool = True
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
    eval_only: bool = False
    use_counterfactuals: bool = False
    evaluate_ood_detection: bool = False
    ood_detection_logsumexp: bool = False
    outdist_dataset_ood_detection: str = "noise"
    openimages_max_samples: int | None = None
    openimages_augm: str | None = None

    total_epochs: int | None = None

    @property
    def dtype(self) -> torch.dtype:
        return torch.float16 if self.fp16 else torch.float32

    def __post_init__(self):
        if self.wandb_dir is None:
            self.wandb_dir = "./"
        if not self.fixed_lr:
            raise ValueError(
                "fixed_lr must be True. Dynamic learning rate schedules are not supported."
            )

    @property
    def device(self) -> torch.device:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_train_config(
    config_file: str, overrides: Iterable[str] | None = None
) -> TrainConfig:
    """Load TrainConfig from an OmegaConf file with optional overrides."""

    omega_cfg = OmegaConf.load(config_file)
    if overrides:
        override_cfg = OmegaConf.from_dotlist(list(overrides))
        omega_cfg = OmegaConf.merge(omega_cfg, override_cfg)

    config_dict = copy.deepcopy(OmegaConf.to_container(omega_cfg, resolve=True))

    config_dict["data"] = DataConfig(**config_dict.get("data", {}))
    config_dict["attack"] = AttackConfig(**config_dict.get("attack", {}))
    config_dict["model"] = create_model_config(config_dict.get("model", {}))
    config_dict["image_log"] = ImageLogConfig(
        **config_dict.get("image_log", {})
    )
    config_dict["indist_attack_clf"] = AttackConfig(
        **config_dict["indist_attack_clf"]
    )

    return TrainConfig(**config_dict)
