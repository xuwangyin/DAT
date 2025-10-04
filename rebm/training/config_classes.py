from dataclasses import dataclass
from typing import List, Literal, Optional


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
    step_sweep_range: Optional[List[int]] = None  # Steps to sweep (e.g. [1, 5, 10, 20, 30])
    sweep_num_samples: int = 500  # Smaller sample count for step sweep
    
    # Dependent generation params
    lr: Optional[float] = None  # only for Adam
    step_size: Optional[float] = None  # only for PGD
    eps: Optional[float | int] = None  # only for PGD


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
