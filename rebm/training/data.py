"""Data management for training - handles dataset and dataloader creation."""

import logging
from typing import Literal, Optional

import torch
import torch.distributed as dist
import torch.utils.data
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler

import InNOutRobustness.utils.datasets as dl
from InNOutRobustness.utils.datasets.augmentations.cifar_augmentation import (
    get_cifar10_augmentation,
)
from InNOutRobustness.utils.datasets.augmentations.imagenet_augmentation import (
    get_imageNet_augmentation,
)
from rebm.training.config_classes import DataConfig

LOGGER = logging.getLogger(__name__)


class CIFAR10Unconditional(torchvision.datasets.CIFAR10):
    """CIFAR10 dataset variant for unsupervised learning that ignores labels."""

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.num_classes = 1  # Override the default num_classes (10)
        self.classes = [0]

    def __getitem__(self, index):
        """Override __getitem__ to return zero as label."""
        image, _ = super().__getitem__(index)
        return image, 0


DEFAULT_CIFAR10_PARAMETERS = {
    "interpolation": "bilinear",
    "mean": (0, 0, 0),
    "crop_pct": 0.875,
}


def get_cifar10_dataset(
    data_dir,
    split: Literal["train", "test", "val"] = "train",
    conditional=True,
    augm_type: str = "autoaugment_cutout",
):
    # if split == "train" and augm_type == "none":
    #     raise ValueError("CIFAR10 training set must use augmentation.")
    if split != "train" and augm_type != "none":
        raise ValueError(
            "CIFAR10 test/validation set should not use augmentation."
        )
    if augm_type == "original":
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    else:
        augm_config = {}
        transform = get_cifar10_augmentation(
            augm_type,
            cutout_window=16,
            out_size=32,
            augm_parameters=DEFAULT_CIFAR10_PARAMETERS,
            config_dict=augm_config,
        )
    is_train_split = True if split == "train" else False
    kwargs = dict(
        root=data_dir,
        train=is_train_split,
        download=True,
        transform=transform,
    )
    indist_dataset = (
        torchvision.datasets.CIFAR10(**kwargs)
        if conditional
        else CIFAR10Unconditional(**kwargs)
    )
    return indist_dataset


def get_cifar100_dataset(
    data_dir,
    split: Literal["train", "test", "val"] = "train",
    conditional=True,
    augm_type: str = "autoaugment_cutout",
):
    assert conditional
    # if split == "train" and augm_type == "none":
    #     raise ValueError("CIFAR10 training set must use augmentation.")
    if split != "train" and augm_type != "none":
        raise ValueError(
            "CIFAR100 test/validation set should not use augmentation."
        )
    if augm_type == "original":
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    else:
        augm_config = {}
        transform = get_cifar10_augmentation(
            augm_type,
            cutout_window=16,
            out_size=32,
            augm_parameters=DEFAULT_CIFAR10_PARAMETERS,
            config_dict=augm_config,
        )
    is_train_split = True if split == "train" else False
    kwargs = dict(
        root=data_dir,
        train=is_train_split,
        download=True,
        transform=transform,
    )
    indist_dataset = torchvision.datasets.CIFAR100(**kwargs)
    return indist_dataset


def get_tinyimages_dataset(
    data_dir,
    augm_type: str = "autoaugment_cutout",
    tinyimages_loader: str = "innout",
):
    if augm_type == "original":
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    else:
        augm_config = {}
        transform = get_cifar10_augmentation(
            augm_type,
            cutout_window=16,
            out_size=32,
            augm_parameters=DEFAULT_CIFAR10_PARAMETERS,
            config_dict=augm_config,
        )

    if tinyimages_loader != "innout":
        raise ValueError(f"Unknown tinyimages_loader: {tinyimages_loader}")

    return dl.TinyImagesDataset(
        data_dir, transform, exclude_cifar=True, exclude_cifar10_1=True
    )


def get_indist_dataset(
    config: DataConfig,
    split: str = "train",
    attack: bool = False,
    augm_type: str = "autoaugment_cutout",
):
    """Get in-distribution dataset."""
    assert split in ["train", "val"], (
        f"split must be 'train' or 'val', got {split}"
    )

    match config.indist_dataset:
        case "cifar10-conditional":
            if split == "val":
                assert augm_type == "none", (
                    f"For CIFAR val split, augm_type must be 'none', got {augm_type}"
                )
            indist_dataset = get_cifar10_dataset(
                data_dir=config.indist_ds_dir,
                split=split,
                conditional=True,
                augm_type=augm_type,
            )
        case "cifar100-conditional":
            if split == "val":
                assert augm_type == "none", (
                    f"For CIFAR val split, augm_type must be 'none', got {augm_type}"
                )
            indist_dataset = get_cifar100_dataset(
                data_dir=config.indist_ds_dir,
                split=split,
                conditional=True,
                augm_type=augm_type,
            )
        case "ImageNet":
            LOGGER.info("Using ImageNet dataset")

            # Validate augmentation type for ImageNet
            if split == "train":
                assert augm_type in [
                    "madry",
                    "generation_id",
                    "generation_id_randomcrop",
                    "none",
                    "default",
                ]
            else:
                assert augm_type in ["none", "test"]

            transform = get_imageNet_augmentation(type=augm_type, out_size=224)
            dataset_split = "train" if split == "train" else "val"
            indist_dataset = torchvision.datasets.ImageNet(
                config.indist_ds_dir, split=dataset_split, transform=transform
            )
        case _:
            raise ValueError(f"Unknown dataset: {config.indist_dataset}")

    assert len(indist_dataset.classes) == config.num_classes
    return indist_dataset


def get_indist_dataloader(
    config: DataConfig,
    batch_size: int,
    split: str = "train",
    attack: bool = False,
    shuffle: bool = True,
    augm_type: str = "autoaugment_cutout",
    balanced: bool = True,
    use_ddp: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
    """Get in-distribution dataloader with optional DDP support."""
    dataset = get_indist_dataset(
        config, split=split, attack=attack, augm_type=augm_type
    )

    # Setup sampler for DDP
    sampler = None
    if use_ddp:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
        )
        shuffle = False  # Sampler handles shuffling

    # Only drop last batch during training, not during evaluation
    drop_last = (split == "train")

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=drop_last,
        pin_memory=True,
        num_workers=config.num_workers,
        persistent_workers=True,
    )


def get_outdist_dataset(
    config: DataConfig,
    split: str = "train",
    augm_type_generation: str = "original",
    tinyimages_loader: str = "innout",
    openimages_max_samples: int | None = None,
    openimages_augm: str | None = None,
):
    """Get out-distribution dataset."""
    match config.outdist_dataset:
        case "OpenImageO":
            LOGGER.info("Using OpenImageO outdist dataset")

            # Determine augmentation type
            if openimages_augm is not None:
                augm_type = openimages_augm
                LOGGER.info(
                    f"Using custom OpenImageO augmentation: {augm_type}"
                )
            else:
                augm_type = (
                    "generation_od_randomcrop"
                    if augm_type_generation == "generation_id_randomcrop"
                    else "generation_od"
                )

            transform = get_imageNet_augmentation(type=augm_type, out_size=224)
            dataset = torchvision.datasets.ImageFolder(
                config.outdist_std_dir, transform=transform
            )
            original_size = len(dataset)

            # Create random subset if max_samples is specified
            if (
                openimages_max_samples is not None
                and openimages_max_samples < len(dataset)
            ):
                generator = torch.Generator()
                generator.manual_seed(42)
                indices = torch.randperm(len(dataset), generator=generator)[
                    :openimages_max_samples
                ].tolist()
                outdist_dataset = torch.utils.data.Subset(dataset, indices)
                LOGGER.info(
                    f"OpenImageO dataset: Using {len(outdist_dataset)} samples out of {original_size} total samples"
                )
            else:
                outdist_dataset = dataset
                LOGGER.info(
                    f"OpenImageO dataset: Using all {original_size} samples"
                )

        case "tinyimages":
            if split != "train":
                LOGGER.warning(
                    f"TinyImages don't have a {split} split, using the train split instead"
                )
            outdist_dataset = get_tinyimages_dataset(
                data_dir=config.outdist_std_dir,
                augm_type=augm_type_generation,
                tinyimages_loader=tinyimages_loader,
            )
        case _:
            raise ValueError(
                f"Unknown outdist dataset: {config.outdist_dataset}"
            )

    return outdist_dataset


def get_outdist_dataloader(
    config: DataConfig,
    batch_size: int,
    split: str = "train",
    shuffle: bool = True,
    augm_type_generation: str = "original",
    tinyimages_loader: str = "innout",
    openimages_max_samples: int | None = None,
    openimages_augm: str | None = None,
    use_ddp: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
    """Get out-distribution dataloader with optional DDP support."""
    outdist_dataset = get_outdist_dataset(
        config=config,
        split=split,
        augm_type_generation=augm_type_generation,
        tinyimages_loader=tinyimages_loader,
        openimages_max_samples=openimages_max_samples,
        openimages_augm=openimages_augm,
    )

    # Setup sampler for DDP
    sampler = None
    if use_ddp:
        sampler = DistributedSampler(
            outdist_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
        )
        shuffle = False  # Sampler handles shuffling

    return torch.utils.data.DataLoader(
        outdist_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=True,
        pin_memory=True,
        num_workers=config.num_workers,
        persistent_workers=True,
    )
