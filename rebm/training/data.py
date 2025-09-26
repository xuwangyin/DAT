import os.path
import pathlib
import random
from typing import Iterable, Literal

import datasets
import torch
import torch.utils.data
import torchvision.datasets
import torchvision.transforms as transforms

import InNOutRobustness.utils.datasets as dl
from InNOutRobustness.utils.datasets.augmentations.cifar_augmentation import (
    get_cifar10_augmentation,
)


class FilteredImageFolder(torchvision.datasets.ImageFolder):
    def __init__(
        self,
        bad_indices: Iterable[int],
        **kwargs,
    ):
        self.bad_indices = bad_indices  # Need to set first since super().__init__ calls find_classes
        super().__init__(**kwargs)

    def find_classes(
        self, directory: str | pathlib.Path
    ) -> tuple[list[str], dict[str, int]]:
        """Override find_classes to filter out bird classes."""
        classes, class_to_idx = super().find_classes(directory)

        # Remove the bad indices from the classes
        classes = [
            cls for i, cls in enumerate(classes) if i not in self.bad_indices
        ]
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        return classes, class_to_idx


# fmt: off
# TODO: We should recheck these indices at some point.
IMAGENET_BIRD_INDICES = [
    7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
    127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146,
    162, 448, 517, 665, 715, 793, 885, 902, 908
]
# fmt: on


def get_hf_imagenet256_dataset(
    hf_cache_dir: str,
    shuffle: bool = True,
    split: Literal["train", "validation", "test"] = "train",
    shuffle_seed: int = 42,
    shuffle_buffer_size: int = 10_000,
    filter: str | None = None,
):
    """
    If this function crashes due to the dataset not being found, double check
    that you have followed the instructions in the README that download the
    data.
    """
    tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(256, interpolation=2),
            transforms.RandomHorizontalFlip(),
        ]
    )

    def transform_fn(examples):
        examples["pixel_values"] = [
            tf(image.convert("RGB")) for image in examples["image"]
        ]
        return examples

    ds = datasets.load_dataset(
        os.path.join(hf_cache_dir, "imagenet-1k"), streaming=True, split=split
    )

    # removed filter type images from imagenet
    if filter == "birds" or filter == "bird":
        # see prepare_dataset/imagenet_no_birds.py to see how i got these
        ds = ds.filter(lambda x: x["label"] not in IMAGENET_BIRD_INDICES)
    elif filter == "only birds" or filter == "only bird":
        # just the birds from imagenet
        ds = ds.filter(lambda x: x["label"] in IMAGENET_BIRD_INDICES)
    elif filter is not None:
        raise ValueError(f"Unknown filter: {filter}")

    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size, seed=shuffle_seed)

    return ds.map(
        transform_fn,
        remove_columns=["image"],
        batched=True,
    ).with_format("torch")


def get_imagenet256_dataset(
    datadir,
    split: Literal["train", "test", "val"] = "train",
    interpolation=2,
    transform=None,
    drop_birds: bool = True,
):
    if split == "test":
        # imagenet doesn't have labels on the test set
        # TODO: Implement https://pytorch.org/vision/main/generated/torchvision.datasets.DatasetFolder.html#torchvision.datasets.DatasetFolder.find_classes
        raise NotImplementedError("Need to override find_classes for test set")

    # the straight up image folder version
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(256, interpolation=interpolation),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    if drop_birds:
        return FilteredImageFolder(
            bad_indices=IMAGENET_BIRD_INDICES,
            root=os.path.join(datadir, split),
            transform=transform,
        )

    return torchvision.datasets.ImageFolder(
        root=os.path.join(datadir, split), transform=transform
    )


def get_lsun_bird_dataset(
    lsun_path: str,
    split: Literal["train", "val"] = "train",
):
    return torchvision.datasets.ImageFolder(
        root=os.path.join(lsun_path, split),
        transform=transforms.Compose([transforms.ToTensor()]),
    )


def get_lsun_bird__cluster_dataset(lsun_path: str):
    return torchvision.datasets.ImageFolder(
        root=lsun_path,
        transform=transforms.Compose([transforms.ToTensor()]),
    )


def get_hf_afhqv2_cat_dataset(
    hf_cache_dir: str,
    extra_transforms: list[torch.nn.Module] | None = None,
    split: Literal["train", "test"] = "train",
    num_proc: int = 8,
) -> tuple[torch.utils.data.IterableDataset, int]:
    """
    If this function crashes due to the dataset not being found, double check
    that you have followed the instructions in the README that download the
    data.
    """
    tf = transforms.Compose(
        [
            # Resize so minimum dimension is 256, then crop to 256x256
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.RandomHorizontalFlip(),
        ]
        + ([] if extra_transforms is None else extra_transforms)
    )

    def transform_fn(examples):
        examples["pixel_values"] = [
            tf(image.convert("RGB")) for image in examples["image"]
        ]
        return examples

    ds = (
        datasets.load_dataset(
            os.path.join(hf_cache_dir, "huggan___afh_qv2"),
            split="train",
            num_proc=num_proc,
        )
        .filter(lambda x: x == 0, input_columns="label")
        .train_test_split(test_size=0.1, seed=0)[split]
    )

    return ds.map(
        transform_fn, remove_columns=["image"], batched=True
    ).with_format("torch")


def get_uaec_dataset(
    data_dir: str,
    split: Literal["train", "test", "extras"] = "train",
    target_class: Literal["bicycle", "bird"] | None = None,
    tfs: transforms.Compose | None = None,
) -> torchvision.datasets.ImageFolder:
    ds = torchvision.datasets.ImageFolder(
        root=pathlib.Path(data_dir) / split,
        transform=tfs
        or transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(
                    256
                ),  # causing wildly different model scores (see load tradesv2.ipynb)
                transforms.ToTensor(),
            ]
        ),
    )

    if target_class is not None:
        ds.samples = [
            sample
            for sample in ds.samples
            if target_class in ds.classes[sample[1]]
        ]

    return ds


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

def get_tinyimages_dataset_original(data_dir):
    from GOOD.tiny_utils.tinyimages_80mn_loader import TinyImages
    cifar10_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    dataset = TinyImages(
        tiny_path=data_dir,
        transform=cifar10_transform,
        exclude_cifar=["H", "CEDA11"],
    )
    return dataset


def get_tinyimages_dataset(
    data_dir,
    augm_type: str = "autoaugment_cutout",
    tinyimages_loader: Literal["GOOD", "innout"] = "GOOD",
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

    if tinyimages_loader == "innout":
        dataset = dl.TinyImagesDataset(
            data_dir, transform, exclude_cifar=True, exclude_cifar10_1=True
        )
    elif tinyimages_loader == "GOOD":
        dataset = TinyImages(
            tiny_path=data_dir,
            transform=transform,
            exclude_cifar=["H", "CEDA11"],
        )
    else:
        raise ValueError(f"Unknown tinyimages_loader: {tinyimages_loader}")
    return dataset


def get_afhq_dataset(
    data_dir: str,
    extra_transform: bool = False,
):
    if extra_transform:
        img_size = 256
        scale, ratio = (0.8, 1.0), (0.9, 1.1)
        crop = transforms.RandomResizedCrop(img_size, scale=scale, ratio=ratio)
        rand_crop = transforms.Lambda(
            lambda x: crop(x) if random.random() < 0.5 else x
        )
        transform = transforms.Compose(
            [
                rand_crop,
                transforms.Resize([img_size, img_size]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    else:
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
        )

    return torchvision.datasets.ImageFolder(root=data_dir, transform=transform)


def get_restrictedimagenet(data_dir: str):
    # return torchvision.datasets.ImageFolder(
    #     data_dir,
    #     transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]))

    # transform = transforms.Compose([transforms.RandomResizedCrop(256, scale=(0.7, 1.0), ratio=(0.9, 1.1), interpolation=2),
    #                                 transforms.RandomHorizontalFlip(),
    #                                 transforms.ToTensor()])
    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
    )
    return torchvision.datasets.ImageFolder(data_dir, transform)


class RemapLabelsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, label_offset):
        self.dataset = dataset
        self.label_offset = label_offset

    def __getitem__(self, index):
        data, label = self.dataset[index]
        return data, label + self.label_offset

    def __len__(self):
        return len(self.dataset)
