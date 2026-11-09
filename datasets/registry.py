"""Dataset construction using the benchmark's only transform recipe."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from torchvision import datasets, transforms

from utils.logging import logger

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_DATA_ROOT = Path.home() / "datasets"
CIFAR100_FOLDER = "cifar-100-python"
IMAGENET_R_ALIASES = ("imagenet-r", "imagenet_r", "ImageNet-R")


def _to_rgb(image):
    return image.convert("RGB")


def canonical_transform(train: bool, recipe: str = "canonical"):
    if recipe not in {"canonical", "sema"}:
        raise ValueError(f"Unknown preprocessing recipe: {recipe}")
    if train:
        ops = [
            transforms.Lambda(_to_rgb),
            transforms.RandomResizedCrop(
                224, scale=(0.05, 1.0) if recipe == "sema" else (0.8, 1.0)
            ),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        ops = [
            transforms.Lambda(_to_rgb),
            transforms.Resize(
                256,
                interpolation=(
                    transforms.InterpolationMode.BICUBIC
                    if recipe == "sema"
                    else transforms.InterpolationMode.BILINEAR
                ),
            ),
            transforms.CenterCrop(224),
        ]
    ops.append(transforms.ToTensor())
    if recipe == "canonical":
        ops.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(ops)


def evaluation_view(dataset):
    """Shallow-copy a training dataset with deterministic prototype transforms."""
    view = copy.copy(dataset)
    view.transform = canonical_transform(
        False, getattr(dataset, "_preprocessing_recipe", "canonical")
    )
    return view


def _expand(path: str | Path) -> Path:
    return Path(path).expanduser()


def _is_cifar100_parent(path: Path) -> bool:
    extract = path / CIFAR100_FOLDER
    return (extract / "train").is_file()


def _is_cifar100_extract(path: Path) -> bool:
    return path.name == CIFAR100_FOLDER and (path / "train").is_file()


def _is_imagenet_r(path: Path) -> bool:
    return (path / "train").is_dir() and (
        (path / "val").is_dir() or (path / "test").is_dir()
    )


def resolve_data_root(name: str, data_root: str | Path | None = None) -> Path:
    root = _expand(data_root or DEFAULT_DATA_ROOT)
    if name == "cifar100":
        if _is_cifar100_parent(root):
            return root
        if _is_cifar100_extract(root):
            return root.parent
        raise FileNotFoundError(
            f"CIFAR-100 not found under {root}. Expected {CIFAR100_FOLDER}/train "
            f"(pass ~/datasets or ~/datasets/{CIFAR100_FOLDER}, or --download)."
        )
    if name == "imagenet_r":
        if _is_imagenet_r(root):
            return root
        for alias in IMAGENET_R_ALIASES:
            candidate = root / alias
            if _is_imagenet_r(candidate):
                return candidate
        raise FileNotFoundError(
            f"ImageNet-R not found under {root}. Expected train/ plus val/ or test/ "
            f"(or a subdirectory named imagenet-r)."
        )
    if name == "vtab":
        for candidate in (root, root / "vtab"):
            if _is_imagenet_r(candidate):
                return candidate
        raise FileNotFoundError(
            f"VTAB-CIL not found under {root}. Expected vtab/train and vtab/test "
            f"(the RevisitingCIL release)."
        )
    raise ValueError(f"Unsupported benchmark dataset: {name}")


def build_datasets(
    name: str,
    data_root: str | Path | None = None,
    download: bool = False,
    preprocessing: str = "canonical",
):
    requested = _expand(data_root or DEFAULT_DATA_ROOT)
    try:
        root = resolve_data_root(name, requested)
    except FileNotFoundError:
        if name == "cifar100" and download:
            root = requested.parent if requested.name == CIFAR100_FOLDER else requested
            logger.info(f"Downloading CIFAR-100 into {root}")
        else:
            raise
    logger.info(f"Data root {root}")
    if name == "cifar100":
        result = (
            datasets.CIFAR100(
                root,
                train=True,
                transform=canonical_transform(True, preprocessing),
                download=download,
            ),
            datasets.CIFAR100(
                root,
                train=False,
                transform=canonical_transform(False, preprocessing),
                download=download,
            ),
        )
    else:
        evaluation_root = root / "val"
        if not evaluation_root.is_dir():
            evaluation_root = root / "test"
        result = (
            datasets.ImageFolder(
                root / "train", transform=canonical_transform(True, preprocessing)
            ),
            datasets.ImageFolder(
                evaluation_root, transform=canonical_transform(False, preprocessing)
            ),
        )
    for dataset in result:
        dataset._preprocessing_recipe = preprocessing
    return result


def _keep(dataset, indices):
    """Shallow copy of an ImageFolder/CIFAR dataset restricted to ``indices``."""
    part = copy.copy(dataset)
    part.targets = [dataset.targets[i] for i in indices]
    if hasattr(dataset, "samples"):  # ImageFolder
        part.samples = [dataset.samples[i] for i in indices]
        part.imgs = part.samples
    else:  # CIFAR
        part.data = dataset.data[indices]
    return part


def holdout_split(train, fraction: float, seed: int = 0):
    """Per-class held-out validation split of the training set (for tuning).

    Returns (train_part, validation_part); the validation part uses the
    deterministic evaluation transform.  The split depends only on ``seed``,
    so every tuning run sees the same validation images whatever its class order.
    """
    rng = np.random.RandomState(seed)
    by_class: dict[int, list[int]] = {}
    for i, target in enumerate(train.targets):
        by_class.setdefault(int(target), []).append(i)
    held = []
    for indices in by_class.values():
        n = max(1, int(round(fraction * len(indices))))
        held.extend(rng.choice(indices, n, replace=False).tolist())
    held_set = set(held)
    kept = [i for i in range(len(train.targets)) if i not in held_set]
    return _keep(train, kept), evaluation_view(_keep(train, sorted(held)))
