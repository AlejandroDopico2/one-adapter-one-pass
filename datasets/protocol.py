"""Immutable class orders and task-local dataset views."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml
from torch.utils.data import Dataset


def load_class_order(path: str | Path, num_classes: int) -> tuple[list[int], str]:
    raw = Path(path).read_bytes()
    document = yaml.safe_load(raw)
    order = document.get("class_order")
    if not isinstance(order, list) or sorted(order) != list(range(num_classes)):
        raise ValueError(
            f"{path} must contain each class ID 0..{num_classes - 1} exactly once"
        )
    return order, hashlib.sha256(raw).hexdigest()


def shuffle_class_order(
    order: Sequence[int], seed: int, block: int | None = None
) -> list[int]:
    """Shuffled class order.  With ``block``, consecutive groups of ``block``
    classes (e.g. one VTAB domain each) stay together: the group order and the
    order inside each group are shuffled, so every task is one group."""
    rng = np.random.RandomState(seed)
    if not block:
        return rng.permutation(order).tolist()
    groups = [list(order[i : i + block]) for i in range(0, len(order), block)]
    return [
        int(c) for g in rng.permutation(len(groups)) for c in rng.permutation(groups[g])
    ]


def concentration_class_order(
    order: Sequence[int], seed: int, k: int, group: int = 10
) -> list[int]:
    """Shuffled order in which every task mixes exactly ``k`` groups (domains).

    Groups (consecutive ``group`` classes, e.g. VTAB domains) are split into
    ``k`` chunks.  Task t takes chunk j of group (t - j) mod G for j < k, so
    every group spreads over k consecutive tasks and every task has ``group``
    classes from k groups.  Group order, classes inside each group, task order
    and class order inside each task are all shuffled.
    """
    rng = np.random.RandomState(seed)
    groups = [list(order[i : i + group]) for i in range(0, len(order), group)]
    G = len(groups)
    if not 1 <= k <= G:
        raise ValueError(f"k must be in [1, {G}]")
    groups = [list(rng.permutation(groups[g])) for g in rng.permutation(G)]
    base, rem = divmod(group, k)
    sizes = [base + 1] * rem + [base] * (k - rem)
    bounds = np.cumsum([0] + sizes)
    chunks = [[g[bounds[j] : bounds[j + 1]] for j in range(k)] for g in groups]
    tasks = [
        list(rng.permutation([c for j in range(k) for c in chunks[(t - j) % G][j]]))
        for t in range(G)
    ]
    return [int(c) for t in rng.permutation(G) for c in tasks[t]]


@dataclass(frozen=True)
class TaskProtocol:
    class_order: tuple[int, ...]
    initial_classes: int
    increment_classes: int

    def __post_init__(self) -> None:
        if self.initial_classes < 1 or self.increment_classes < 1:
            raise ValueError("Task sizes must be positive")
        if len(self.class_order) < self.initial_classes:
            raise ValueError("Initial task exceeds class order")

    @property
    def num_tasks(self) -> int:
        remaining = len(self.class_order) - self.initial_classes
        return 1 + (remaining + self.increment_classes - 1) // self.increment_classes

    def classes_for_task(self, task_id: int) -> tuple[int, ...]:
        if not 0 <= task_id < self.num_tasks:
            raise IndexError(f"Unknown task {task_id}")
        start = (
            0
            if task_id == 0
            else self.initial_classes + (task_id - 1) * self.increment_classes
        )
        size = self.initial_classes if task_id == 0 else self.increment_classes
        return self.class_order[start : start + size]

    def seen_classes(self, task_id: int) -> tuple[int, ...]:
        return self.class_order[
            : sum(len(self.classes_for_task(i)) for i in range(task_id + 1))
        ]

    def remap(self, original_label: int) -> int:
        return self.class_order.index(original_label)


class RemappedSubset(Dataset):
    """Samples selected original labels and emits their fixed incremental IDs."""

    def __init__(
        self,
        dataset: Dataset,
        original_labels: Sequence[int],
        class_order: Sequence[int],
    ):
        self.dataset = dataset
        selected = set(original_labels)
        self.indices = [
            i
            for i, target in enumerate(dataset.targets)  # type: ignore[attr-defined]
            if target in selected
        ]
        self._remap = {label: new_label for new_label, label in enumerate(class_order)}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        image, target = self.dataset[self.indices[index]]
        return image, self._remap[int(target)]


def sample_identity(dataset: Dataset, index: int) -> tuple[str, int, str]:
    """Return ``(sample_key, class_id, absolute_path)`` without reordering.

    ``sample_key`` is the image path relative to the image-folder root. Wrappers
    that hide paths delegate to the underlying dataset and keep its index order.
    """
    if isinstance(dataset, RemappedSubset):
        key, original, path = sample_identity(dataset.dataset, dataset.indices[index])
        return key, dataset._remap[original], path
    samples = getattr(dataset, "samples", None)
    root = getattr(dataset, "root", None)
    targets = getattr(dataset, "targets", None)
    if samples is None or root is None or targets is None:
        raise TypeError(
            f"{type(dataset).__name__} does not expose image paths "
            "(expected an image-folder dataset with samples, root, and targets)"
        )
    absolute = Path(samples[index][0]).resolve()
    key = absolute.relative_to(Path(root).resolve()).as_posix()
    return key, int(targets[index]), str(absolute)
