"""EASE under the shared frozen ViT protocol.

Adapted from EASE (Zhou et al., CVPR 2024):
https://github.com/zhoudw-zdw/CVPR24-Ease

The canonical checkpoint is loaded once by the shared model loader, then its
frozen weights are copied into this method.  EASE adapters remain parallel
residual adapters in every transformer block.
"""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F

from models.adapter import Adapter
from utils.progress import epoch_progress

from .base import IncrementalMethod


class _EASEClassifier(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        bottleneck_dim: int,
        scale: float,
        dropout: float,
        include_initial_features: bool,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.bottleneck_dim, self.scale, self.dropout = bottleneck_dim, scale, dropout
        self.include_initial_features = include_initial_features
        self.adapters = nn.ModuleList()
        self.register_parameter("weight", None)

    def add_adapter(self, block_count: int, device: torch.device) -> None:
        self.adapters.append(
            nn.ModuleList(
                Adapter(self.feature_dim, self.bottleneck_dim, self.scale, self.dropout)
                for _ in range(block_count)
            ).to(device)
        )

    def expand_weight(self, seen_classes: int, device: torch.device) -> None:
        width = self.feature_dim * (len(self.adapters) + self.include_initial_features)
        expanded = torch.zeros(seen_classes, width, device=device)
        if self.weight is not None:
            expanded[: self.weight.shape[0], : self.weight.shape[1]].copy_(
                self.weight.detach()
            )
        self.weight = nn.Parameter(expanded, requires_grad=False)


class EASE(IncrementalMethod):
    """Task-adapter EASE migration for an immutable canonical ViT backbone."""

    def __init__(
        self,
        backbone: nn.Module,
        learning_rate: float,
        weight_decay: float,
        momentum: float,
        epochs: int,
        device: torch.device,
        *,
        adapter_dim: int = 64,
        adapter_scale: float = 0.1,
        adapter_dropout: float = 0.1,
        alpha: float = 0.1,
        beta: float = 0.0,
        include_initial_features: bool = False,
        use_reweight: bool = True,
        recalculate_similarity: bool = True,
        diagonal_only: bool = False,
        adapt_start_layer: int = 0,
    ) -> None:
        if not hasattr(backbone, "blocks") or any(
            not hasattr(block, "norm2") or not hasattr(block, "mlp")
            for block in backbone.blocks
        ):
            raise ValueError(
                "EASE requires standard ViT blocks with norm2 and mlp modules"
            )
        # Keep the shared canonical model untouched while reusing its loaded weights.
        self.backbone, self.device = copy.deepcopy(backbone).to(device), device
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.learning_rate, self.weight_decay, self.momentum, self.epochs = (
            learning_rate,
            weight_decay,
            momentum,
            epochs,
        )
        self.adapter_dim, self.adapter_scale, self.adapter_dropout = (
            adapter_dim,
            adapter_scale,
            adapter_dropout,
        )
        self.alpha, self.beta = alpha, beta
        # Adapters on blocks[adapt_start_layer:] only (official: all blocks, 0).
        self.start = adapt_start_layer
        self.include_initial_features = include_initial_features
        self.use_reweight, self.recalculate_similarity, self.diagonal_only = (
            use_reweight,
            recalculate_similarity,
            diagonal_only,
        )
        self.classifier: _EASEClassifier | None = None
        self.task_sizes: list[int] = []
        self._active_adapter: int | None = None
        self._adapter_inputs: list[torch.Tensor | None] = [None] * len(
            self.backbone.blocks
        )
        self._hooks = [
            block.norm2.register_forward_pre_hook(
                lambda _module, inputs, block_id=block_id: self._capture_adapter_input(
                    block_id, inputs[0]
                )
            )
            for block_id, block in enumerate(self.backbone.blocks)
        ] + [
            block.mlp.register_forward_hook(
                lambda _module, _inputs, output, block_id=block_id: (
                    self._inject_adapter(block_id, output)
                )
            )
            for block_id, block in enumerate(self.backbone.blocks)
        ]

    def _capture_adapter_input(self, block_id: int, features: torch.Tensor) -> None:
        self._adapter_inputs[block_id] = features

    def _inject_adapter(self, block_id: int, output: torch.Tensor) -> torch.Tensor:
        if self._active_adapter is None or block_id < self.start:
            return output
        assert self.classifier is not None
        features = self._adapter_inputs[block_id]
        assert features is not None
        return output + self.classifier.adapters[self._active_adapter][
            block_id - self.start
        ](features)

    def _features(self, images: torch.Tensor, adapter_id: int | None) -> torch.Tensor:
        self._active_adapter = adapter_id
        try:
            features = self.backbone(images.to(self.device))
        finally:
            self._active_adapter = None
        if isinstance(features, (tuple, list)):
            features = features[0]
        return features.flatten(1)

    def _ensure_classifier(
        self, feature_dim: int, seen_classes: int
    ) -> _EASEClassifier:
        if self.classifier is None:
            self.classifier = _EASEClassifier(
                feature_dim,
                self.adapter_dim,
                self.adapter_scale,
                self.adapter_dropout,
                self.include_initial_features,
            ).to(self.device)
        self.classifier.add_adapter(len(self.backbone.blocks) - self.start, self.device)
        self.classifier.expand_weight(seen_classes, self.device)
        return self.classifier

    def _subspaces(self, images: torch.Tensor) -> torch.Tensor:
        assert self.classifier is not None
        parts = [self._features(images, None)] if self.include_initial_features else []
        parts.extend(
            self._features(images, adapter_id)
            for adapter_id in range(len(self.classifier.adapters))
        )
        return torch.cat(parts, dim=1)

    def _task_range(self, task_id: int) -> slice:
        start = sum(self.task_sizes[:task_id])
        return slice(start, start + self.task_sizes[task_id])

    @staticmethod
    def _cosine_logits(features: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return F.linear(F.normalize(features, dim=1), F.normalize(weight, dim=1))

    def _reweighted_logits(self, subspaces: torch.Tensor) -> torch.Tensor:
        """Source EASE's task-wise diagonal/cross-subspace cosine scoring."""
        assert self.classifier is not None and self.classifier.weight is not None
        current_task = len(self.task_sizes) - 1
        logits = []
        for class_task in range(current_task + 1):
            scores = 0.0
            class_rows = self._task_range(class_task)
            if self.include_initial_features:
                initial = slice(0, self.classifier.feature_dim)
                scores = self.beta * self._cosine_logits(
                    subspaces[:, initial], self.classifier.weight[class_rows, initial]
                )
            for feature_task in range(current_task + 1):
                feature_columns = self._subspace_slice(feature_task)
                score = self._cosine_logits(
                    subspaces[:, feature_columns],
                    self.classifier.weight[class_rows, feature_columns],
                )
                if feature_task != class_task:
                    score = self.alpha * score / current_task
                scores = scores + score
            logits.append(scores)
        return torch.cat(logits, dim=1)

    @torch.no_grad()
    def _replace_prototypes(self, loader, new_classes: slice) -> None:
        assert self.classifier is not None and self.classifier.weight is not None
        sums = torch.zeros_like(self.classifier.weight)
        counts = torch.zeros(self.classifier.weight.shape[0], device=self.device)
        self.classifier.eval()
        for images, labels in loader:
            labels = labels.to(self.device)
            features = self._subspaces(images)
            sums.index_add_(0, labels, features)
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=counts.dtype))
        class_ids = torch.arange(
            new_classes.start, new_classes.stop, device=self.device
        )
        if torch.any(counts[class_ids] == 0):
            raise ValueError(
                "Every current-task class must have at least one training sample"
            )
        self.classifier.weight[new_classes] = (
            sums[new_classes] / counts[new_classes, None]
        )

    @torch.no_grad()
    def _reweight(self) -> None:
        """Fill upper task-subspace prototype blocks as in EASE."""
        assert self.classifier is not None and self.classifier.weight is not None
        if not self.use_reweight or self.diagonal_only or len(self.task_sizes) < 2:
            return
        current_task = len(self.task_sizes) - 1
        for source_task in range(current_task):
            target_tasks = (
                range(source_task + 1, current_task + 1)
                if self.recalculate_similarity
                else (current_task,)
            )
            old_rows = self._task_range(source_task)
            for target_task in target_tasks:
                reference_rows = slice(
                    self._task_range(target_task).start, self.classifier.weight.shape[0]
                )
                source_columns = self._subspace_slice(source_task)
                target_columns = self._subspace_slice(target_task)
                anchors = self.classifier.weight[reference_rows, source_columns]
                targets = self.classifier.weight[reference_rows, target_columns]
                queries = self.classifier.weight[old_rows, source_columns]
                similarity = F.softmax(self._cosine_logits(queries, anchors), dim=1)
                self.classifier.weight[old_rows, target_columns] = similarity @ targets

    def _subspace_slice(self, task_id: int) -> slice:
        offset = int(self.include_initial_features) + task_id
        return slice(
            offset * self.classifier.feature_dim,
            (offset + 1) * self.classifier.feature_dim,
        )  # type: ignore[union-attr]

    def train_task(
        self, loader, task_id: int, seen_classes: int, *, prototype_loader=None
    ) -> None:
        if task_id != len(self.task_sizes):
            raise ValueError("Tasks must be trained sequentially")
        first_images, _ = next(iter(loader))
        known_classes = sum(self.task_sizes)
        if seen_classes <= known_classes:
            raise ValueError("A task must introduce at least one class")
        classifier = self._ensure_classifier(
            self._features(first_images, None).shape[1], seen_classes
        )
        for adapter_bank in classifier.adapters[:-1]:
            adapter_bank.requires_grad_(False)
        current_adapter = classifier.adapters[-1]
        current_adapter.train()
        local_classifier = nn.Linear(
            classifier.feature_dim,
            seen_classes - known_classes,
            bias=False,
            device=self.device,
        )
        local_scale = nn.Parameter(torch.ones((), device=self.device))
        optimizer = torch.optim.SGD(
            [
                *current_adapter.parameters(),
                *local_classifier.parameters(),
                local_scale,
            ],
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )
        with epoch_progress(self.epochs, task_id) as progress:
            for _ in progress:
                for images, labels in loader:
                    labels = labels.to(self.device)
                    features = F.normalize(self._features(images, task_id), dim=1)
                    logits = local_scale * F.linear(
                        features, F.normalize(local_classifier.weight, dim=1)
                    )
                    local_labels = labels - known_classes
                    loss = F.cross_entropy(logits, local_labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    progress.update(
                        loss.item(),
                        correct=(logits.argmax(dim=1) == local_labels).sum().item(),
                        samples=labels.numel(),
                    )
                scheduler.step()
        self.task_sizes.append(seen_classes - known_classes)
        self._replace_prototypes(
            prototype_loader if prototype_loader is not None else loader,
            self._task_range(task_id),
        )
        self._reweight()
        current_adapter.requires_grad_(False)

    def logits(self, images: torch.Tensor) -> torch.Tensor:
        if self.classifier is None or self.classifier.weight is None:
            raise RuntimeError("Call train_task before evaluation")
        self.classifier.eval()
        with torch.no_grad():
            subspaces = self._subspaces(images)
            if self.use_reweight:
                return self._reweighted_logits(subspaces)
            return self._cosine_logits(subspaces, self.classifier.weight)

    def state_dict(self) -> dict:
        if self.classifier is None:
            return {}
        return {
            "classifier": self.classifier.state_dict(),
            "task_sizes": self.task_sizes,
        }

    def parameter_modules(self) -> tuple[nn.Module, ...]:
        return () if self.classifier is None else (self.classifier,)

    def load_state_dict(self, state: dict) -> None:
        if not state:
            return
        task_sizes = list(state["task_sizes"])
        weight = state["classifier"]["weight"]
        feature_dim = weight.shape[1] // (
            len(task_sizes) + int(self.include_initial_features)
        )
        classifier = _EASEClassifier(
            feature_dim,
            self.adapter_dim,
            self.adapter_scale,
            self.adapter_dropout,
            self.include_initial_features,
        ).to(self.device)
        for _ in task_sizes:
            classifier.add_adapter(len(self.backbone.blocks) - self.start, self.device)
        classifier.expand_weight(weight.shape[0], self.device)
        classifier.load_state_dict(state["classifier"])
        classifier.requires_grad_(False)
        self.classifier, self.task_sizes = classifier, task_sizes
