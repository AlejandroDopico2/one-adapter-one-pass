"""Frozen-backbone linear reference used to validate the common runner."""

from __future__ import annotations

import torch
from torch import nn

from utils.progress import epoch_progress

from .base import IncrementalMethod


class FrozenLinear(IncrementalMethod):
    def __init__(
        self,
        backbone: nn.Module,
        learning_rate: float,
        weight_decay: float,
        momentum: float,
        epochs: int,
        device: torch.device,
    ):
        self.backbone, self.device = backbone.to(device), device
        self.learning_rate, self.weight_decay, self.momentum = (
            learning_rate,
            weight_decay,
            momentum,
        )
        self.epochs = epochs
        self.classifier: nn.Linear | None = None

    @torch.no_grad()
    def _features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images.to(self.device))
        if isinstance(features, (tuple, list)):
            features = features[0]
        return features.flatten(1)

    def _expand_classifier(self, feature_dim: int, seen_classes: int) -> None:
        if self.classifier is not None and self.classifier.out_features == seen_classes:
            return
        expanded = nn.Linear(feature_dim, seen_classes, device=self.device)
        if self.classifier is not None:
            with torch.no_grad():
                expanded.weight[: self.classifier.out_features].copy_(
                    self.classifier.weight
                )
                expanded.bias[: self.classifier.out_features].copy_(
                    self.classifier.bias
                )
        self.classifier = expanded

    def train_task(self, loader, task_id: int, seen_classes: int) -> None:
        first_images, _ = next(iter(loader))
        self._expand_classifier(self._features(first_images).shape[1], seen_classes)
        assert self.classifier is not None
        optimizer = torch.optim.SGD(
            self.classifier.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        self.classifier.train()
        with epoch_progress(self.epochs, task_id) as progress:
            for _ in progress:
                for images, labels in loader:
                    labels = labels.to(self.device)
                    logits = self.classifier(self._features(images))
                    loss = nn.functional.cross_entropy(logits, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    progress.update(
                        loss.item(),
                        correct=(logits.argmax(dim=1) == labels).sum().item(),
                        samples=labels.numel(),
                    )

    def logits(self, images: torch.Tensor) -> torch.Tensor:
        if self.classifier is None:
            raise RuntimeError("Call train_task before evaluation")
        self.classifier.eval()
        return self.classifier(self._features(images))

    def state_dict(self) -> dict:
        if self.classifier is None:
            return {}
        return {
            "classifier": self.classifier.state_dict(),
            "out_features": self.classifier.out_features,
        }

    def parameter_modules(self) -> tuple[nn.Module, ...]:
        return () if self.classifier is None else (self.classifier,)

    def load_state_dict(self, state: dict) -> None:
        if not state:
            return
        feature_dim = state["classifier"]["weight"].shape[1]
        self._expand_classifier(feature_dim, state["out_features"])
        assert self.classifier is not None
        self.classifier.load_state_dict(state["classifier"])
