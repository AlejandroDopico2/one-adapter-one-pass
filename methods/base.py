"""Minimal contract for method migrations."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class IncrementalMethod(ABC):
    @abstractmethod
    def train_task(self, loader, task_id: int, seen_classes: int) -> None:
        """Update method state from this task's labelled loader."""

    @abstractmethod
    def logits(self, images: torch.Tensor) -> torch.Tensor:
        """Return logits over every class seen so far."""

    @abstractmethod
    def state_dict(self) -> dict:
        """Return serializable method state."""

    def parameter_modules(self) -> tuple[torch.nn.Module, ...]:
        """Return persistent non-backbone modules for parameter accounting."""
        return ()
