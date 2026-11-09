"""ROLANN: recursive closed-form ridge classifier over stored feature statistics (SVD form).

Derived from Ani-CIFNet's ``models/ROLANN.py``. It accumulates, per class, the moments of the features
seen so far and recomputes the classifier analytically; the paper's drift compensation (Eq. 3) is applied
to these statistics by ``methods.shared_adapter`` and ``methods.srw``.
"""

from __future__ import annotations

import torch
from torch import nn


class ROLANN(nn.Module):
    """Recursive orthogonal linear ridge classifier from Ani-CIFNet/models/ROLANN.py."""

    def __init__(self, ridge: float):
        super().__init__()
        self.ridge = ridge
        self.mg: list[torch.Tensor] = []
        self.ug: torch.Tensor | None = None
        self.sg: torch.Tensor | None = None
        self.weights: list[torch.Tensor | None] = []

    @property
    def num_classes(self) -> int:
        return len(self.mg)

    @torch.no_grad()
    def update(
        self, features: torch.Tensor, labels: torch.Tensor, classes: torch.Tensor
    ) -> None:
        augmented = torch.cat((torch.ones_like(features[:, :1]), features), dim=1).T
        basis, singular_values, _ = torch.linalg.svd(augmented, full_matrices=False)
        targets = torch.nn.functional.one_hot(
            labels, num_classes=int(classes.max()) + 1
        ).float()
        moments = targets[:, classes].T @ augmented.T

        for index, class_id in enumerate(classes.tolist()):
            while len(self.mg) <= class_id:
                self.mg.append(torch.zeros_like(moments[index]))
            self.mg[class_id].add_(moments[index])

        if self.ug is None:
            self.ug, self.sg = basis, singular_values
        else:
            assert self.sg is not None
            merged = torch.cat(
                (self.ug * self.sg.unsqueeze(0), basis * singular_values.unsqueeze(0)),
                dim=1,
            )
            self.ug, self.sg, _ = torch.linalg.svd(merged, full_matrices=False)

        assert self.ug is not None and self.sg is not None
        moments = torch.stack(self.mg)
        inverse = 1.0 / (self.sg.square().clamp_min(1e-10) + self.ridge)
        solved = self.ug @ (inverse.unsqueeze(1) * (self.ug.T @ moments.T))
        self.weights = list(solved.T.unbind(0))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not self.weights:
            raise RuntimeError("Call train_task before evaluation")
        augmented = torch.cat((torch.ones_like(features[:, :1]), features), dim=1)
        return (
            augmented
            @ torch.stack([weight for weight in self.weights if weight is not None]).T
        )

    def export_state(self) -> dict:
        return {
            "ridge": self.ridge,
            "mg": self.mg,
            "ug": self.ug,
            "sg": self.sg,
            "weights": self.weights,
        }

    def import_state(self, state: dict) -> None:
        self.ridge = state["ridge"]
        self.mg, self.ug, self.sg, self.weights = (
            state["mg"],
            state["ug"],
            state["sg"],
            state["weights"],
        )
