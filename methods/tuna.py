"""TUNA under the shared frozen ViT protocol.

Port of the official ICCV 2025 code (https://github.com/LAMDA-CL/ICCV2025-TUNA,
models/tuna.py, backbone/vit_tuna.py, backbone/linears.py):

* one AdaptFormer adapter per block (all 12 blocks, bottleneck 16, scale 0.1,
  dropout 0.1, no LayerNorm, LoRA-style init), parallel to the MLP;
* the adapter keeps training across tasks; its snapshot at the end of each task
  is that task's "task-specific" adapter; an orthogonality penalty separates
  the current up-projections from earlier snapshots;
* the "universal" adapter is the EMR merge of all snapshots;
* a cosine head per task (no bias, trunc-normal init; features and weights
  L2-normalised), all heads trainable, trained with CosFace (s=20) on the
  current task's classes;
* classifier alignment with Gaussian feature samples (class mean and
  covariance stored when the class is learned);
* inference: every snapshot predicts, the lowest-entropy one is kept and its
  softmax is added to the merged adapter's softmax.

Quirks of the official code kept on purpose, so the comparison is faithful:
the alignment decay uses a negative task size (``known - total``), and its cosine schedule
has ``T_max=5`` over 30 epochs.
"""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import functional as F

from models.adapter import Adapter
from utils.progress import epoch_progress

from .base import IncrementalMethod


def _cosface(
    logits: torch.Tensor, labels: torch.Tensor, scale: float, margin: float
) -> torch.Tensor:
    """Official AngularPenaltySMLoss('cosface') on cosine logits."""
    target = logits.gather(1, labels[:, None]).squeeze(1)
    numerator = scale * (target - margin)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(1, labels[:, None], False)
    others = logits[mask].view(len(logits), -1)
    denominator = torch.exp(numerator) + torch.exp(scale * others).sum(1)
    return -(numerator - torch.log(denominator)).mean()


def _emr_merge(snapshots: list[nn.ModuleList]) -> dict[str, torch.Tensor]:
    """Official emr_merge on raw adapter parameters (rescalers unused, as there)."""
    states = [s.state_dict() for s in snapshots]
    merged = {}
    for key in states[0]:
        stack = torch.stack([state[key] for state in states])
        flag = (stack.mean(0) > 0).to(stack.dtype) * 2 - 1
        agree = (stack * flag) > 0
        merged[key] = (stack.abs() * agree).max(0).values * flag
    return merged


class TUNA(IncrementalMethod):
    def __init__(
        self,
        backbone: nn.Module,
        learning_rate: float,
        weight_decay: float,
        momentum: float,
        epochs: int,
        device: torch.device,
        *,
        adapter_dim: int = 16,
        adapter_scale: float = 0.1,
        adapter_dropout: float = 0.1,
        scale: float = 20.0,
        margin: float = 0.0,
        orth_reg: float = 0.001,
        use_orth: bool = True,
        ca_epochs: int = 30,
        ca_lr: float = 0.005,
        ca_decay: bool = True,
        samples_per_class: int = 160,
        adapt_start_layer: int = 0,
    ) -> None:
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
        self.scale, self.margin = scale, margin
        self.orth_reg, self.use_orth = orth_reg, use_orth
        self.ca_epochs, self.ca_lr, self.ca_decay = ca_epochs, ca_lr, ca_decay
        self.samples_per_class = samples_per_class
        dim = self.backbone.blocks[0].norm2.normalized_shape[0]
        self.dim = dim
        # Adapters on blocks[adapt_start_layer:] only (official: all blocks, 0).
        self.start = adapt_start_layer
        self.current = nn.ModuleList(
            Adapter(dim, adapter_dim, adapter_scale, adapter_dropout)
            for _ in self.backbone.blocks[self.start :]
        ).to(device)
        self.snapshots: list[nn.ModuleList] = []
        self.merged: nn.ModuleList | None = None
        self.heads = nn.ModuleList()
        self.cls_mean: dict[int, torch.Tensor] = {}
        self.cls_cov: dict[int, torch.Tensor] = {}
        self.task_sizes: list[int] = []
        self._active: nn.ModuleList | None = None
        self._inputs: list[torch.Tensor | None] = [None] * len(self.current)
        for index, block in enumerate(self.backbone.blocks[self.start :]):
            block.norm2.register_forward_pre_hook(
                lambda _m, inputs, index=index: self._inputs.__setitem__(
                    index, inputs[0]
                )
            )
            block.mlp.register_forward_hook(
                lambda _m, _i, output, index=index: (
                    output
                    if self._active is None
                    else output + self._active[index](self._inputs[index])
                )
            )

    # -- forward -------------------------------------------------------------
    def _features(
        self, images: torch.Tensor, adapters: nn.ModuleList | None
    ) -> torch.Tensor:
        self._active = adapters
        try:
            features = self.backbone(images.to(self.device))
        finally:
            self._active = None
        if isinstance(features, (tuple, list)):
            features = features[0]
        return features.flatten(1)

    def _head(self, features: torch.Tensor) -> torch.Tensor:
        # Official SimpleContinualLinear: cosine of features and per-task weights.
        features = F.normalize(features, dim=1)
        return torch.cat(
            [
                F.linear(features, F.normalize(head.weight, dim=1))
                for head in self.heads
            ],
            dim=1,
        )

    # -- training ------------------------------------------------------------
    def _orth_loss(self) -> torch.Tensor:
        total = torch.zeros((), device=self.device)
        for snapshot in self.snapshots:
            loss = 0.0
            for current, previous in zip(self.current, snapshot):
                a = F.normalize(current.up.weight, dim=1)
                b = F.normalize(previous.up.weight, dim=1)
                loss = loss + (a @ b.T).mean().abs() / len(self.current)
            total = total + loss
        return total

    def train_task(
        self, loader, task_id: int, seen_classes: int, *, prototype_loader=None
    ) -> None:
        known = sum(self.task_sizes)
        new = seen_classes - known
        head = nn.Linear(self.dim, new, bias=False)
        nn.init.trunc_normal_(head.weight, std=0.02)  # on CPU: CUDA erfinv needs NVRTC
        head = head.to(self.device)
        self.heads.append(head)
        params = [
            {
                "params": list(self.current.parameters()),
                "lr": self.learning_rate,
                "weight_decay": self.weight_decay,
            },
            {
                "params": list(self.heads.parameters()),
                "lr": self.learning_rate,
                "weight_decay": self.weight_decay,
            },
        ]
        optimizer = torch.optim.SGD(params, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=0
        )
        self.current.train()
        with epoch_progress(self.epochs, task_id) as progress:
            for _ in progress:
                for images, labels in loader:
                    labels = labels.to(self.device)
                    logits = self._head(self._features(images, self.current))
                    loss = _cosface(
                        logits[:, known:], labels - known, self.scale, self.margin
                    )
                    if task_id > 0 and self.use_orth:
                        loss = loss + self._orth_loss() * self.orth_reg * torch.exp(
                            -torch.tensor(task_id + 1.0, device=self.device)
                        )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    progress.update(
                        loss.item(),
                        correct=(logits.argmax(1) == labels).sum().item(),
                        samples=labels.numel(),
                    )
                scheduler.step()
        self.task_sizes.append(new)
        snapshot = copy.deepcopy(self.current).eval()
        snapshot.requires_grad_(False)
        self.snapshots.append(snapshot)
        if task_id > 0:
            self.merged = copy.deepcopy(self.current).eval()
            self.merged.load_state_dict(_emr_merge(self.snapshots))
            self.merged.requires_grad_(False)
        self._compute_stats(prototype_loader or loader, known, seen_classes)
        if task_id > 0:
            self._align(known, seen_classes)

    @torch.no_grad()
    def _compute_stats(self, loader, known: int, total: int) -> None:
        self.current.eval()
        feats, labels = [], []
        for images, targets in loader:
            feats.append(self._features(images, self.current))
            labels.append(targets.to(self.device))
        feats, labels = torch.cat(feats), torch.cat(labels)
        eye = torch.eye(self.dim, device=self.device) * 1e-4
        for class_id in range(known, total):
            members = feats[labels == class_id]
            self.cls_mean[class_id] = members.mean(0)
            self.cls_cov[class_id] = torch.cov(members.T) + eye

    def _align(self, known: int, total: int) -> None:
        optimizer = torch.optim.SGD(
            [
                {
                    "params": list(self.heads.parameters()),
                    "lr": self.ca_lr,
                    "weight_decay": self.weight_decay,
                }
            ],
            lr=self.ca_lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
        task_size = known - total  # official: negative on purpose-or-not; kept
        per_class = self.samples_per_class
        for _ in range(self.ca_epochs):
            data, labels = [], []
            for class_id in range(total):
                mean = self.cls_mean[class_id].double()
                if self.ca_decay:
                    task = class_id // task_size
                    mean = mean * (0.9 + (task + 1) / (len(self.task_sizes)) * 0.1)
                sampler = MultivariateNormal(
                    mean.float(), self.cls_cov[class_id].float()
                )
                data.append(sampler.sample((per_class,)))
                labels.extend([class_id] * per_class)
            data = torch.cat(data)
            labels = torch.tensor(labels, device=self.device)
            order = torch.randperm(len(data), device=self.device)
            data, labels = data[order], labels[order]
            for step in range(total):
                chunk = slice(step * per_class, (step + 1) * per_class)
                loss = F.cross_entropy(
                    self.scale * self._head(data[chunk]), labels[chunk]
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

    # -- inference -----------------------------------------------------------
    @torch.no_grad()
    def logits(self, images: torch.Tensor) -> torch.Tensor:
        total = sum(self.task_sizes)
        if len(self.snapshots) == 1:
            return self._head(self._features(images, self.snapshots[0]))[:, :total]
        specific, entropy = [], []
        for snapshot in self.snapshots:
            logit = self._head(self._features(images, snapshot))[:, :total] * self.scale
            prob = F.softmax(logit, dim=1)
            specific.append(logit)
            entropy.append(-(prob * torch.log(prob + 1e-10)).sum(1))
        specific, entropy = torch.stack(specific), torch.stack(entropy)
        chosen = specific[
            entropy.argmin(0), torch.arange(specific.shape[1], device=specific.device)
        ]
        merged = self._head(self._features(images, self.merged))[:, :total] * self.scale
        return F.softmax(merged, dim=1) + F.softmax(chosen, dim=1)

    @torch.no_grad()
    def selection(self, images: torch.Tensor) -> torch.Tensor:
        """Index of the snapshot TUNA's entropy rule picks for each image."""
        total = sum(self.task_sizes)
        entropy = []
        for snapshot in self.snapshots:
            prob = F.softmax(
                self._head(self._features(images, snapshot))[:, :total] * self.scale,
                dim=1,
            )
            entropy.append(-(prob * torch.log(prob + 1e-10)).sum(1))
        return torch.stack(entropy).argmin(0)

    # -- state ---------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "current": self.current.state_dict(),
            "snapshots": [s.state_dict() for s in self.snapshots],
            "merged": None if self.merged is None else self.merged.state_dict(),
            "heads": [h.state_dict() for h in self.heads],
            "head_sizes": [h.out_features for h in self.heads],
            "task_sizes": self.task_sizes,
            "cls_mean": self.cls_mean,
            "cls_cov": self.cls_cov,
        }

    def load_state_dict(self, state: dict) -> None:
        self.current.load_state_dict(state["current"])
        self.snapshots = []
        for snapshot_state in state["snapshots"]:
            snapshot = copy.deepcopy(self.current).eval()
            snapshot.load_state_dict(snapshot_state)
            snapshot.requires_grad_(False)
            self.snapshots.append(snapshot)
        if state["merged"] is not None:
            self.merged = copy.deepcopy(self.current).eval()
            self.merged.load_state_dict(state["merged"])
        self.heads = nn.ModuleList()
        for size, head_state in zip(state["head_sizes"], state["heads"]):
            head = nn.Linear(self.dim, size, bias=False).to(self.device)
            head.load_state_dict(head_state)
            self.heads.append(head)
        self.task_sizes = list(state["task_sizes"])
        self.cls_mean, self.cls_cov = state["cls_mean"], state["cls_cov"]

    def parameter_modules(self) -> tuple[nn.Module, ...]:
        return (
            self.current,
            self.heads,
            *self.snapshots,
            *(m for m in [self.merged] if m is not None),
        )
