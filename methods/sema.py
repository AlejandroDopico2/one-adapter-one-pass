"""SEMA migration, adapted from Wang et al. (CVPR 2025).

Source: https://github.com/Huiyi-Wang/SEMA-CL (``backbone/sema_block.py``).
The canonical ViT remains the sole frozen backbone; this module attaches
functional adapter mixtures to its existing transformer blocks.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from models.adapter import Adapter
from utils.logging import logger
from utils.progress import epoch_progress

from .base import IncrementalMethod


class RepresentationDescriptor(nn.Module):
    """Autoencoder and task-zero loss statistics used for expansion."""

    def __init__(self, dimension: int, bottleneck: int, record_size: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(dimension, bottleneck)
        self.decoder = nn.Linear(bottleneck, dimension)
        self.register_buffer("records", torch.zeros(record_size))
        self.register_buffer("record_count", torch.zeros((), dtype=torch.long))

    def errors(self, tokens: torch.Tensor) -> torch.Tensor:
        features = tokens.mean(dim=1) if tokens.ndim == 3 else tokens
        return (self.decoder(self.encoder(features)) - features).square().mean(dim=-1)

    @torch.no_grad()
    def record(self, errors: torch.Tensor) -> None:
        values = errors.detach().flatten().to(self.records)
        if not values.numel():
            return
        current = self.records[: self.record_count]
        combined = torch.cat((current, values))[-self.records.numel() :]
        self.records[: combined.numel()] = combined
        self.record_count.fill_(combined.numel())

    def z_scores(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.record_count.item() < 3:
            return torch.zeros(tokens.shape[0], device=tokens.device)
        reference = self.records[: self.record_count].to(tokens)
        return (
            self.errors(tokens) - reference.mean()
        ).abs() / reference.std().clamp_min(1e-6)


class AdapterMixture(nn.Module):
    """Expandable functional adapters plus their descriptors and router."""

    def __init__(
        self,
        dimension: int,
        adapter_dim: int,
        descriptor_dim: int,
        dropout: float,
        record_size: int,
        expandable: bool,
    ) -> None:
        super().__init__()
        self.dimension, self.adapter_dim, self.descriptor_dim = (
            dimension,
            adapter_dim,
            descriptor_dim,
        )
        self.dropout, self.record_size, self.expandable = (
            dropout,
            record_size,
            expandable,
        )
        self.functional = nn.ModuleList()
        self.descriptors = nn.ModuleList()
        self.router = nn.Linear(dimension, 1)
        self.functional.append(Adapter(dimension, adapter_dim, dropout=dropout))
        if expandable:
            self.descriptors.append(
                RepresentationDescriptor(dimension, descriptor_dim, record_size)
            )

    def add(self) -> None:
        if not self.expandable:
            raise RuntimeError("This SEMA layer is not expandable")
        device, dtype = self.router.weight.device, self.router.weight.dtype
        self.functional.append(
            Adapter(self.dimension, self.adapter_dim, dropout=self.dropout).to(
                device=device, dtype=dtype
            )
        )
        self.descriptors.append(
            RepresentationDescriptor(
                self.dimension, self.descriptor_dim, self.record_size
            ).to(device=device, dtype=dtype)
        )
        expanded = nn.Linear(
            self.dimension, len(self.functional), device=device, dtype=dtype
        )
        with torch.no_grad():
            if self.router.out_features:
                expanded.weight[:-1].copy_(self.router.weight)
                expanded.bias[:-1].copy_(self.router.bias)
        self.router = expanded

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        weights = self.router(
            tokens.mean(dim=1) if tokens.ndim == 3 else tokens
        ).softmax(dim=-1)
        outputs = torch.stack([adapter(tokens) for adapter in self.functional])
        return (
            outputs
            * weights.transpose(0, 1).reshape(
                -1, tokens.shape[0], *([1] * (tokens.ndim - 1))
            )
        ).sum(dim=0)

    def should_expand(self, tokens: torch.Tensor, threshold: float) -> bool:
        if not self.descriptors:
            return False
        scores = torch.stack(
            [descriptor.z_scores(tokens) for descriptor in self.descriptors]
        )
        return bool(scores.mean(dim=1).gt(threshold).all())


class SEMA(IncrementalMethod):
    """Self-expanding adapter mixtures over an immutable canonical ViT."""

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
        descriptor_dim: int = 128,
        descriptor_learning_rate: float = 0.01,
        descriptor_epochs: int = 20,
        expansion_threshold: float = 2.0,
        adapt_start_layer: int = 9,
        adapt_end_layer: int = 11,
        adapter_dropout: float = 0.1,
        record_size: int = 500,
        detect_batch_size: int = 128,
    ) -> None:
        self.backbone, self.device = backbone.to(device), device
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.learning_rate, self.weight_decay, self.momentum, self.epochs = (
            learning_rate,
            weight_decay,
            momentum,
            epochs,
        )
        self.descriptor_learning_rate, self.descriptor_epochs = (
            descriptor_learning_rate,
            descriptor_epochs,
        )
        self.expansion_threshold = expansion_threshold
        self.detect_batch_size = detect_batch_size
        self.classifier: nn.Linear | None = None
        blocks = list(getattr(self.backbone, "blocks", ()))
        self.adapters = nn.ModuleDict()
        self._block_inputs: dict[str, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        for index, block in enumerate(blocks):
            normalized = getattr(
                getattr(block, "norm2", None), "normalized_shape", None
            )
            dimension = (
                int(normalized[0])
                if normalized
                else getattr(self.backbone, "num_features", None)
            )
            if not dimension:
                raise ValueError(
                    "SEMA requires ViT blocks with norm2.normalized_shape or backbone.num_features"
                )
            bank = AdapterMixture(
                dimension,
                adapter_dim,
                descriptor_dim,
                adapter_dropout,
                record_size,
                adapt_start_layer <= index <= adapt_end_layer,
            ).to(device)
            key = str(index)
            self.adapters[key] = bank
            self._hooks.extend(
                (
                    block.norm2.register_forward_pre_hook(
                        lambda _module, inputs, key=key: self._capture_input(
                            key, inputs[0]
                        )
                    ),
                    block.mlp.register_forward_hook(
                        lambda _module, _inputs, output, key=key: (
                            output + self.adapters[key](self._block_inputs[key])
                        )
                    ),
                )
            )
        if not self.adapters:
            raise ValueError("SEMA requires transformer blocks in backbone.blocks")

    def _capture_input(self, key: str, tokens: torch.Tensor) -> None:
        self._block_inputs[key] = tokens

    def _features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images.to(self.device))
        if isinstance(features, dict):
            features = features.get("features", features.get("pre_logits"))
        if isinstance(features, (tuple, list)):
            features = features[0]
        if not isinstance(features, torch.Tensor):
            raise TypeError("Backbone must return tensor features")
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

    def _batches(self, loader) -> Iterable[torch.Tensor]:
        for images, _ in loader:
            yield images.to(self.device)

    def _first_outlier(self, images: torch.Tensor) -> str | None:
        """Lowest block whose descriptors all flag this batch.

        The official forward stops at the first such layer, so deeper layers
        are scored only after it has been expanded and trained.
        """
        self._features(images)
        for key, bank in self.adapters.items():
            if bank.should_expand(self._block_inputs[key], self.expansion_threshold):
                return key
        return None

    def train_task(self, loader, task_id: int, seen_classes: int) -> None:
        task_start = 0 if self.classifier is None else self.classifier.out_features
        first_images, _ = next(iter(loader))
        self._expand_classifier(self._features(first_images).shape[1], seen_classes)
        if task_id == 0:
            self._train_round(loader, task_id, task_start, set(self.adapters))
            return
        # Official _detect_outlier: scan shuffled batches; each trigger adds one
        # adapter, trains it (functional + descriptor) and freezes it, then the
        # scan goes on; rescan from scratch until a full pass adds nothing.
        detect = torch.utils.data.DataLoader(
            loader.dataset,
            batch_size=self.detect_batch_size,
            shuffle=True,
            num_workers=getattr(loader, "num_workers", 0),
        )
        expansions = 0
        while True:
            added = False
            for images in self._batches(detect):
                self.backbone.eval()
                self.adapters.eval()
                with torch.no_grad():
                    key = self._first_outlier(images)
                if key is None:
                    continue
                self.adapters[key].add()
                logger.info(f"SEMA expanded adapter mixture at block {key}")
                self._train_round(loader, task_id, task_start, {key})
                added, expansions = True, expansions + 1
            if not added:
                break
        if not expansions:
            self._train_round(loader, task_id, task_start, set())

    def _train_round(self, loader, task_id: int, task_start: int, keys: set) -> None:
        """Train the head plus the newest adapter and router of ``keys``."""
        assert self.classifier is not None
        trainable = list(self.classifier.parameters())
        router_history = {}
        for key in keys:
            trainable += list(self.adapters[key].functional[-1].parameters()) + list(
                self.adapters[key].router.parameters()
            )
            router_history[key] = (
                self.adapters[key].router.weight[:-1].detach().clone(),
                self.adapters[key].router.bias[:-1].detach().clone(),
            )
        optimizer = torch.optim.SGD(
            trainable,
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )
        self.backbone.eval()
        self.adapters.train()
        with epoch_progress(self.epochs, task_id) as progress:
            for _ in progress:
                for images, labels in loader:
                    labels = labels.to(self.device)
                    logits = self.classifier(self._features(images))
                    if task_id:
                        logits[:, :task_start] = float("-inf")
                    loss = nn.functional.cross_entropy(logits, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        for key, (weight, bias) in router_history.items():
                            self.adapters[key].router.weight[:-1].copy_(weight)
                            self.adapters[key].router.bias[:-1].copy_(bias)
                    progress.update(
                        loss.item(),
                        correct=(logits.argmax(dim=1) == labels).sum().item(),
                        samples=labels.numel(),
                    )
                scheduler.step()
        descriptor_parameters = [
            parameter
            for key in keys
            if self.adapters[key].descriptors
            for parameter in self.adapters[key].descriptors[-1].parameters()
        ]
        if descriptor_parameters:
            optimizer = torch.optim.SGD(
                descriptor_parameters,
                lr=self.descriptor_learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.descriptor_epochs
            )
            with epoch_progress(
                self.descriptor_epochs, task_id, stage="descriptor"
            ) as progress:
                for _ in progress:
                    for images in self._batches(loader):
                        self._features(images)
                        loss = sum(
                            self.adapters[key].descriptors[-1].errors(tokens).mean()
                            for key, tokens in self._block_inputs.items()
                            if key in keys and self.adapters[key].descriptors
                        )
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        # Source records reconstruction errors during RD training,
                        # not after convergence.  Post-fit errors are too tight for
                        # the z-score threshold, so expansion never fired.
                        with torch.no_grad():
                            for key, tokens in self._block_inputs.items():
                                if key in keys and self.adapters[key].descriptors:
                                    self.adapters[key].descriptors[-1].record(
                                        self.adapters[key]
                                        .descriptors[-1]
                                        .errors(tokens)
                                    )
                        progress.update(loss.item())
                    scheduler.step()
        self.adapters.requires_grad_(False)

    def logits(self, images: torch.Tensor) -> torch.Tensor:
        if self.classifier is None:
            raise RuntimeError("Call train_task before evaluation")
        self.backbone.eval()
        self.adapters.eval()
        self.classifier.eval()
        with torch.no_grad():
            return self.classifier(self._features(images))

    def state_dict(self) -> dict:
        return {
            "adapters": self.adapters.state_dict(),
            "adapter_counts": {
                key: len(bank.functional) for key, bank in self.adapters.items()
            },
            "classifier": (
                None if self.classifier is None else self.classifier.state_dict()
            ),
        }

    def parameter_modules(self) -> tuple[nn.Module, ...]:
        return (
            (self.adapters,)
            if self.classifier is None
            else (self.adapters, self.classifier)
        )

    def load_state_dict(self, state: dict) -> None:
        for key, count in state.get("adapter_counts", {}).items():
            while len(self.adapters[key].functional) < count:
                self.adapters[key].add()
        self.adapters.load_state_dict(state["adapters"])
        self.adapters.requires_grad_(False)
        classifier = state.get("classifier")
        if classifier:
            self._expand_classifier(
                classifier["weight"].shape[1], classifier["weight"].shape[0]
            )
            assert self.classifier is not None
            self.classifier.load_state_dict(classifier)
