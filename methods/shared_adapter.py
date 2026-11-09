"""One adapter shared by all tasks, kept consistent with its closed-form classifier.

A lightweight bottleneck adapter is inserted in parallel to the MLP of each selected block of a frozen ViT.
All tasks train the same adapter (initialised from the previous task), through a temporary linear head that
is discarded afterwards. The classifier is a ROLANN ridge head built on accumulated feature statistics.
Two mechanisms keep those statistics valid while the adapter keeps changing:

* ``compensate``: after each task, the stored statistics are transported to the new feature space with the
  ridge-to-identity linear map ``P`` fitted on the current data through the previous and the new adapter;
* ``protect="distill"``: while training, the embeddings of the current batch are kept close to the ones
  produced by the adapter of the previous task.

``mode`` selects the paper's arms: ``shared`` (continual adaptation), ``frozen`` (adapter trained on the
first task only) and ``noadapt`` (frozen backbone, classifier only).
"""

from __future__ import annotations

import copy
from contextlib import contextmanager, nullcontext

import torch
from torch import nn

from models.adapter import Adapter
from utils.logging import logger
from utils.progress import epoch_progress

from .base import IncrementalMethod
from .rolann import ROLANN

MODES = {"shared", "frozen", "noadapt"}
PROTECTIONS = {"none", "distill"}


class SharedAdapter(IncrementalMethod):
    def __init__(
        self,
        backbone: nn.Module,
        learning_rate: float,
        weight_decay: float,
        momentum: float,
        epochs: int,
        adapter_bottleneck: int,
        adapter_scale: float,
        adapter_dropout: float,
        adapt_start_layer: int,
        adapt_end_layer: int,
        mode: str,
        rolann_ridge: float,
        device: torch.device,
        *,
        probe_learning_rate: float | None = None,
        adapter_layernorm: bool = False,
        adapter_weight_decay: float = 0.0,
        cosine_schedule: bool = False,
        normalize_classifier_features: bool = False,
        mask_old_probe_logits: bool = False,
        amp: bool = True,
        protect: str = "none",
        protect_lambda: float = 0.0,
        compensate: bool = False,
        compensate_ridge: float = 1.0,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {sorted(MODES)}, got {mode!r}")
        if protect not in PROTECTIONS:
            raise ValueError(f"protect must be one of {sorted(PROTECTIONS)}")
        if protect == "distill" and protect_lambda <= 0:
            raise ValueError("protect_lambda must be positive")
        if (protect != "none" or compensate) and mode != "shared":
            raise ValueError("distillation and compensation need the shared mode")

        self.backbone, self.device = backbone.to(device), device
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.learning_rate, self.weight_decay, self.momentum = (
            learning_rate,
            weight_decay,
            momentum,
        )
        self.epochs, self.mode = epochs, mode
        self.protect, self.protect_lambda = protect, protect_lambda
        self.compensate, self.compensate_ridge = compensate, compensate_ridge
        self.adapter_weight_decay = adapter_weight_decay
        self.cosine_schedule = cosine_schedule
        self.normalize_classifier_features = normalize_classifier_features
        self.mask_old_probe_logits = mask_old_probe_logits
        self.amp = amp and device.type == "cuda"
        self.probe_learning_rate = (
            learning_rate * 0.1 if probe_learning_rate is None else probe_learning_rate
        )
        self.rolann = ROLANN(rolann_ridge)
        self.probe: nn.Linear | None = None
        self._seen_classes = 0
        self._adapters_enabled = mode != "noadapt"
        self._previous: nn.ModuleList | None = None  # adapter after the last task

        self._selected_blocks = tuple(range(adapt_start_layer, adapt_end_layer + 1))
        blocks = getattr(self.backbone, "blocks", ())
        if (
            not blocks
            or min(self._selected_blocks) < 0
            or max(self._selected_blocks) >= len(blocks)
        ):
            raise ValueError("Adaptation layers must be valid ViT block indices")
        self.adapters = nn.ModuleList()
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        if mode != "noadapt":
            for block_index in self._selected_blocks:
                dim = self.backbone.blocks[block_index].norm2.normalized_shape[0]
                self.adapters.append(
                    Adapter(
                        dim,
                        adapter_bottleneck,
                        adapter_scale,
                        adapter_dropout,
                        layernorm=adapter_layernorm,
                    ).to(device)
                )
            self._install_hooks()

    # -- adapter injection -------------------------------------------------------------------

    def _install_hooks(self) -> None:
        """Adapter in parallel to the MLP: it reads the residual stream before ``norm2``."""
        for local_index, block_index in enumerate(self._selected_blocks):
            state: dict[str, torch.Tensor] = {}

            def before_norm2(module, inputs, state=state, local_index=local_index):
                if self._adapters_enabled:
                    state["parallel"] = self.adapters[local_index](inputs[0])

            def after_mlp(module, inputs, output, state=state):
                return (
                    output + state.pop("parallel") if self._adapters_enabled else output
                )

            block = self.backbone.blocks[block_index]
            self._handles.extend(
                [
                    block.norm2.register_forward_pre_hook(before_norm2),
                    block.mlp.register_forward_hook(after_mlp),
                ]
            )

    @contextmanager
    def _clean_pass(self):
        previous = self._adapters_enabled
        self._adapters_enabled = False
        try:
            yield
        finally:
            self._adapters_enabled = previous

    @contextmanager
    def _with_adapters(self, adapters: nn.ModuleList):
        current, self.adapters = self.adapters, adapters
        try:
            yield
        finally:
            self.adapters = current

    def _autocast(self):
        return (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.amp
            else nullcontext()
        )

    # -- features ----------------------------------------------------------------------------

    @staticmethod
    def _flatten(features: torch.Tensor | tuple | list) -> torch.Tensor:
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim == 3:
            features = features[:, 0]
        return features.flatten(1)

    @torch.no_grad()
    def _frozen_features(self, images: torch.Tensor) -> torch.Tensor:
        """Features of the backbone without adapters."""
        self.backbone.eval()
        with self._clean_pass():
            return self._flatten(self.backbone(images.to(self.device)))

    def _features(self, images: torch.Tensor) -> torch.Tensor:
        if self.mode == "noadapt":
            return self._frozen_features(images)
        return self._flatten(self.backbone(images.to(self.device)))

    def _classifier_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self._features(images)
        if self.normalize_classifier_features:
            features = nn.functional.normalize(features, dim=1)
        return features

    # -- classifier --------------------------------------------------------------------------

    def _expand_probe(self, feature_dim: int, seen_classes: int) -> None:
        if self.probe is not None and self.probe.out_features == seen_classes:
            return
        probe = nn.Linear(feature_dim, seen_classes, device=self.device)
        if self.probe is not None:
            with torch.no_grad():
                probe.weight[: self.probe.out_features].copy_(self.probe.weight)
                probe.bias[: self.probe.out_features].copy_(self.probe.bias)
        self.probe = probe

    @torch.no_grad()
    def _fit_rolann(self, loader) -> None:
        """Add the current task's feature statistics to the closed-form classifier."""
        self.adapters.eval()
        if self.probe is not None:
            self.probe.eval()
        for images, labels in loader:
            labels = labels.to(self.device)
            self.rolann.update(
                self._classifier_features(images), labels, labels.unique(sorted=True)
            )

    @torch.no_grad()
    def _compensate_head(self, loader) -> None:
        """Transport the stored statistics to the new features (paper Eqs. 3-4)."""
        assert self._previous is not None
        self.adapters.eval()
        old, new = [], []
        for images, _ in loader:
            now = self._features(images)
            with self._with_adapters(self._previous):
                then = self._features(images)
            if self.normalize_classifier_features:
                now = nn.functional.normalize(now, dim=1)
                then = nn.functional.normalize(then, dim=1)
            old.append(then.double())
            new.append(now.double())
        x, y = torch.cat(old), torch.cat(new)
        d = x.shape[1]
        gram = x.T @ x
        mu = self.compensate_ridge * gram.trace() / d + 1e-8
        eye = torch.eye(d, dtype=gram.dtype, device=gram.device)
        p = eye + torch.linalg.solve(gram + mu * eye, x.T @ (y - x))
        head = self.rolann

        def move(v: torch.Tensor) -> torch.Tensor:  # rows of augmented [1, x]
            v = v.clone()
            v[1:] = (p.T @ v[1:].double()).to(v.dtype)
            return v

        head.mg = [move(m.unsqueeze(1)).squeeze(1) for m in head.mg]
        head.ug, head.sg, _ = torch.linalg.svd(
            move(head.ug * head.sg.unsqueeze(0)), full_matrices=False
        )
        moments = torch.stack(head.mg)
        inverse = 1.0 / (head.sg.square().clamp_min(1e-10) + head.ridge)
        solved = head.ug @ (inverse.unsqueeze(1) * (head.ug.T @ moments.T))
        head.weights = list(solved.T.unbind(0))
        drift = float((y - x).norm(dim=1).mean())
        logger.info(f"compensation: mean feature drift {drift:.4f}")

    # -- training ----------------------------------------------------------------------------

    def _skip_sgd(self) -> bool:
        return self.mode == "noadapt" or (
            self.mode == "frozen" and bool(self._seen_classes)
        )

    def train_task(
        self,
        loader,
        task_id: int,
        seen_classes: int,
        eval_loader=None,
        prototype_loader=None,
    ) -> None:
        if task_id and seen_classes <= self._seen_classes:
            raise ValueError("seen_classes must grow for an incremental task")
        self.backbone.eval()
        statistics_loader = prototype_loader or loader
        if self._skip_sgd():
            logger.info(f"Task {task_id} {self.mode}: skipping SGD, ROLANN only")
            self._fit_rolann(statistics_loader)
            self._seen_classes = seen_classes
            return

        first_images, _ = next(iter(loader))
        self._expand_probe(self._frozen_features(first_images).shape[-1], seen_classes)
        assert self.probe is not None
        optimizer = torch.optim.SGD(
            [
                {"params": self.probe.parameters(), "lr": self.probe_learning_rate},
                {
                    "params": list(self.adapters.parameters()),
                    "lr": self.learning_rate,
                    "weight_decay": self.adapter_weight_decay,
                },
            ],
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
            if self.cosine_schedule
            else None
        )
        distill = self.protect == "distill" and self._previous is not None
        self.adapters.train()
        self.probe.train()
        task_start = self._seen_classes
        distill_sum, distill_steps = 0.0, 0
        with epoch_progress(self.epochs, task_id, stage="adapter") as progress:
            for _ in progress:
                for images, labels in loader:
                    labels = labels.to(self.device)
                    if distill:
                        with torch.no_grad(), self._with_adapters(self._previous):
                            previous = self._features(images)
                    with self._autocast():
                        features = self._features(images)
                        logits = self.probe(features)
                        if self.mask_old_probe_logits and task_start:
                            logits[:, :task_start] = float("-inf")
                        loss = nn.functional.cross_entropy(logits, labels)
                    if distill:
                        # Keep the embeddings of the current images where the previous adapter put them.
                        now = nn.functional.normalize(features.float(), dim=1)
                        then = nn.functional.normalize(previous.float(), dim=1)
                        term = (now - then).square().sum(1).mean()
                        loss = loss + self.protect_lambda * term
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if distill:
                        distill_sum += float(term.detach())
                        distill_steps += 1
                    progress.update(
                        loss.item(),
                        correct=(logits.argmax(dim=1) == labels).sum().item(),
                        samples=labels.numel(),
                    )
                if eval_loader is not None:
                    progress.set_test_acc(self._probe_accuracy(eval_loader))
                    self.adapters.train()
                    self.probe.train()
                if scheduler is not None:
                    scheduler.step()
        if distill_steps:
            logger.info(
                f"Task {task_id} distillation: mean term "
                f"{distill_sum / distill_steps:.6g} (lambda {self.protect_lambda})"
            )
        if self.compensate and self._previous is not None:
            self._compensate_head(statistics_loader)
        self._fit_rolann(statistics_loader)
        if self.protect == "distill" or self.compensate:
            self._previous = copy.deepcopy(self.adapters).eval()
            self._previous.requires_grad_(False)
        self._seen_classes = seen_classes

    @torch.no_grad()
    def _probe_accuracy(self, loader) -> float:
        assert self.probe is not None
        self.probe.eval()
        self.adapters.eval()
        correct = total = 0
        for images, labels in loader:
            labels = labels.to(self.device)
            predicted = self.probe(self._features(images)).argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.numel()
        if not total:
            raise ValueError("Probe evaluation loader has no samples")
        return 100 * correct / total

    # -- inference and state -----------------------------------------------------------------

    @torch.no_grad()
    def logits(self, images: torch.Tensor) -> torch.Tensor:
        if not self.rolann.weights:
            raise RuntimeError("Call train_task before evaluation")
        self.backbone.eval()
        self.adapters.eval()
        return self.rolann(self._classifier_features(images))

    def state_dict(self) -> dict:
        return {
            "adapters": self.adapters.state_dict(),
            "probe": None if self.probe is None else self.probe.state_dict(),
            "rolann": self.rolann.export_state(),
            "seen_classes": self._seen_classes,
        }

    def load_state_dict(self, state: dict) -> None:
        self.adapters.load_state_dict(state["adapters"])
        if state["probe"] is not None:
            self._expand_probe(
                state["probe"]["weight"].shape[1], state["probe"]["weight"].shape[0]
            )
            assert self.probe is not None
            self.probe.load_state_dict(state["probe"])
        self.rolann.import_state(state["rolann"])
        self._seen_classes = state["seen_classes"]

    def parameter_modules(self) -> tuple[nn.Module, ...]:
        modules: list[nn.Module] = [self.adapters]
        if self.probe is not None:
            modules.append(self.probe)
        return tuple(modules)
