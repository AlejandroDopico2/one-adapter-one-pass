"""SRW: semantically routed slots composed by an analytic head (H9).

The incremental version of T4.
* A frozen ViT trunk (blocks before ``start_block``) is computed once per image.
* M slot adapters (AdaptFormer) live in the blocks from ``start_block`` on.
  Slot m's feature f_m is the rest of the ViT with only slot m active.
* Every class gets one slot a(c), from its frozen-CLS class mean:
  - semantic: greedy, most-similar-first, under a balanced capacity, to
    centroids from a cosine k-means of the first task's classes;
  - random: balanced at random (control).
* Training routes reads and writes: an image of class c only goes through slot
  a(c).  Loss: a cosine probe over seen classes (old logits masked), plus
  optional distillation of f_m on current images towards the slot as it was
  before the task, which protects the slot's old classes.
* Reads are composed by the head with no router: ROLANN on
  z = [phi0, f_1 - phi0, ..., f_M - phi0] / ||phi0||, updated once per task.  Writing slot m can only move old classes through block m of z.
"""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F

from models.adapter import Adapter
from utils.logging import logger
from utils.progress import epoch_progress

from .base import IncrementalMethod
from .rolann import ROLANN


class SRW(IncrementalMethod):
    def __init__(
        self,
        backbone: nn.Module,
        learning_rate: float,
        weight_decay: float,
        momentum: float,
        epochs: int,
        device: torch.device,
        *,
        slots: int = 4,
        bottleneck: int = 16,
        adapter_scale: float = 1.0,
        adapter_dropout: float = 0.1,
        adapter_layernorm: bool = True,
        start_block: int = 9,
        assignment: str = "semantic",
        distill_lambda: float = 1.0,
        ridge: float = 0.1,
        probe_scale: float = 16.0,
        seed: int = 0,
        groups: dict[int, int] | None = None,
        compensate: bool = False,
        compensate_ridge: float = 1.0,
    ) -> None:
        if assignment not in {"semantic", "random", "oracle"}:
            raise ValueError(f"Unknown SRW assignment: {assignment}")
        self.backbone, self.device = backbone.to(device).eval(), device
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.learning_rate, self.weight_decay = learning_rate, weight_decay
        self.momentum, self.epochs = momentum, epochs
        self.count, self.start = slots, start_block
        self.assignment, self.distill_lambda = assignment, distill_lambda
        self.probe_scale, self.seed = probe_scale, seed
        # oracle: slot = a known latent group (VTAB domain), to test the
        # mechanism (write concentration) apart from the assignment heuristic.
        self.groups = groups or {}
        # H11: move the head's stored statistics to the new slot features with
        # a per-slot linear map fitted on current images (no replay).
        self.compensate, self.compensate_ridge = compensate, compensate_ridge
        blocks = self.backbone.blocks
        dim = blocks[0].norm2.normalized_shape[0]
        self.dim = dim
        self.slots = nn.ModuleList(
            nn.ModuleList(
                Adapter(
                    dim, bottleneck, adapter_scale, adapter_dropout, adapter_layernorm
                )
                for _ in blocks[start_block:]
            )
            for _ in range(slots)
        ).to(device)
        self._active: nn.ModuleList | None = None
        self._inputs: dict[int, torch.Tensor] = {}
        for local, block in enumerate(blocks[start_block:]):
            block.norm2.register_forward_pre_hook(
                lambda _m, inputs, local=local: self._inputs.__setitem__(
                    local, inputs[0]
                )
            )
            block.mlp.register_forward_hook(
                lambda _m, _i, output, local=local: (
                    output
                    if self._active is None
                    else output + self._active[local](self._inputs[local])
                )
            )
        self.rolann = ROLANN(ridge)
        self.probe: nn.Linear | None = None
        self.slot_of: dict[int, int] = {}
        self.centroids: torch.Tensor | None = None
        self.members = torch.zeros(slots)
        self.seen = 0

    # -- forward pieces --------------------------------------------------------
    def _trunk(self, images: torch.Tensor) -> torch.Tensor:
        bb = self.backbone
        images = images.to(self.device)
        for hook in bb._forward_pre_hooks.values():  # e.g. input normalisation
            images = hook(bb, (images,))[0]
        x = bb.patch_embed(images)
        x = bb._pos_embed(x)
        x = bb.patch_drop(x)
        x = bb.norm_pre(x)
        for block in bb.blocks[: self.start]:
            x = block(x)
        return x

    def _tail(self, hidden: torch.Tensor, bank: nn.ModuleList | None) -> torch.Tensor:
        self._active = bank
        try:
            x = hidden
            for block in self.backbone.blocks[self.start :]:
                x = block(x)
            x = self.backbone.norm(x)
            return self.backbone.forward_head(x).flatten(1)
        finally:
            self._active = None

    @torch.no_grad()
    def _parts(self, images: torch.Tensor):
        hidden = self._trunk(images)
        phi0 = self._tail(hidden, None).float()
        return phi0, [self._tail(hidden, bank).float() for bank in self.slots]

    def _z(self, phi0: torch.Tensor, slot_feats: list[torch.Tensor]) -> torch.Tensor:
        # One per-image scale for every block (the frozen feature's norm): a
        # scale frozen when a slot was barely trained would blow up its later
        # deltas and make the head's stored statistics stale.
        norm = phi0.norm(dim=1, keepdim=True).clamp_min(1e-6)
        return torch.cat([phi0, *(f - phi0 for f in slot_feats)], dim=1) / norm

    # -- slot assignment -------------------------------------------------------
    @torch.no_grad()
    def _assign(self, loader) -> None:
        sums, counts = {}, {}
        for images, labels in loader:
            phi0 = F.normalize(
                self._tail(self._trunk(images), None).float(), dim=1
            ).cpu()
            for label in labels.unique().tolist():
                if label in self.slot_of:
                    continue
                rows = phi0[labels == label]
                sums[label] = sums.get(label, 0) + rows.sum(0)
                counts[label] = counts.get(label, 0) + len(rows)
        new = sorted(sums)
        if not new:
            return
        means = F.normalize(torch.stack([sums[c] / counts[c] for c in new]), dim=1)
        generator = torch.Generator().manual_seed(self.seed + len(self.slot_of))
        load = [list(self.slot_of.values()).count(m) for m in range(self.count)]
        if self.assignment == "oracle":
            for c in new:
                self.slot_of[c] = self.groups[c] % self.count
                load[self.slot_of[c]] += 1
        elif self.assignment == "random":
            for index in torch.randperm(len(new), generator=generator).tolist():
                slot = min(range(self.count), key=lambda m: (load[m], m))
                self.slot_of[new[index]] = slot
                load[slot] += 1
        else:
            if self.centroids is None:
                take = torch.randperm(len(new), generator=generator)[: self.count]
                centers = means[take]
                for _ in range(50):
                    assign = (means @ centers.T).argmax(1)
                    centers = F.normalize(
                        torch.stack(
                            [
                                (
                                    means[assign == m].mean(0)
                                    if (assign == m).any()
                                    else centers[m]
                                )
                                for m in range(self.count)
                            ]
                        ),
                        dim=1,
                    )
                self.centroids = centers
            capacity = -(-(len(self.slot_of) + len(new)) // self.count)
            sims = means @ self.centroids.T
            pairs = sorted(
                (
                    (float(sims[i, m]), i, m)
                    for i in range(len(new))
                    for m in range(self.count)
                ),
                reverse=True,
            )
            placed: set[int] = set()
            for _, i, m in pairs:
                if i in placed or load[m] >= capacity:
                    continue
                self.slot_of[new[i]] = m
                placed.add(i)
                load[m] += 1
                n = self.members[m]
                self.centroids[m] = F.normalize(
                    (self.centroids[m] * n + means[i]) / (n + 1), dim=0
                )
                self.members[m] += 1
        logger.info(f"SRW ({self.assignment}) slot sizes {load}")
        logger.info(f"SRW slot map {dict(sorted(self.slot_of.items()))}")

    # -- training ----------------------------------------------------------------
    def train_task(
        self, loader, task_id: int, seen_classes: int, *, prototype_loader=None
    ) -> None:
        stats_loader = prototype_loader or loader
        self._assign(stats_loader)
        known = self.seen
        probe = nn.Linear(self.dim, seen_classes, bias=False).to(self.device)
        if self.probe is not None:
            with torch.no_grad():
                probe.weight[:known].copy_(self.probe.weight)
        self.probe = probe
        snapshot = (
            copy.deepcopy(self.slots).eval()
            if (self.distill_lambda or self.compensate) and task_id > 0
            else None
        )
        previous = snapshot if self.distill_lambda else None
        optimizer = torch.optim.SGD(
            [*self.slots.parameters(), *self.probe.parameters()],
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)
        lookup = torch.full((seen_classes,), -1, dtype=torch.long)
        for c, m in self.slot_of.items():
            if c < seen_classes:
                lookup[c] = m
        self.slots.train()
        with epoch_progress(self.epochs, task_id, stage="srw") as progress:
            for _ in progress:
                for images, labels in loader:
                    with torch.no_grad():
                        hidden = self._trunk(images)
                    labels = labels.to(self.device)
                    route = lookup.to(self.device)[labels]
                    loss = torch.zeros((), device=self.device)
                    correct = 0
                    for m in route.unique().tolist():
                        rows = route == m
                        feats = self._tail(hidden[rows], self.slots[m])
                        logits = self.probe_scale * F.linear(
                            F.normalize(feats, dim=1),
                            F.normalize(self.probe.weight, dim=1),
                        )
                        if known:
                            logits[:, :known] = float("-inf")
                        share = rows.float().mean()
                        loss = loss + share * F.cross_entropy(logits, labels[rows])
                        if previous is not None:
                            with torch.no_grad():
                                then = self._tail(hidden[rows], previous[m])
                            drift = (
                                (F.normalize(feats, dim=1) - F.normalize(then, dim=1))
                                .square()
                                .sum(1)
                                .mean()
                            )
                            loss = loss + share * self.distill_lambda * drift
                        correct += int((logits.argmax(1) == labels[rows]).sum())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    progress.update(
                        loss.item(), correct=correct, samples=labels.numel()
                    )
                scheduler.step()
        self.slots.eval()
        if self.compensate and snapshot is not None:
            self._compensate(stats_loader, snapshot)
        self._fit_head(stats_loader, known, seen_classes)
        self.seen = seen_classes

    @torch.no_grad()
    def _compensate(self, loader, snapshot: nn.ModuleList) -> None:
        """Ridge-to-identity map old -> new slot deltas; apply to ROLANN state."""
        old, new = [[] for _ in range(self.count)], [[] for _ in range(self.count)]
        for images, _ in loader:
            hidden = self._trunk(images)
            phi0 = self._tail(hidden, None).float()
            norm = phi0.norm(dim=1, keepdim=True).clamp_min(1e-6)
            for m in range(self.count):
                old[m].append((self._tail(hidden, snapshot[m]).float() - phi0) / norm)
                new[m].append((self._tail(hidden, self.slots[m]).float() - phi0) / norm)
        head, d = self.rolann, self.dim
        maps = []
        for m in range(self.count):
            x, y = torch.cat(old[m]).double(), torch.cat(new[m]).double()
            if torch.allclose(x, y):
                maps.append(None)
                continue
            gram = x.T @ x
            mu = self.compensate_ridge * gram.trace() / d + 1e-8
            eye = torch.eye(d, dtype=gram.dtype, device=gram.device)
            maps.append(eye + torch.linalg.solve(gram + mu * eye, x.T @ (y - x)))

        def move(v: torch.Tensor) -> torch.Tensor:  # rows of augmented [1, z]
            v = v.clone()
            for m, w in enumerate(maps):
                if w is not None:
                    block = slice(1 + d * (m + 1), 1 + d * (m + 2))
                    v[block] = (w.T @ v[block].double()).to(v.dtype)
            return v

        head.mg = [move(moment.unsqueeze(1)).squeeze(1) for moment in head.mg]
        head.ug, head.sg, _ = torch.linalg.svd(
            move(head.ug * head.sg.unsqueeze(0)), full_matrices=False
        )
        moments = torch.stack(head.mg)
        inverse = 1.0 / (head.sg.square().clamp_min(1e-10) + head.ridge)
        solved = head.ug @ (inverse.unsqueeze(1) * (head.ug.T @ moments.T))
        head.weights = list(solved.T.unbind(0))
        moved = sum(w is not None for w in maps)
        logger.info(f"SRW compensation: {moved}/{self.count} slots moved")

    @torch.no_grad()
    def _fit_head(self, loader, known: int, seen_classes: int) -> None:
        phis, feats, labels = [], [], []
        for images, targets in loader:
            phi0, slot_feats = self._parts(images)
            phis.append(phi0)
            feats.append(torch.stack(slot_feats, 1))
            labels.append(targets.to(self.device))
        phi0, slot_feats, labels = torch.cat(phis), torch.cat(feats), torch.cat(labels)
        z = self._z(phi0, list(slot_feats.unbind(1)))
        self.rolann.update(z, labels, torch.arange(known, seen_classes).to(self.device))

    # -- inference -------------------------------------------------------------
    @torch.no_grad()
    def logits(self, images: torch.Tensor) -> torch.Tensor:
        phi0, slot_feats = self._parts(images)
        return self.rolann(self._z(phi0, slot_feats))[:, : self.seen]

    def state_dict(self) -> dict:
        return {
            "slots": self.slots.state_dict(),
            "probe": None if self.probe is None else self.probe.state_dict(),
            "rolann": self.rolann.export_state(),
            "slot_of": self.slot_of,
            "seen": self.seen,
        }

    def load_state_dict(self, state: dict) -> None:
        self.slots.load_state_dict(state["slots"])
        self.rolann.import_state(state["rolann"])
        self.slot_of = dict(state["slot_of"])
        self.seen = state["seen"]

    def parameter_modules(self) -> tuple[nn.Module, ...]:
        return (self.slots,)
