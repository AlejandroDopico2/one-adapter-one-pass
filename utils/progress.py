"""Compact terminal progress for training loops."""

from __future__ import annotations

import sys
from collections.abc import Iterator

from tqdm import tqdm

from utils.logging import logger


class EpochProgress:
    def __init__(self, epochs: int, task_id: int, stage: str = ""):
        encoding = getattr(sys.stderr, "encoding", None) or ""
        ascii_only = not encoding.lower().startswith(("utf", "iso"))
        self._task_id = task_id
        self._stage = stage
        self._epoch = 0
        label = f"Task {task_id}"
        if stage:
            label = f"{label} {stage}"
        self._bar = tqdm(
            range(epochs),
            desc=label,
            unit="epoch",
            dynamic_ncols=True,
            ascii=ascii_only,
            leave=False,
            file=sys.stderr,
        )
        self._loss = 0.0
        self._samples = 0
        self._correct = 0
        self._diversity: float | None = None
        self._router: float | None = None
        self._test_acc: float | None = None

    def __iter__(self) -> Iterator[int]:
        for epoch in self._bar:
            self._epoch = epoch
            self._loss = self._samples = self._correct = 0
            self._diversity = None
            self._router = None
            self._test_acc = None
            yield epoch
            self._finish_epoch()

    def update(
        self,
        loss: float,
        *,
        correct: int = 0,
        samples: int = 0,
        diversity: float | None = None,
        router: float | None = None,
    ) -> None:
        self._loss += loss * samples if samples else loss
        self._samples += samples
        self._correct += correct
        if diversity is not None:
            self._diversity = (self._diversity or 0.0) + (
                diversity * samples if samples else diversity
            )
        if router is not None:
            self._router = (self._router or 0.0) + (
                router * samples if samples else router
            )
        self._bar.set_postfix(self._postfix())

    def set_test_acc(self, accuracy: float) -> None:
        self._test_acc = accuracy
        self._bar.set_postfix(self._postfix())

    def _postfix(self) -> dict[str, float | str]:
        postfix: dict[str, float | str] = {"loss": self._loss / max(self._samples, 1)}
        if self._samples:
            postfix["acc"] = f"{100 * self._correct / self._samples:.1f}%"
        if self._diversity is not None:
            postfix["div"] = self._diversity / max(self._samples, 1)
        if self._router is not None:
            postfix["route"] = self._router / max(self._samples, 1)
        if self._test_acc is not None:
            postfix["test"] = f"{self._test_acc:.1f}%"
        return postfix

    def _label(self) -> str:
        label = f"Task {self._task_id}"
        if self._stage:
            label = f"{label} {self._stage}"
        return f"{label}, epoch {self._epoch + 1}"

    def _finish_epoch(self) -> None:
        average_loss = self._loss / max(self._samples, 1)
        summary = f"{self._label()}: loss={average_loss:.4f}"
        if self._samples:
            summary += f" train_acc={100 * self._correct / self._samples:.1f}%"
        if self._diversity is not None:
            summary += f" div={self._diversity / max(self._samples, 1):.4f}"
        if self._router is not None:
            summary += f" route={self._router / max(self._samples, 1):.4f}"
        if self._test_acc is not None:
            summary += f" test_acc={self._test_acc:.1f}%"
        logger.info(summary)

    def close(self) -> None:
        self._bar.close()

    def __enter__(self) -> "EpochProgress":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def epoch_progress(epochs: int, task_id: int, stage: str = "") -> EpochProgress:
    return EpochProgress(epochs, task_id, stage)
