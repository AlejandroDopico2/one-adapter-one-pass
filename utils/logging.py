from __future__ import annotations

import contextlib
import csv
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

import torch
from tqdm import tqdm

from datasets.protocol import RemappedSubset, TaskProtocol


class _TqdmHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        tqdm.write(self.format(record), file=sys.stderr)


def configure_logging() -> None:
    logger.handlers.clear()
    handler = _TqdmHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


logger = logging.getLogger("unified-vit-cil")
configure_logging()


def log_evaluation(
    evaluations: dict[str, float],
    *,
    overall_accuracy: float | None = None,
    plasticity: float | None = None,
    stability: float | None = None,
) -> None:
    logger.info("Start evaluation")
    for task_id, accuracy in evaluations.items():
        logger.info(f"  Task {task_id}: {accuracy:.2f}%")
    if overall_accuracy is not None:
        logger.info(f"  Overall (all seen classes): {overall_accuracy:.2f}%")
    if plasticity is not None:
        logger.info(f"  Plasticity: {plasticity:.2f}%")
    if stability is not None:
        logger.info(f"  Stability: {stability:.2f}%")


def log_class_order(protocol: TaskProtocol, seed: int) -> None:
    logger.info(f"Class order (seed={seed}): {list(protocol.class_order)}")
    for task_id in range(protocol.num_tasks):
        logger.info(f"  Task {task_id}: {list(protocol.classes_for_task(task_id))}")


def log_eval_set_sizes(dataset, protocol: TaskProtocol) -> None:
    sizes = []
    for task_id in range(protocol.num_tasks):
        count = len(
            RemappedSubset(
                dataset, protocol.classes_for_task(task_id), protocol.class_order
            )
        )
        sizes.append(count)
        logger.info(f"  Task {task_id} eval images: {count}")
    total = sum(sizes)
    if not total:
        raise ValueError("Evaluation set is empty")
    logger.info(
        f"  All-class eval images: {total} "
        f"(1 image = {100.0 / total:.3f} overall points)"
    )


class _Tee:
    def __init__(self, terminal: TextIO, log: TextIO):
        self.terminal, self.log = terminal, log

    def write(self, text: str) -> int:
        self.terminal.write(text)
        if "\r" not in text:
            self.log.write(text)
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()


class RunLogger:
    def __init__(self, root: str | Path, config: dict, provenance: dict):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = Path(root) / stamp
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self._jsonl = (self.run_dir / "events.jsonl").open("w", encoding="utf-8")
        self._csv = (self.run_dir / "tasks.csv").open("w", newline="", encoding="utf-8")
        self._output = (self.run_dir / "run.log").open("w", encoding="utf-8")
        self._writer: csv.DictWriter | None = None
        self.write_json("config.json", config)
        self.write_json("checkpoint.json", provenance)
        self.write_json("environment.json", environment_snapshot())

    @contextlib.contextmanager
    def capture_output(self):
        with (
            contextlib.redirect_stdout(_Tee(sys.stdout, self._output)),
            contextlib.redirect_stderr(_Tee(sys.stderr, self._output)),
        ):
            yield

    def write_json(self, name: str, payload: dict) -> None:
        (self.run_dir / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )

    def event(self, payload: dict) -> None:
        self._jsonl.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        self._jsonl.flush()

    def task(self, payload: dict) -> None:
        if self._writer is None:
            self._writer = csv.DictWriter(self._csv, fieldnames=sorted(payload))
            self._writer.writeheader()
        self._writer.writerow(payload)
        self._csv.flush()

    def close(self) -> None:
        self._jsonl.close()
        self._csv.close()
        self._output.close()


def environment_snapshot() -> dict[str, str | None]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_count": str(torch.cuda.device_count()),
    }


def parameter_counts(module: torch.nn.Module) -> dict[str, int]:
    return {
        "parameters_total": sum(parameter.numel() for parameter in module.parameters()),
        "parameters_trainable": sum(
            parameter.numel()
            for parameter in module.parameters()
            if parameter.requires_grad
        ),
    }
