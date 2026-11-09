"""Head-refit oracle: how much accuracy does a stale ROLANN head cost?

ROLANN stores a global SVD and per-class moments of the features seen when
each task was learned.  If adapters keep moving, old-class statistics no
longer describe the features old images get at test time.  This script keeps
the final adapter, fits a fresh ROLANN on
the training images of every task (same task order, same evaluation view the
run used), and evaluates both heads on the same test images.

    staleness = refit - actual

It reads training images of old tasks, so it is a diagnostic, not a method.
Sanity: for ``noadapt`` and ``frozen`` the features never change after task
0, so refit must equal actual up to numerics.

    python scripts/head_refit.py --run-dir runs/.../stamp [--run-dir ...] --device 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from datasets.protocol import TaskProtocol  # noqa: E402
from datasets.registry import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    build_datasets,
    evaluation_view,
)
from methods.rolann import ROLANN  # noqa: E402
from models.vit import load_frozen_vit  # noqa: E402
from runner import make_loader, make_method  # noqa: E402
from utils.reproducibility import parse_device  # noqa: E402


def _resolve(run_dir: Path) -> Path:
    if (run_dir / "method.pt").is_file():
        return run_dir
    runs = sorted(path.parent for path in run_dir.glob("*/method.pt"))
    if not runs:
        raise FileNotFoundError(f"No method.pt in {run_dir}")
    return runs[-1]


def _load(run_dir: Path, data_root: str, device: torch.device):
    config = json.loads((run_dir / "config.json").read_text())
    if config.get("method") != "shared_adapter":
        raise ValueError(f"{run_dir} is not a shared-adapter run")
    provenance = json.loads((run_dir / "checkpoint.json").read_text())
    backbone, _ = load_frozen_vit(
        config.get("checkpoint"),
        model_name=config.get("model_name"),
        expected_sha256=provenance.get("sha256"),
    )
    method = make_method(config, backbone, device)
    state = torch.load(run_dir / "method.pt", map_location=device, weights_only=True)
    method.load_state_dict(state["method"])
    train, test = build_datasets(
        config["dataset"],
        data_root,
        preprocessing=config.get("preprocessing", "canonical"),
    )
    protocol = TaskProtocol(
        tuple(config["class_order"]),
        config["initial_classes"],
        config["increment_classes"],
    )
    return method, config, protocol, train, test


@torch.no_grad()
def _per_task(method, protocol, test, config) -> tuple[list[float], float]:
    correct, total = [], []
    for task in range(protocol.num_tasks):
        loader = make_loader(
            test, protocol.classes_for_task(task), protocol, config, False, task
        )
        hits = count = 0
        for images, labels in loader:
            hits += (method.logits(images).argmax(1).cpu() == labels).sum().item()
            count += labels.numel()
        correct.append(hits)
        total.append(count)
    per_task = [100 * c / n for c, n in zip(correct, total)]
    return per_task, 100 * sum(correct) / sum(total)


@torch.no_grad()
def _refit(method, protocol, train, config) -> None:
    """Fresh ROLANN over all tasks, with the final adapter."""
    method.rolann = ROLANN(method.rolann.ridge)
    for task in range(protocol.num_tasks):
        loader = make_loader(
            evaluation_view(train),
            protocol.classes_for_task(task),
            protocol,
            config,
            False,
            task,
        )
        method._fit_rolann(loader)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results" / "head_refit.json"
    )
    args = parser.parse_args()
    device = parse_device(args.device)
    results = json.loads(args.out.read_text()) if args.out.is_file() else {}
    for run_dir in args.run_dir:
        run_dir = _resolve(run_dir)
        method, config, protocol, train, test = _load(run_dir, args.data_root, device)
        recorded = json.loads((run_dir / "metrics.json").read_text())["final_accuracy"]
        actual_tasks, actual = _per_task(method, protocol, test, config)
        _refit(method, protocol, train, config)
        refit_tasks, refit = _per_task(method, protocol, test, config)
        entry = {
            "mode": config.get("mode"),
            "recorded_final": recorded,
            "actual": actual,
            "refit": refit,
            "staleness": refit - actual,
            "actual_per_task": actual_tasks,
            "refit_per_task": refit_tasks,
        }
        results[str(run_dir)] = entry
        print(
            f"{run_dir}: recorded {recorded:.2f} actual {actual:.2f} "
            f"refit {refit:.2f} staleness {refit - actual:+.2f}",
            flush=True,
        )
        print(
            "  per-task delta:",
            " ".join(f"{r - a:+.2f}" for r, a in zip(refit_tasks, actual_tasks)),
            flush=True,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        del method
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
