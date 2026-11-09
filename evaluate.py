"""Re-evaluate a completed canonical run's saved method state."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean

import torch

from datasets.protocol import TaskProtocol, load_class_order
from datasets.registry import DEFAULT_DATA_ROOT, build_datasets
from models.vit import load_frozen_vit
from runner import evaluate_task, make_loader, make_method
from utils.logging import log_class_order, log_evaluation, logger
from utils.reproducibility import parse_device

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="Dataset parent directory (default: ~/datasets). CIFAR-100 and ImageNet-R layouts are auto-detected.",
    )
    parser.add_argument(
        "--checkpoint",
        help="Optional local canonical checkpoint; defaults to timm pretrained weights.",
    )
    parser.add_argument(
        "--device",
        default="0" if torch.cuda.is_available() else "cpu",
        help="GPU index (0 or 1) or cpu",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    provenance = json.loads((run_dir / "checkpoint.json").read_text())
    order, actual_manifest_hash = load_class_order(
        config["class_order_manifest"], config["num_classes"]
    )
    if actual_manifest_hash != config["class_order_manifest_sha256"]:
        raise ValueError("Class-order manifest changed since this run")
    order = config.get("class_order", order)
    checkpoint = args.checkpoint or config.get("checkpoint")
    backbone, _ = load_frozen_vit(
        checkpoint,
        model_name=config.get("model_name"),
        expected_sha256=provenance["sha256"],
        input_norm=config.get("input_norm", False),
    )
    device = parse_device(args.device)
    logger.info(f"Using device {device}")
    _, test_dataset = build_datasets(
        config["dataset"],
        args.data_root,
        preprocessing=config.get("preprocessing", "canonical"),
    )
    protocol = TaskProtocol(
        tuple(order), config["initial_classes"], config["increment_classes"]
    )
    log_class_order(protocol, config["seed"])
    method = make_method(config, backbone, device)
    state = torch.load(run_dir / "method.pt", map_location=device, weights_only=True)
    if state.get("format_version") != 1 or state.get("method_name") != config["method"]:
        raise ValueError(
            "Method checkpoint is incompatible with this run configuration"
        )
    method.load_state_dict(state["method"])

    final_task = protocol.num_tasks - 1
    accuracies = {}
    for task_id in range(protocol.num_tasks):
        accuracies[str(task_id)] = (
            evaluate_task(
                method,
                make_loader(
                    test_dataset,
                    protocol.classes_for_task(task_id),
                    protocol,
                    config,
                    False,
                    task_id,
                ),
                device,
            )
            * 100.0
        )
    overall = (
        evaluate_task(
            method,
            make_loader(
                test_dataset,
                protocol.seen_classes(final_task),
                protocol,
                config,
                False,
                final_task,
            ),
            device,
        )
        * 100.0
    )
    result = {
        "final_task": final_task,
        "accuracy_by_task": accuracies,
        "final_accuracy": mean(accuracies.values()),
        "overall_accuracy": overall,
    }
    log_evaluation(accuracies, overall_accuracy=overall)
    (run_dir / "evaluation.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
