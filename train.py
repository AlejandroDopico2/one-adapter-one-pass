"""Train a canonical shared-protocol CIL run."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import yaml

from datasets.protocol import (
    TaskProtocol,
    concentration_class_order,
    load_class_order,
    shuffle_class_order,
)
from datasets.registry import DEFAULT_DATA_ROOT, build_datasets, holdout_split
from models.vit import load_frozen_vit
from runner import make_method, train_protocol
from utils.energy import start_carbon, stop_carbon
from utils.logging import RunLogger, log_class_order, log_eval_set_sizes, logger
from utils.reproducibility import parse_device, seed_everything

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parent


def read_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--method", dest="method_config", required=True)
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="Dataset parent directory (default: ~/datasets). CIFAR-100 and ImageNet-R layouts are auto-detected.",
    )
    parser.add_argument(
        "--checkpoint",
        help="Optional local canonical checkpoint; defaults to timm pretrained weights.",
    )
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--checkpoint-source-url")
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument(
        "--device",
        default="0" if torch.cuda.is_available() else "cpu",
        help="GPU index (0 or 1) or cpu",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow torchvision to download CIFAR-100.",
    )
    args = parser.parse_args()

    config = {
        **read_config(args.benchmark),
        **read_config(args.method_config),
        **vars(args),
    }
    manifest_path = ROOT / config["class_order_manifest"]
    order, manifest_sha256 = load_class_order(manifest_path, config["num_classes"])
    config["class_order_manifest"] = str(manifest_path)
    config["class_order_manifest_sha256"] = manifest_sha256
    if config.get("concentration_k"):
        order = concentration_class_order(
            order,
            config["seed"],
            config["concentration_k"],
            config.get("concentration_group", 10),
        )
    else:
        order = shuffle_class_order(order, config["seed"], config.get("shuffle_blocks"))
    config["class_order"] = order
    seed_everything(config["seed"])
    device = parse_device(config["device"])
    config["device"] = str(device)
    logger.info(f"Using device {device}")
    backbone, provenance = load_frozen_vit(
        config["checkpoint"],
        model_name=config.get("model_name"),
        expected_sha256=config["checkpoint_sha256"],
        source_url=config["checkpoint_source_url"],
        input_norm=config.get("input_norm", False),
    )
    train_dataset, test_dataset = build_datasets(
        config["dataset"],
        config["data_root"],
        config["download"],
        config.get("preprocessing", "canonical"),
    )
    if config.get("validation_fraction"):  # tuning: evaluate on held-out train images
        train_dataset, test_dataset = holdout_split(
            train_dataset,
            config["validation_fraction"],
            config.get("validation_seed", 0),
        )
        logger.info(
            f"Validation split: {len(train_dataset)} train / {len(test_dataset)} held out"
        )
    protocol = TaskProtocol(
        tuple(order), config["initial_classes"], config["increment_classes"]
    )
    config["n_tasks"] = protocol.num_tasks
    run_logger = RunLogger(config["output_root"], config, provenance.as_dict())
    try:
        with run_logger.capture_output():
            log_class_order(protocol, config["seed"])
            log_eval_set_sizes(test_dataset, protocol)
            method = make_method(config, backbone, device)
            carbon = start_carbon(
                run_logger.run_dir, device, config.get("codecarbon", True)
            )
            matrix = train_protocol(
                method,
                train_dataset,
                test_dataset,
                protocol,
                config,
                device,
                run_logger,
            )
            metrics = matrix.as_dict()
            metrics["codecarbon"] = stop_carbon(carbon)
            run_logger.write_json("metrics.json", metrics)
            if hasattr(method, "compositions"):
                run_logger.write_json("compositions.json", method.compositions())
            if hasattr(method, "drift"):
                run_logger.write_json("drift.json", method.drift())
            torch.save(
                {
                    "format_version": 1,
                    "method_name": config["method"],
                    "method": method.state_dict(),
                },
                run_logger.run_dir / "method.pt",
            )
            run_logger.event({"event": "run_complete", **metrics})
            logger.info(f"Run complete  dir={run_logger.run_dir}")
    finally:
        run_logger.close()


if __name__ == "__main__":
    main()
