"""Shared task loop and evaluation for all future method migrations."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from datasets.protocol import RemappedSubset, TaskProtocol
from datasets.registry import evaluation_view
from methods.ease import EASE
from methods.frozen_linear import FrozenLinear
from methods.sema import SEMA
from methods.shared_adapter import SharedAdapter
from methods.srw import SRW
from methods.tuna import TUNA
from utils.energy import measure
from utils.logging import log_evaluation, parameter_counts
from utils.metrics import AccuracyMatrix
from utils.reproducibility import worker_init_fn


def make_method(config: dict, backbone: torch.nn.Module, device: torch.device):
    common = (backbone, config["learning_rate"], config["weight_decay"])
    factories = {
        "frozen_linear": lambda: FrozenLinear(
            *common, config["momentum"], config["epochs"], device
        ),
        "ease": lambda: EASE(
            *common,
            config["momentum"],
            config["epochs"],
            device,
            adapter_dim=config["adapter_dim"],
            adapter_scale=config["adapter_scale"],
            adapter_dropout=config["adapter_dropout"],
            alpha=config["alpha"],
            beta=config["beta"],
            include_initial_features=config["include_initial_features"],
            use_reweight=config["use_reweight"],
            recalculate_similarity=config["recalculate_similarity"],
            diagonal_only=config["diagonal_only"],
            adapt_start_layer=config.get("adapt_start_layer", 0),
        ),
        "sema": lambda: SEMA(
            *common,
            config["momentum"],
            config["epochs"],
            device,
            adapter_dim=config["adapter_dim"],
            descriptor_dim=config["descriptor_dim"],
            descriptor_learning_rate=config["descriptor_learning_rate"],
            descriptor_epochs=config["descriptor_epochs"],
            expansion_threshold=config["expansion_threshold"],
            adapt_start_layer=config["adapt_start_layer"],
            adapt_end_layer=config["adapt_end_layer"],
            adapter_dropout=config["adapter_dropout"],
            record_size=config["record_size"],
            detect_batch_size=config.get("detect_batch_size", 128),
        ),
        "srw": lambda: SRW(
            *common,
            config["momentum"],
            config["epochs"],
            device,
            slots=config.get("slots", 4),
            bottleneck=config.get("adapter_bottleneck", 16),
            adapter_scale=config.get("adapter_scale", 1.0),
            adapter_dropout=config.get("adapter_dropout", 0.1),
            adapter_layernorm=config.get("adapter_layernorm", True),
            start_block=config.get("adapt_start_layer", 9),
            assignment=config.get("assignment", "semantic"),
            distill_lambda=config.get("distill_lambda", 1.0),
            ridge=config.get("rolann_ridge", 0.1),
            probe_scale=config.get("probe_scale", 16.0),
            seed=config.get("seed", 0),
            groups=(
                {p: c // 10 for p, c in enumerate(config["class_order"])}
                if config.get("assignment") == "oracle"
                else None
            ),
            compensate=config.get("compensate", False),
            compensate_ridge=config.get("compensate_ridge", 1.0),
        ),
        "tuna": lambda: TUNA(
            *common,
            config["momentum"],
            config["epochs"],
            device,
            adapter_dim=config.get("adapter_dim", 16),
            adapter_scale=config.get("adapter_scale", 0.1),
            adapter_dropout=config.get("adapter_dropout", 0.1),
            scale=config.get("scale", 20.0),
            margin=config.get("margin", 0.0),
            orth_reg=config.get("orth_reg", 0.001),
            use_orth=config.get("use_orth", True),
            ca_epochs=config.get("ca_epochs", 30),
            ca_lr=config.get("ca_lr", 0.005),
            ca_decay=config.get("ca_decay", True),
            samples_per_class=5 * config.get("batch_size", 32),
            adapt_start_layer=config.get("adapt_start_layer", 0),
        ),
        "shared_adapter": lambda: SharedAdapter(
            *common,
            config["momentum"],
            config["epochs"],
            config["adapter_bottleneck"],
            config["adapter_scale"],
            config["adapter_dropout"],
            config["adapt_start_layer"],
            config["adapt_end_layer"],
            config["mode"],
            config["rolann_ridge"],
            device,
            probe_learning_rate=config.get("probe_learning_rate"),
            adapter_layernorm=config.get("adapter_layernorm", False),
            adapter_weight_decay=config.get("adapter_weight_decay", 0.0),
            cosine_schedule=config.get("cosine_schedule", False),
            normalize_classifier_features=config.get(
                "normalize_classifier_features", False
            ),
            mask_old_probe_logits=config.get("mask_old_probe_logits", False),
            amp=config.get("amp", True),
            protect=config.get("protect", "none"),
            protect_lambda=config.get("protect_lambda", 0.0),
            compensate=config.get("compensate", False),
            compensate_ridge=config.get("compensate_ridge", 1.0),
        ),
    }
    factory = factories.get(config["method"])
    if factory is None:
        raise ValueError(f"Method is not migrated: {config['method']}")
    return factory()


def make_loader(dataset, classes, protocol, config, shuffle: bool, task_id: int):
    generator = torch.Generator().manual_seed(config["seed"] + task_id)
    return DataLoader(
        RemappedSubset(dataset, classes, protocol.class_order),
        batch_size=config["batch_size"],
        shuffle=shuffle,
        num_workers=config["workers"],
        worker_init_fn=worker_init_fn,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def evaluate_task(method, loader, device: torch.device) -> float:
    correct = total = 0
    for images, labels in loader:
        correct += (method.logits(images).argmax(dim=1).cpu() == labels).sum().item()
        total += labels.numel()
    if not total:
        raise ValueError("Evaluation task has no samples")
    return correct / total


def train_protocol(
    method,
    train_dataset,
    test_dataset,
    protocol: TaskProtocol,
    config: dict,
    device: torch.device,
    logger,
):
    matrix = AccuracyMatrix()
    for task_id in range(protocol.num_tasks):
        task_classes = protocol.classes_for_task(task_id)
        with measure(device) as cost:  # time, GPU energy, peak memory
            train_loader = make_loader(
                train_dataset,
                task_classes,
                protocol,
                config,
                shuffle=True,
                task_id=task_id,
            )
            train_args = (train_loader, task_id, len(protocol.seen_classes(task_id)))
            if isinstance(method, (EASE, TUNA, SRW)):
                method.train_task(
                    *train_args,
                    prototype_loader=make_loader(
                        evaluation_view(train_dataset),
                        task_classes,
                        protocol,
                        config,
                        shuffle=False,
                        task_id=task_id,
                    ),
                )
            elif isinstance(method, SharedAdapter):
                method.train_task(
                    *train_args,
                    eval_loader=make_loader(
                        test_dataset,
                        task_classes,
                        protocol,
                        config,
                        shuffle=False,
                        task_id=task_id,
                    ),
                    prototype_loader=make_loader(
                        evaluation_view(train_dataset),
                        task_classes,
                        protocol,
                        config,
                        shuffle=False,
                        task_id=task_id,
                    ),
                )
            else:
                method.train_task(*train_args)
        seconds = cost["seconds"]
        evaluations = {}
        for eval_task in range(task_id + 1):
            accuracy = evaluate_task(
                method,
                make_loader(
                    test_dataset,
                    protocol.classes_for_task(eval_task),
                    protocol,
                    config,
                    False,
                    eval_task,
                ),
                device,
            )
            accuracy *= 100.0
            matrix.record(task_id, eval_task, accuracy)
            evaluations[str(eval_task)] = accuracy
        overall = (
            evaluate_task(
                method,
                make_loader(
                    test_dataset,
                    protocol.seen_classes(task_id),
                    protocol,
                    config,
                    False,
                    task_id,
                ),
                device,
            )
            * 100.0
        )
        matrix.record_overall(overall)
        summary = matrix.summary()
        log_evaluation(
            evaluations,
            overall_accuracy=overall,
            plasticity=summary["plasticity"],
            stability=summary["stability"],
        )
        backbone_counts = parameter_counts(method.backbone)
        method_counts = {
            key: sum(
                parameter_counts(module)[key] for module in method.parameter_modules()
            )
            for key in ("parameters_total", "parameters_trainable")
        }
        record = {
            "task": task_id,
            "training_seconds": seconds,
            "train_energy_j": cost["energy_j"],
            "train_peak_mem_mb": cost["peak_mem_mb"],
            "gpu_processes": cost.get("gpu_processes"),
            "task_active_parameters": method_counts["parameters_trainable"],
            "cumulative_parameters_total": backbone_counts["parameters_total"]
            + method_counts["parameters_total"],
            "cumulative_parameters_trainable": method_counts["parameters_trainable"],
            "evaluations": evaluations,
            "overall_accuracy": overall,
        }
        task_metrics = getattr(method, "task_metrics", None)
        if callable(task_metrics):
            record.update(task_metrics())
        logger.event({"event": "task_complete", **record})
        logger.task(
            {key: value for key, value in record.items() if key != "evaluations"}
        )
    return matrix
