import csv
import tempfile
import unittest
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import TensorDataset

from datasets.protocol import TaskProtocol
from runner import make_method, train_protocol
from utils.logging import RunLogger

ROOT = Path(__file__).resolve().parents[1]


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(2)
        self.norm2 = nn.LayerNorm(2)
        self.mlp = nn.Linear(2, 2)

    def forward(self, inputs):
        return self.mlp(self.norm2(self.norm1(inputs)))


class _ToyViT(nn.Module):
    embed_dim = 2
    num_features = 2

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(_Block() for _ in range(12))

    def forward(self, inputs):
        for block in self.blocks:
            inputs = block(inputs)
        return inputs


class RunnerIntegrationTests(unittest.TestCase):
    def test_every_method_config_is_constructible(self):
        benchmark = yaml.safe_load(
            (ROOT / "configs/benchmarks/cifar100.yaml").read_text()
        )
        expected = {
            "frozen_linear": "FrozenLinear",
            "ours": "SharedAdapter",
            "ablation_none": "SharedAdapter",
            "ablation_comp_only": "SharedAdapter",
            "ablation_distill_only": "SharedAdapter",
            "frozen_after_task1": "SharedAdapter",
            "no_adapter": "SharedAdapter",
            "domain_isolated_oracle": "SRW",
            "domain_isolated_oracle_comp": "SRW",
        }
        for method_name, class_name in expected.items():
            with self.subTest(method=method_name):
                method_config = yaml.safe_load(
                    (ROOT / f"configs/methods/{method_name}.yaml").read_text()
                )
                method = make_method(
                    {
                        **benchmark,
                        **method_config,
                        "n_tasks": 10,
                        "class_order": list(range(benchmark["num_classes"])),
                    },
                    _ToyViT(),
                    torch.device("cpu"),
                )
                self.assertEqual(type(method).__name__, class_name)

    def test_baselines_share_data_protocol_but_may_tune_optimization(self):
        protocol = yaml.safe_load(
            (ROOT / "configs/baselines/imagenet_r.yaml").read_text()
        )
        self.assertEqual(protocol["dataset"], "imagenet_r")
        self.assertEqual(protocol["initial_classes"], 20)
        self.assertEqual(protocol["increment_classes"], 20)
        self.assertEqual(protocol["epochs"], 20)
        self.assertEqual(protocol["optimizer"], "sgd")
        expected = {
            "ease": "EASE",
            "sema": "SEMA",
            "tuna": "TUNA",
        }
        for method_name, class_name in expected.items():
            with self.subTest(method=method_name):
                method_config = yaml.safe_load(
                    (ROOT / f"configs/baselines/{method_name}.yaml").read_text()
                )
                for key in (
                    "dataset",
                    "num_classes",
                    "initial_classes",
                    "increment_classes",
                    "class_order_manifest",
                    "workers",
                ):
                    self.assertNotIn(key, method_config)
                method = make_method(
                    {**protocol, **method_config, "n_tasks": 10},
                    _ToyViT(),
                    torch.device("cpu"),
                )
                self.assertEqual(type(method).__name__, class_name)

    def test_runner_uses_method_parameter_contract(self):
        config = {"seed": 7, "batch_size": 2, "workers": 0}
        dataset = TensorDataset(
            torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 2.0]]),
            torch.tensor([0, 0, 1, 1]),
        )
        dataset.targets = [0, 0, 1, 1]  # type: ignore[attr-defined]
        protocol = TaskProtocol((0, 1), initial_classes=1, increment_classes=1)
        from methods.frozen_linear import FrozenLinear

        method = FrozenLinear(nn.Identity(), 0.1, 0.0, 0.0, 1, torch.device("cpu"))
        with tempfile.TemporaryDirectory() as root:
            logger = RunLogger(root, config, {})
            try:
                train_protocol(
                    method,
                    dataset,
                    dataset,
                    protocol,
                    config,
                    torch.device("cpu"),
                    logger,
                )
            finally:
                logger.close()
            with (logger.run_dir / "tasks.csv").open(newline="") as handle:
                records = list(csv.DictReader(handle))
        self.assertEqual(records[-1]["cumulative_parameters_total"], "6")
        self.assertEqual(records[-1]["cumulative_parameters_trainable"], "6")


if __name__ == "__main__":
    unittest.main()
