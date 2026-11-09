import tempfile
import unittest
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset

from datasets.protocol import (
    RemappedSubset,
    TaskProtocol,
    load_class_order,
    shuffle_class_order,
)


class Samples(Dataset):
    def __init__(self):
        self.targets = [2, 0, 1, 2]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return torch.tensor([index]), self.targets[index]


class ProtocolTests(unittest.TestCase):
    def test_manifest_is_validated_and_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "order.yaml"
            path.write_text(yaml.safe_dump({"class_order": [2, 0, 1]}))
            order, digest = load_class_order(path, 3)
        self.assertEqual(order, [2, 0, 1])
        self.assertEqual(len(digest), 64)

    def test_task_slicing_and_label_remapping_are_fixed(self):
        protocol = TaskProtocol((2, 0, 1), initial_classes=2, increment_classes=1)
        self.assertEqual(protocol.num_tasks, 2)
        self.assertEqual(protocol.classes_for_task(0), (2, 0))
        self.assertEqual(protocol.seen_classes(1), (2, 0, 1))
        subset = RemappedSubset(
            Samples(), protocol.classes_for_task(0), protocol.class_order
        )
        self.assertEqual([subset[i][1] for i in range(len(subset))], [0, 1, 0])

    def test_seed_shuffle_is_deterministic_and_complete(self):
        identity = list(range(10))
        first = shuffle_class_order(identity, seed=1993)
        second = shuffle_class_order(identity, seed=1993)
        other = shuffle_class_order(identity, seed=7)
        self.assertEqual(first, second)
        self.assertNotEqual(first, identity)
        self.assertNotEqual(first, other)
        self.assertEqual(sorted(first), identity)

    def test_paper_imagenet_r_protocol_is_b0_inc20(self):
        root = Path(__file__).resolve().parents[1]
        order, _ = load_class_order(
            root / "configs/manifests/imagenet_r_identity.yaml", 200
        )
        self.assertEqual(order, list(range(200)))
        benchmark = yaml.safe_load(
            (root / "configs/benchmarks/imagenet_r_b0_inc20_200_in21k.yaml").read_text()
        )
        self.assertEqual(benchmark["num_classes"], 200)
        protocol = TaskProtocol(
            tuple(order),
            benchmark["initial_classes"],
            benchmark["increment_classes"],
        )
        self.assertEqual(protocol.num_tasks, 10)
        self.assertEqual(len(protocol.classes_for_task(0)), 20)


if __name__ == "__main__":
    unittest.main()
