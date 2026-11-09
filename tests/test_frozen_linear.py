import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from methods.frozen_linear import FrozenLinear


class FrozenLinearTests(unittest.TestCase):
    def test_expansion_preserves_existing_classifier_weights(self):
        torch.manual_seed(7)
        method = FrozenLinear(
            torch.nn.Identity(),
            learning_rate=0.1,
            weight_decay=0.0,
            momentum=0.0,
            epochs=1,
            device=torch.device("cpu"),
        )
        task_zero = DataLoader(
            TensorDataset(torch.tensor([[1.0, 0.0], [2.0, 0.0]]), torch.tensor([0, 0])),
            batch_size=2,
        )
        method.train_task(task_zero, task_id=0, seen_classes=1)
        old_weight = method.classifier.weight.detach().clone()
        old_bias = method.classifier.bias.detach().clone()
        method._expand_classifier(feature_dim=2, seen_classes=2)
        self.assertTrue(torch.equal(method.classifier.weight[0], old_weight[0]))
        self.assertTrue(torch.equal(method.classifier.bias[0], old_bias[0]))
        self.assertEqual(method.logits(torch.ones(1, 2)).shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
