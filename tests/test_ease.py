import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from methods.ease import EASE


class _ToyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm2 = torch.nn.Identity()
        self.mlp = torch.nn.Identity()

    def forward(self, features):
        return features + self.mlp(self.norm2(features))


class _ToyViT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_ToyBlock(), _ToyBlock()])

    def forward(self, features):
        for block in self.blocks:
            features = block(features)
        return features


class EASETests(unittest.TestCase):
    def test_task_subspaces_prototypes_and_state_round_trip(self):
        method = EASE(
            _ToyViT(),
            learning_rate=0.1,
            weight_decay=0.0,
            momentum=0.0,
            epochs=1,
            device=torch.device("cpu"),
            adapter_dim=1,
        )
        first_task = DataLoader(
            TensorDataset(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([0, 1])),
            batch_size=2,
        )
        second_task = DataLoader(
            TensorDataset(torch.tensor([[-1.0, 0.0]]), torch.tensor([2])), batch_size=1
        )
        method.train_task(first_task, task_id=0, seen_classes=2)
        method.train_task(second_task, task_id=1, seen_classes=3)

        self.assertEqual(method.classifier.weight.shape, (3, 4))
        self.assertEqual(len(method.classifier.adapters[0]), 2)
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in method.classifier.adapters[0].parameters()
            )
        )
        inputs = torch.tensor([[1.0, 0.0]])
        baseline = method._features(inputs, None)
        with torch.no_grad():
            method.classifier.adapters[0][0].up.bias.fill_(1.0)
        self.assertFalse(torch.equal(method._features(inputs, 0), baseline))
        logits = method.logits(torch.tensor([[1.0, 0.0]]))
        self.assertEqual(logits.shape, (1, 3))

        restored = EASE(_ToyViT(), 0.1, 0.0, 0.0, 1, torch.device("cpu"), adapter_dim=1)
        restored.load_state_dict(method.state_dict())
        self.assertTrue(
            torch.equal(restored.logits(torch.tensor([[1.0, 0.0]])), logits)
        )

    def test_prototypes_use_the_deterministic_loader(self):
        method = EASE(_ToyViT(), 0.1, 0.0, 0.0, 1, torch.device("cpu"), adapter_dim=1)
        training = DataLoader(
            TensorDataset(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([0, 1])),
            batch_size=2,
        )
        prototypes = DataLoader(
            TensorDataset(
                torch.tensor([[10.0, 0.0], [0.0, 20.0]]), torch.tensor([0, 1])
            ),
            batch_size=2,
        )
        method.train_task(training, 0, 2, prototype_loader=prototypes)
        expected = method._subspaces(torch.tensor([[10.0, 0.0], [0.0, 20.0]]))
        self.assertTrue(torch.allclose(method.classifier.weight, expected))

    @unittest.skipUnless(torch.cuda.is_available(), "cuda")
    def test_new_adapters_are_created_on_the_method_device(self):
        device = torch.device("cuda")
        method = EASE(
            _ToyViT().to(device),
            learning_rate=0.1,
            weight_decay=0.0,
            momentum=0.0,
            epochs=1,
            device=device,
            adapter_dim=1,
        )
        loader = DataLoader(
            TensorDataset(
                torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device),
                torch.tensor([0, 1], device=device),
            ),
            batch_size=2,
        )
        method.train_task(loader, task_id=0, seen_classes=2)
        self.assertTrue(
            all(
                parameter.device.type == "cuda"
                for parameter in method.classifier.adapters.parameters()
            )
        )
        method.logits(torch.tensor([[1.0, 0.0]], device=device))


if __name__ == "__main__":
    unittest.main()
