import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from methods.shared_adapter import SharedAdapter


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm2 = torch.nn.LayerNorm(2)
        self.mlp = torch.nn.Linear(2, 2, bias=False)

    def forward(self, inputs):
        return self.mlp(self.norm2(inputs))


class _ViT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block(), _Block()])

    def forward(self, inputs):
        for block in self.blocks:
            inputs = block(inputs)
        return inputs


def _method(mode: str, **kwargs) -> SharedAdapter:
    torch.manual_seed(11)
    return SharedAdapter(
        _ViT(),
        learning_rate=0.01,
        weight_decay=0.0,
        momentum=0.0,
        epochs=1,
        adapter_bottleneck=1,
        adapter_scale=0.1,
        adapter_dropout=0.0,
        adapt_start_layer=0,
        adapt_end_layer=1,
        mode=mode,
        rolann_ridge=0.1,
        device=torch.device("cpu"),
        **kwargs,
    )


def _task(points, labels):
    return DataLoader(
        TensorDataset(torch.tensor(points), torch.tensor(labels)), batch_size=2
    )


FIRST = ([[1.0, 0.0], [2.0, 0.0]], [0, 0])
SECOND = ([[0.0, 1.0], [0.0, 2.0]], [1, 1])


class SharedAdapterTests(unittest.TestCase):
    def test_adapter_reads_the_residual_stream_before_norm2(self):
        method = _method("shared")
        seen = {}
        block = method.backbone.blocks[0]
        block.norm2.register_forward_pre_hook(
            lambda module, inputs: seen.setdefault("norm2_input", inputs[0].clone())
        )
        original = method.adapters[0].forward
        method.adapters[0].forward = lambda x: (
            seen.setdefault("adapter_input", x.clone()),
            original(x),
        )[1]
        method._features(torch.ones(1, 2))
        self.assertTrue(torch.equal(seen["adapter_input"], seen["norm2_input"]))

    def test_adapter_starts_as_identity(self):
        method = _method("shared")
        images = torch.randn(3, 2)
        self.assertTrue(
            torch.equal(method._features(images), method._frozen_features(images))
        )

    def test_shared_adapter_keeps_training_on_later_tasks(self):
        method = _method("shared")
        method.train_task(_task(*FIRST), 0, 1)
        after_first = [p.detach().clone() for p in method.adapters.parameters()]
        method.train_task(_task(*SECOND), 1, 2)
        self.assertFalse(
            all(
                torch.equal(a, b)
                for a, b in zip(after_first, method.adapters.parameters())
            )
        )
        self.assertEqual(method.logits(torch.ones(1, 2)).shape, (1, 2))

    def test_frozen_mode_stops_adapter_updates_after_task_zero(self):
        method = _method("frozen")
        method.train_task(_task(*FIRST), 0, 1)
        frozen = [p.detach().clone() for p in method.adapters.parameters()]
        probe = method.probe.weight.detach().clone()
        method.train_task(_task(*SECOND), 1, 2)
        self.assertTrue(
            all(torch.equal(a, b) for a, b in zip(frozen, method.adapters.parameters()))
        )
        self.assertTrue(torch.equal(probe, method.probe.weight))
        self.assertEqual(method.logits(torch.ones(1, 2)).shape, (1, 2))

    def test_noadapt_fits_rolann_on_vanilla_features(self):
        method = _method("noadapt")
        self.assertEqual(len(method.adapters), 0)
        method.train_task(_task(*FIRST), 0, 1)
        method.train_task(_task(*SECOND), 1, 2)
        self.assertIsNone(method.probe)
        self.assertEqual(method.logits(torch.ones(1, 2)).shape, (1, 2))

    def test_distillation_and_compensation_run_and_round_trip(self):
        method = _method(
            "shared", protect="distill", protect_lambda=1.0, compensate=True
        )
        method.train_task(_task(*FIRST), 0, 1)
        self.assertIsNotNone(method._previous)
        method.train_task(_task(*SECOND), 1, 2)
        images = torch.tensor([[1.0, 0.5], [0.2, 1.0]])
        expected = method.logits(images)
        clone = _method(
            "shared", protect="distill", protect_lambda=1.0, compensate=True
        )
        clone.load_state_dict(method.state_dict())
        self.assertTrue(torch.allclose(clone.logits(images), expected))

    def test_compensation_is_identity_when_features_do_not_move(self):
        method = _method("shared", compensate=True)
        method.train_task(_task(*FIRST), 0, 1)
        before = [m.clone() for m in method.rolann.mg]
        method.adapters.requires_grad_(False)
        method._compensate_head(_task(*SECOND))
        for old, new in zip(before, method.rolann.mg):
            self.assertTrue(torch.allclose(old, new, atol=1e-5))

    def test_probe_test_accuracy_runs_on_eval_loader(self):
        method = _method("shared")
        data = _task([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 2.0]], [0, 0, 1, 1])
        with self.assertLogs("unified-vit-cil", level="INFO") as captured:
            method.train_task(data, 0, 2, eval_loader=data)
        self.assertTrue(any("test_acc=" in r.getMessage() for r in captured.records))

    def test_amp_is_automatically_disabled_without_cuda(self):
        self.assertFalse(_method("shared", amp=True).amp)

    def test_probe_learning_rate_defaults_to_tenth_of_adapter_lr(self):
        self.assertAlmostEqual(_method("shared").probe_learning_rate, 0.001)

    def test_rejects_unsupported_combinations(self):
        with self.assertRaises(ValueError):
            _method("frozen", compensate=True)
        with self.assertRaises(ValueError):
            _method("shared", protect="distill")  # lambda must be positive
        with self.assertRaises(ValueError):
            _method("selective")


if __name__ == "__main__":
    unittest.main()
