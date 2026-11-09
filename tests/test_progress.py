import unittest

from utils.logging import log_eval_set_sizes, log_evaluation
from utils.progress import EpochProgress


class ProgressTests(unittest.TestCase):
    def test_epoch_summary_includes_task_and_epoch_number(self):
        with self.assertLogs("unified-vit-cil", level="INFO") as captured:
            with EpochProgress(2, task_id=0) as progress:
                for _ in progress:
                    progress.update(0.5, correct=1, samples=2)
        messages = [record.getMessage() for record in captured.records]
        self.assertIn("Task 0, epoch 1: loss=0.5000 train_acc=50.0%", messages)
        self.assertIn("Task 0, epoch 2: loss=0.5000 train_acc=50.0%", messages)

    def test_epoch_summary_includes_test_acc_when_set(self):
        with self.assertLogs("unified-vit-cil", level="INFO") as captured:
            with EpochProgress(1, task_id=0) as progress:
                for _ in progress:
                    progress.update(0.25, correct=3, samples=4)
                    progress.set_test_acc(75.0)
        messages = [record.getMessage() for record in captured.records]
        self.assertIn(
            "Task 0, epoch 1: loss=0.2500 train_acc=75.0% test_acc=75.0%", messages
        )

    def test_epoch_summary_includes_diversity_only_when_reported(self):
        with self.assertLogs("unified-vit-cil", level="INFO") as captured:
            with EpochProgress(1, task_id=0) as progress:
                for _ in progress:
                    progress.update(0.5, correct=1, samples=2, diversity=0.01)
        messages = [record.getMessage() for record in captured.records]
        self.assertIn("div=0.0100", messages[0])


class EvaluationLogTests(unittest.TestCase):
    def test_evaluation_logs_decomposition_and_optional_metrics(self):
        with self.assertLogs("unified-vit-cil", level="INFO") as captured:
            log_evaluation(
                {"0": 80.0, "1": 60.0},
                overall_accuracy=70.0,
                plasticity=90.0,
                stability=None,
            )
        messages = [record.getMessage() for record in captured.records]
        self.assertIn("Start evaluation", messages)
        self.assertIn("  Task 0: 80.00%", messages)
        self.assertIn("  Task 1: 60.00%", messages)
        self.assertIn("  Overall (all seen classes): 70.00%", messages)
        self.assertIn("  Plasticity: 90.00%", messages)
        self.assertFalse(any("Stability" in message for message in messages))

    def test_eval_set_sizes_log_image_scale(self):
        from datasets.protocol import TaskProtocol

        class _Eval:
            targets = [0, 0, 1, 1, 1]

        with self.assertLogs("unified-vit-cil", level="INFO") as captured:
            log_eval_set_sizes(_Eval(), TaskProtocol((0, 1), 1, 1))
        messages = [record.getMessage() for record in captured.records]
        self.assertIn("  Task 0 eval images: 2", messages)
        self.assertIn("  Task 1 eval images: 3", messages)
        self.assertTrue(
            any("All-class eval images: 5" in message for message in messages)
        )


if __name__ == "__main__":
    unittest.main()
