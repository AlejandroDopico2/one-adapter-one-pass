import unittest

from utils.metrics import AccuracyMatrix


class MetricTests(unittest.TestCase):
    def test_matrix_metrics_match_documented_formulas(self):
        matrix = AccuracyMatrix()
        matrix.record(0, 0, 0.8)
        matrix.record(1, 0, 0.6)
        matrix.record(1, 1, 0.9)
        matrix.record(2, 0, 0.5)
        matrix.record(2, 1, 0.7)
        matrix.record(2, 2, 1.0)
        summary = matrix.summary()
        self.assertAlmostEqual(summary["final_accuracy"], 0.7333333333)
        self.assertAlmostEqual(
            summary["mean_incremental_accuracy"], (0.8 + 0.75 + 0.7333333333) / 3
        )
        self.assertAlmostEqual(summary["plasticity"], 0.9)
        self.assertAlmostEqual(summary["stability"], 0.6)
        self.assertAlmostEqual(summary["mean_forgetting"], 0.25)
        self.assertNotIn("overall_accuracy", summary)

        matrix.record_overall(0.8)
        matrix.record_overall(0.75)
        matrix.record_overall(0.65)
        summary = matrix.summary()
        self.assertAlmostEqual(summary["overall_accuracy"], 0.65)
        self.assertEqual(summary["overall_accuracies"], [0.8, 0.75, 0.65])
        self.assertAlmostEqual(summary["final_accuracy"], 0.65)
        self.assertAlmostEqual(
            summary["mean_incremental_accuracy"], (0.8 + 0.75 + 0.65) / 3
        )


if __name__ == "__main__":
    unittest.main()
