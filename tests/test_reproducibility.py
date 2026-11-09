import unittest
from unittest.mock import patch

import torch

from utils.reproducibility import parse_device, seed_everything


class ReproducibilityTests(unittest.TestCase):
    def test_deterministic_mode_selects_math_sdp_backend(self):
        with (
            patch(
                "utils.reproducibility.torch.backends.cuda.enable_flash_sdp"
            ) as flash,
            patch(
                "utils.reproducibility.torch.backends.cuda.enable_mem_efficient_sdp"
            ) as memory_efficient,
            patch("utils.reproducibility.torch.backends.cuda.enable_math_sdp") as math,
        ):
            seed_everything(7)

        flash.assert_called_once_with(False)
        memory_efficient.assert_called_once_with(False)
        math.assert_called_once_with(True)

    def test_parse_device_accepts_gpu_index_or_cpu(self):
        self.assertEqual(parse_device("cpu"), torch.device("cpu"))
        if torch.cuda.is_available():
            self.assertEqual(parse_device("0"), torch.device("cuda:0"))
            self.assertEqual(parse_device(0), torch.device("cuda:0"))
            if torch.cuda.device_count() > 1:
                self.assertEqual(parse_device("1"), torch.device("cuda:1"))
            else:
                with self.assertRaises(ValueError):
                    parse_device("1")
        else:
            with self.assertRaises(ValueError):
                parse_device("0")


if __name__ == "__main__":
    unittest.main()
