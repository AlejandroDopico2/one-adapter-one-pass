import tempfile
import unittest
from pathlib import Path

from PIL import Image

from datasets.registry import canonical_transform, resolve_data_root


class RegistryTests(unittest.TestCase):
    def test_cifar100_accepts_extract_or_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            extract = parent / "cifar-100-python"
            extract.mkdir()
            (extract / "train").write_bytes(b"x")
            self.assertEqual(resolve_data_root("cifar100", parent), parent)
            self.assertEqual(resolve_data_root("cifar100", extract), parent)

    def test_imagenet_r_finds_alias_under_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            dataset = parent / "imagenet-r"
            (dataset / "train").mkdir(parents=True)
            (dataset / "val").mkdir()
            self.assertEqual(resolve_data_root("imagenet_r", parent), dataset)
            self.assertEqual(resolve_data_root("imagenet_r", dataset), dataset)

    def test_canonical_transform_converts_non_rgb_to_three_channels(self):
        image = Image.new("L", (32, 32), 128)
        tensor = canonical_transform(False)(image)
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))

    def test_sema_transform_matches_unnormalized_paper_recipe(self):
        image = Image.new("RGB", (256, 256), 128)
        tensor = canonical_transform(False, "sema")(image)
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        self.assertTrue(0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0)


if __name__ == "__main__":
    unittest.main()
