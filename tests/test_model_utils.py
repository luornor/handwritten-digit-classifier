from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from model_utils import (  # noqa: E402
    load_digit_sample,
    load_model_bundle,
    normalize_pixels,
    predict_digit,
    preprocess_uploaded_image,
)


class ModelUtilsTest(unittest.TestCase):
    def test_model_bundle_predicts_dataset_sample(self) -> None:
        bundle = load_model_bundle()
        pixels, true_label, _ = load_digit_sample(12)
        result = predict_digit(bundle, pixels)

        self.assertEqual(result["prediction"], true_label)
        self.assertEqual(len(result["top_probabilities"]), 3)
        self.assertGreaterEqual(result["top_probabilities"][0]["probability"], 0)
        self.assertLessEqual(result["top_probabilities"][0]["probability"], 1)

    def test_normalize_pixels_accepts_flat_or_image_shape(self) -> None:
        flat = np.arange(64)
        image = np.arange(64).reshape(8, 8)

        self.assertEqual(normalize_pixels(flat).shape, (1, 64))
        self.assertEqual(normalize_pixels(image).shape, (1, 64))
        self.assertTrue(np.all(normalize_pixels(flat) <= 16))

    def test_preprocess_uploaded_image_returns_model_input(self) -> None:
        image = Image.new("L", (64, 64), 255)
        draw = ImageDraw.Draw(image)
        draw.line((30, 8, 30, 56), fill=0, width=8)

        pixels = preprocess_uploaded_image(image)

        self.assertEqual(pixels.shape, (1, 64))
        self.assertGreaterEqual(float(pixels.min()), 0.0)
        self.assertLessEqual(float(pixels.max()), 16.0)


if __name__ == "__main__":
    unittest.main()
