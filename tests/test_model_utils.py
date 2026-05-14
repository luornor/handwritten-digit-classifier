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
    model_input_shape,
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
        shape = model_input_shape(load_model_bundle())
        size = int(np.prod(shape))
        flat = np.arange(size)
        image = np.arange(size).reshape(shape)

        normalized_flat = normalize_pixels(flat, input_shape=shape)
        normalized_image = normalize_pixels(image, input_shape=shape)
        self.assertEqual(normalized_flat.shape, (1, size))
        self.assertEqual(normalized_image.shape, (1, size))
        self.assertTrue(np.all(normalized_flat <= 1.0))

        with self.assertRaises(ValueError):
            normalize_pixels(np.arange(64), input_shape=shape)

    def test_preprocess_uploaded_image_returns_model_input(self) -> None:
        image = Image.new("L", (64, 64), 255)
        draw = ImageDraw.Draw(image)
        draw.line((30, 8, 30, 56), fill=0, width=8)

        pixels = preprocess_uploaded_image(image)

        shape = model_input_shape(load_model_bundle())
        self.assertEqual(pixels.shape, (1, int(np.prod(shape))))
        self.assertGreaterEqual(float(pixels.min()), 0.0)
        self.assertLessEqual(float(pixels.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
