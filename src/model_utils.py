from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image, ImageOps
from sklearn.datasets import load_digits


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "digit_classifier.joblib"


def load_model_bundle(model_path: Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Run `python src/train.py` first."
        )

    bundle = joblib.load(model_path)
    required_keys = {"model", "image_shape", "target_names"}
    missing = required_keys.difference(bundle)
    if missing:
        raise KeyError(f"Model bundle is missing keys: {sorted(missing)}")
    return bundle


def normalize_pixels(values: np.ndarray | list[float]) -> np.ndarray:
    pixels = np.asarray(values, dtype=float)
    if pixels.shape == (8, 8):
        pixels = pixels.reshape(1, -1)
    elif pixels.shape == (64,):
        pixels = pixels.reshape(1, -1)
    elif len(pixels.shape) == 2 and pixels.shape[1] == 64:
        pass
    else:
        raise ValueError(f"Expected 64 pixels or an 8x8 image, got shape {pixels.shape}.")

    return np.clip(pixels, 0.0, 16.0)


def preprocess_uploaded_image(image: Image.Image) -> np.ndarray:
    """Convert an uploaded digit image into the 8x8, 0-16 format used by sklearn."""
    grayscale = ImageOps.exif_transpose(image).convert("L")
    grayscale = ImageOps.autocontrast(grayscale)

    raw = np.asarray(grayscale, dtype=float)
    if raw.mean() > 127:
        grayscale = ImageOps.invert(grayscale)

    grayscale = ImageOps.autocontrast(grayscale)
    mask = grayscale.point(lambda pixel: 255 if pixel > 20 else 0)
    bbox = mask.getbbox()
    if bbox:
        digit = grayscale.crop(bbox)
        side = max(digit.size)
        canvas = Image.new("L", (side, side), 0)
        offset = ((side - digit.width) // 2, (side - digit.height) // 2)
        canvas.paste(digit, offset)
        grayscale = canvas

    resized = grayscale.resize((8, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(resized, dtype=float) * (16.0 / 255.0)
    return normalize_pixels(pixels)


def load_digit_sample(index: int) -> tuple[np.ndarray, int, np.ndarray]:
    digits = load_digits()
    if index < 0 or index >= len(digits.data):
        raise ValueError(f"sample-index must be between 0 and {len(digits.data) - 1}.")

    pixels = digits.data[index].reshape(1, -1)
    label = int(digits.target[index])
    image = digits.images[index]
    return pixels, label, image


def predict_digit(
    bundle: dict[str, Any],
    pixels: np.ndarray | list[float],
    top_k: int = 3,
) -> dict[str, Any]:
    model = bundle["model"]
    prepared_pixels = normalize_pixels(pixels)
    probabilities = model.predict_proba(prepared_pixels)[0]
    classes = getattr(model, "classes_", np.arange(len(probabilities)))
    ranked = sorted(
        zip(classes, probabilities),
        key=lambda item: item[1],
        reverse=True,
    )[:top_k]

    return {
        "prediction": int(model.predict(prepared_pixels)[0]),
        "top_probabilities": [
            {"digit": int(label), "probability": float(probability)}
            for label, probability in ranked
        ],
    }
