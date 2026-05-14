from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "digit_classifier.joblib"
DEFAULT_GALLERY_PATH = PROJECT_ROOT / "models" / "sample_gallery.npz"


def load_model_bundle(model_path: Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Run `python src/train.py` first."
        )

    bundle = joblib.load(model_path)
    required_keys = {"model", "target_names"}
    missing = required_keys.difference(bundle)
    if missing:
        raise KeyError(f"Model bundle is missing keys: {sorted(missing)}")
    if "input_shape" not in bundle and "image_shape" not in bundle:
        raise KeyError("Model bundle must include `input_shape` or `image_shape`.")
    return bundle


def model_input_shape(bundle: dict[str, Any] | None = None) -> tuple[int, int]:
    if bundle is None:
        return (28, 28)
    return tuple(bundle.get("input_shape", bundle.get("image_shape", (28, 28))))


def model_pixel_range(bundle: dict[str, Any] | None = None) -> tuple[float, float]:
    if bundle is None:
        return (0.0, 1.0)
    return tuple(bundle.get("pixel_range", (0.0, 1.0)))


def normalize_pixels(
    values: np.ndarray | list[float],
    input_shape: tuple[int, int] = (28, 28),
    pixel_range: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    pixels = np.asarray(values, dtype=float)
    expected_size = int(np.prod(input_shape))
    if pixels.shape == input_shape:
        pixels = pixels.reshape(1, -1)
    elif pixels.shape == (expected_size,):
        pixels = pixels.reshape(1, -1)
    elif len(pixels.shape) == 2 and pixels.shape[1] == expected_size:
        pass
    else:
        raise ValueError(
            f"Expected {expected_size} pixels or a {input_shape} image, "
            f"got shape {pixels.shape}."
        )

    pixel_min, pixel_max = pixel_range
    if pixel_max <= 1.0 and pixels.max(initial=0.0) > 1.0:
        pixels = pixels / 255.0
    elif pixel_max <= 16.0 and pixels.max(initial=0.0) > 16.0:
        pixels = pixels * (16.0 / 255.0)

    return np.clip(pixels, pixel_min, pixel_max)


def preprocess_uploaded_image(
    image: Image.Image,
    input_shape: tuple[int, int] = (28, 28),
    pixel_range: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Convert an uploaded digit image into the format expected by the model."""
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

    width, height = input_shape[1], input_shape[0]
    resized = grayscale.resize((width, height), Image.Resampling.LANCZOS)
    pixel_min, pixel_max = pixel_range
    pixels = np.asarray(resized, dtype=float) * (pixel_max / 255.0)
    pixels = np.clip(pixels, pixel_min, pixel_max)
    return normalize_pixels(pixels, input_shape=input_shape, pixel_range=pixel_range)


def load_sample_gallery(gallery_path: Path = DEFAULT_GALLERY_PATH) -> dict[str, np.ndarray]:
    if not gallery_path.exists():
        raise FileNotFoundError(
            f"Sample gallery not found at {gallery_path}. Run `python src/train.py` first."
        )
    gallery = np.load(gallery_path)
    return {"images": gallery["images"], "labels": gallery["labels"]}


def load_digit_sample(index: int) -> tuple[np.ndarray, int, np.ndarray]:
    gallery = load_sample_gallery()
    images = gallery["images"]
    labels = gallery["labels"]
    if index < 0 or index >= len(images):
        raise ValueError(f"sample-index must be between 0 and {len(images) - 1}.")

    image = images[index]
    pixels = image.reshape(1, -1)
    label = int(labels[index])
    return pixels, label, image


def predict_digit(
    bundle: dict[str, Any],
    pixels: np.ndarray | list[float],
    top_k: int = 3,
) -> dict[str, Any]:
    model = bundle["model"]
    input_shape = model_input_shape(bundle)
    pixel_range = model_pixel_range(bundle)
    prepared_pixels = normalize_pixels(
        pixels,
        input_shape=input_shape,
        pixel_range=pixel_range,
    )
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
