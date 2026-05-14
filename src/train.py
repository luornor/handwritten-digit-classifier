from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from scipy import ndimage
from sklearn.datasets import fetch_openml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "digit_classifier.joblib"
GALLERY_PATH = PROJECT_ROOT / "models" / "sample_gallery.npz"
REPORT_PATH = PROJECT_ROOT / "reports" / "metrics.json"
DATA_CACHE = PROJECT_ROOT / "data_cache"
INPUT_SHAPE = (28, 28)
PIXEL_RANGE = (0.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the digit classifier.")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=20000,
        help="Maximum number of MNIST training samples to use before augmentation.",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=10000,
        help="Number of holdout samples to evaluate.",
    )
    parser.add_argument(
        "--augment-copies",
        type=int,
        default=1,
        help="Number of augmented copies to create per training image.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=20,
        help="Maximum MLP training iterations.",
    )
    return parser.parse_args()


def build_model(max_iter: int) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        batch_size=256,
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        max_iter=max_iter,
        random_state=SEED,
        verbose=False,
    )


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_mnist() -> tuple[np.ndarray, np.ndarray]:
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    mnist = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        parser="auto",
        cache=True,
        data_home=DATA_CACHE,
    )
    x = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int64)
    return x, y


def limit_samples(
    x: np.ndarray,
    y: np.ndarray,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples <= 0 or max_samples >= len(x):
        return x, y

    _, x_subset, _, y_subset = train_test_split(
        x,
        y,
        test_size=max_samples,
        stratify=y,
        random_state=seed,
    )
    return x_subset, y_subset


def augment_training_data(
    x: np.ndarray,
    y: np.ndarray,
    copies: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if copies <= 0:
        return x, y

    rng = np.random.default_rng(seed)
    images = x.reshape(-1, *INPUT_SHAPE)
    augmented_images = [x]
    augmented_labels = [y]

    for _ in range(copies):
        transformed = np.empty_like(images)
        for index, image in enumerate(images):
            rotated = ndimage.rotate(
                image,
                angle=float(rng.uniform(-12, 12)),
                reshape=False,
                order=1,
                mode="constant",
                cval=0.0,
            )
            shifted = ndimage.shift(
                rotated,
                shift=(float(rng.uniform(-2, 2)), float(rng.uniform(-2, 2))),
                order=1,
                mode="constant",
                cval=0.0,
            )
            transformed[index] = np.clip(shifted, 0.0, 1.0)

        augmented_images.append(transformed.reshape(len(transformed), -1))
        augmented_labels.append(y)

    return np.vstack(augmented_images), np.concatenate(augmented_labels)


def save_sample_gallery(
    x_test: np.ndarray,
    y_test: np.ndarray,
    size: int = 240,
    seed: int = SEED,
) -> None:
    rng = np.random.default_rng(seed)
    gallery_size = min(size, len(x_test))
    indices = rng.choice(len(x_test), size=gallery_size, replace=False)
    images = x_test[indices].reshape(-1, *INPUT_SHAPE)
    labels = y_test[indices]

    GALLERY_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(GALLERY_PATH, images=images, labels=labels)


def main() -> None:
    args = parse_args()
    x, y = load_mnist()

    holdout_fraction = min(max(args.test_samples / len(x), 0.05), 0.3)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=holdout_fraction,
        stratify=y,
        random_state=SEED,
    )
    x_test, y_test = limit_samples(x_test, y_test, args.test_samples, SEED)
    x_train, y_train = limit_samples(x_train, y_train, args.max_train_samples, SEED)
    x_augmented, y_augmented = augment_training_data(
        x_train,
        y_train,
        copies=args.augment_copies,
        seed=SEED,
    )

    model = build_model(max_iter=args.max_iter)
    model.fit(x_augmented, y_augmented)
    predictions = model.predict(x_test)

    metrics = {
        "dataset": "mnist_784_openml",
        "input_shape": list(INPUT_SHAPE),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "training_samples_before_augmentation": int(len(x_train)),
        "training_samples_after_augmentation": int(len(x_augmented)),
        "test_samples": int(len(x_test)),
        "augment_copies": int(args.augment_copies),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
        ),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "input_shape": INPUT_SHAPE,
            "pixel_range": PIXEL_RANGE,
            "target_names": [str(target) for target in range(10)],
            "dataset": "OpenML mnist_784",
            "model_type": "MLPClassifier(hidden_layer_sizes=(256, 128))",
            "sklearn_version": sklearn.__version__,
        },
        MODEL_PATH,
    )
    save_sample_gallery(x_test, y_test)
    save_json(REPORT_PATH, metrics)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved sample gallery to {GALLERY_PATH}")
    print(f"Saved metrics to {REPORT_PATH}")
    print(f"Test accuracy: {metrics['accuracy']:.3f}")


if __name__ == "__main__":
    main()
