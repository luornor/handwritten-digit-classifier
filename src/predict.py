from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_digits


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "digit_classifier.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a handwritten digit.")
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index from the bundled scikit-learn digits dataset to score.",
    )
    parser.add_argument(
        "--pixels-json",
        help="Optional JSON list of 64 grayscale pixel values.",
    )
    return parser.parse_args()


def load_input(args: argparse.Namespace) -> tuple[np.ndarray, int | None]:
    if args.pixels_json:
        values = np.asarray(json.loads(args.pixels_json), dtype=float)
        if values.shape != (64,):
            raise ValueError(f"Expected 64 pixel values, got shape {values.shape}.")
        return values.reshape(1, -1), None

    data = load_digits()
    if args.sample_index < 0 or args.sample_index >= len(data.data):
        raise ValueError(f"sample-index must be between 0 and {len(data.data) - 1}.")
    return data.data[args.sample_index].reshape(1, -1), int(data.target[args.sample_index])


def main() -> None:
    args = parse_args()
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    row, true_label = load_input(args)

    predicted = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]
    output = {
        "prediction": predicted,
        "true_label": true_label,
        "top_probabilities": [
            {"digit": int(index), "probability": round(float(probability), 4)}
            for index, probability in sorted(
                enumerate(probabilities),
                key=lambda item: item[1],
                reverse=True,
            )[:3]
        ],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
