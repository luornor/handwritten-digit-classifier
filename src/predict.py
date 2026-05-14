from __future__ import annotations

import argparse
import json

import numpy as np
from model_utils import load_digit_sample, load_model_bundle, predict_digit


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
        return values, None

    pixels, true_label, _ = load_digit_sample(args.sample_index)
    return pixels, true_label


def main() -> None:
    args = parse_args()
    bundle = load_model_bundle()
    row, true_label = load_input(args)

    result = predict_digit(bundle, row)
    output = {
        "prediction": result["prediction"],
        "true_label": true_label,
        "top_probabilities": result["top_probabilities"],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
