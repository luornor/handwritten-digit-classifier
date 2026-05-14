from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "digit_classifier.joblib"
REPORT_PATH = PROJECT_ROOT / "reports" / "metrics.json"


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    C=10,
                    gamma="scale",
                    kernel="rbf",
                    probability=True,
                    random_state=SEED,
                ),
            ),
        ]
    )


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    data = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        stratify=data.target,
        random_state=SEED,
    )

    model = build_model()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="accuracy")

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "cv_accuracy_mean": float(np.mean(cv_scores)),
        "cv_accuracy_std": float(np.std(cv_scores)),
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
            "image_shape": data.images[0].shape,
            "target_names": [str(target) for target in data.target_names],
        },
        MODEL_PATH,
    )
    save_json(REPORT_PATH, metrics)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {REPORT_PATH}")
    print(f"Test accuracy: {metrics['accuracy']:.3f}")


if __name__ == "__main__":
    main()
