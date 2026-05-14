# Handwritten Digit Classifier

This project trains a lightweight classifier for handwritten digits from 0 to 9. It modernizes the old MNIST/Keras notebook into a small scikit-learn project that runs quickly without TensorFlow.

## Why this project is useful

- Uses image-like 8x8 pixel data from scikit-learn's built-in digits dataset.
- Trains a support-vector classifier inside a reproducible preprocessing pipeline.
- Reports accuracy, macro F1, cross-validation accuracy, and confusion matrix.
- Includes a simple prediction CLI for demoing the saved model.

## Dataset

The project uses the scikit-learn digits dataset. No external download is required.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/train.py
python src/predict.py --sample-index 12
```

The training script writes:

- `models/digit_classifier.joblib`
- `reports/metrics.json`

## Model choice

An SVM is used here because the dataset is small and the goal is a compact portfolio project. A CNN would be a good follow-up if this project is expanded with full MNIST or Fashion-MNIST.
