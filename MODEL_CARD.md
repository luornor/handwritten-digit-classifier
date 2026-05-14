# Model Card

## Model Details

- Model type: Support Vector Classifier with RBF kernel
- Framework: scikit-learn
- Dataset: `sklearn.datasets.load_digits`
- Input shape: 64 numeric pixels representing an 8x8 grayscale digit
- Output: Digit class from 0 to 9

## Intended Use

This model is intended as a portfolio demo for handwritten digit recognition. It is suitable for simple examples, sample dataset images, and uploaded high-contrast digit images.

## Not Intended For

- Banking, identity verification, exam grading, or other high-stakes use
- Production OCR
- Complex natural images
- Full MNIST-quality benchmarking

## Evaluation

The saved metrics are in `reports/metrics.json`. The current training run reports about 98% test accuracy and macro F1 on a stratified holdout set.

## Limitations

The training data contains clean 8x8 images. Real phone-camera images, rotated digits, low-contrast writing, and unusual handwriting can perform worse. The app includes preprocessing for uploaded images, but the safest demo path is still the built-in sample browser.

## Maintenance

Rerun `python src/train.py` after dependency upgrades or model changes, then run:

```bash
python -m unittest discover -s tests
```
