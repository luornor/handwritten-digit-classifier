# Handwritten Digit Classifier

This project trains and deploys a lightweight classifier for handwritten digits from 0 to 9. It modernizes the old notebook into a Streamlit demo with MNIST-scale 28x28 inputs, augmentation, reusable preprocessing, tests, and CI.

## Why this project is useful

- Uses MNIST from OpenML for more realistic 28x28 handwritten digit images.
- Trains a compact neural network with rotation and shift augmentation.
- Reports accuracy, macro F1, training sample counts, and confusion matrix.
- Includes CLI prediction, image-upload preprocessing, and a Streamlit app.
- Includes a model card and GitHub Actions workflow.

## Dataset

The model is trained from OpenML's `mnist_784` dataset. Training downloads the dataset once into `data_cache/`, which is ignored by Git. The deployed app does not need to download the dataset because the trained model and a small sample gallery are committed.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/train.py
python src/predict.py --sample-index 12
python -m unittest discover -s tests
streamlit run app.py
```

The training script writes:

- `models/digit_classifier.joblib`
- `models/sample_gallery.npz`
- `reports/metrics.json`

## Current result

The upgraded model was trained on 20,000 MNIST samples plus 20,000 augmented samples and evaluated on 10,000 holdout samples:

- Accuracy: 97.58%
- Macro F1: 97.56%

## Deploy

Recommended platform: Streamlit Community Cloud.

1. Push this folder to a GitHub repo named `handwritten-digit-classifier`.
2. Go to Streamlit Community Cloud and create a new app from that repo.
3. Set the main file path to `app.py`.
4. Keep the Python version from `runtime.txt`.
5. Deploy.

The saved model is already included in `models/digit_classifier.joblib`, and the app sample browser uses `models/sample_gallery.npz`, so the app can start without retraining.

## Model choice

The model uses scikit-learn's `MLPClassifier` instead of a CNN to keep deployment light on Streamlit while still moving beyond the original 8x8 SVM baseline. A CNN can still be a future upgrade if the deployment target can comfortably support TensorFlow or PyTorch.

## Limitations

This model is stronger than the original 8x8 baseline, but unusual handwriting, cluttered photos, rotated pages, and multi-digit uploads can still fail. See `MODEL_CARD.md` for the full model notes.
