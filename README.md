# Handwritten Digit Classifier

This project trains and deploys a lightweight classifier for handwritten digits from 0 to 9. It modernizes the old MNIST/Keras notebook into a small scikit-learn project with a Streamlit demo, reusable preprocessing, tests, and CI.

## Why this project is useful

- Uses image-like 8x8 pixel data from scikit-learn's built-in digits dataset.
- Trains a support-vector classifier inside a reproducible preprocessing pipeline.
- Reports accuracy, macro F1, cross-validation accuracy, and confusion matrix.
- Includes CLI prediction, image-upload preprocessing, and a Streamlit app.
- Includes a model card and GitHub Actions workflow.

## Dataset

The project uses the scikit-learn digits dataset. No external download is required.

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
- `reports/metrics.json`

## Deploy

Recommended platform: Streamlit Community Cloud.

1. Push this folder to a GitHub repo named `handwritten-digit-classifier`.
2. Go to Streamlit Community Cloud and create a new app from that repo.
3. Set the main file path to `app.py`.
4. Keep the Python version from `runtime.txt`.
5. Deploy.

The saved model is already included in `models/digit_classifier.joblib`, so the app can start without retraining. The CI workflow still retrains during checks to make sure the code stays reproducible.

## Model choice

An SVM is used here because the dataset is small and the goal is a compact portfolio project. A CNN would be a good follow-up if this project is expanded with full MNIST or Fashion-MNIST.

## Limitations

This model is trained on clean 8x8 digit images. Uploaded phone photos or stylized handwriting may be less accurate, even with preprocessing. See `MODEL_CARD.md` for the full model notes.
