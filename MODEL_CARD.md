# Model Card

## Model Details

- Model type: Multilayer perceptron classifier
- Framework: scikit-learn
- Dataset: OpenML `mnist_784`
- Input shape: 784 numeric pixels representing a 28x28 grayscale digit
- Output: Digit class from 0 to 9
- Augmentation: random small rotations and shifts applied to training images

## Intended Use

This model is intended as a portfolio demo for handwritten digit recognition. It is suitable for simple examples, sample gallery images, and uploaded high-contrast images containing one digit.

## Not Intended For

- Banking, identity verification, exam grading, or other high-stakes use
- Production OCR
- Complex natural images
- Reading multi-digit numbers
- Reading cluttered natural scenes

## Evaluation

The saved metrics are in `reports/metrics.json`. The current training run used 20,000 MNIST samples plus 20,000 augmented samples and evaluated on 10,000 holdout samples.

- Accuracy: 97.58%
- Macro F1: 97.56%

## Limitations

The model is trained on single, centered MNIST digits. Uploaded phone-camera images, low-contrast writing, heavy rotation, cropped digits, multiple digits in one image, and stylized handwriting can still perform worse. The app includes inversion, cropping, centering, resizing, and confidence display, but it is still a portfolio model rather than production OCR.

## Improvements Over The Baseline

- Uses 28x28 MNIST-style images instead of 8x8 sample digits
- Adds rotation and shift augmentation during training
- Uses a neural network instead of an SVM on flattened 8x8 pixels
- Includes an app sample gallery generated from the holdout set
- Keeps deployment lightweight by avoiding TensorFlow/PyTorch

## Maintenance

Rerun `python src/train.py` after dependency upgrades or model changes. Training requires internet access the first time so scikit-learn can download MNIST from OpenML. Then run:

```bash
python -m unittest discover -s tests
```
