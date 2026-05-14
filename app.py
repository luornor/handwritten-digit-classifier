from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from model_utils import (  # noqa: E402
    load_digit_sample,
    load_sample_gallery,
    load_model_bundle,
    model_input_shape,
    model_pixel_range,
    predict_digit,
    preprocess_uploaded_image,
)


st.set_page_config(page_title="Handwritten Digit Classifier", page_icon="0", layout="centered")


@st.cache_resource
def get_model_bundle() -> dict:
    return load_model_bundle()


def show_prediction(pixels, true_label=None) -> None:
    bundle = get_model_bundle()
    result = predict_digit(bundle, pixels, top_k=5)
    top_probability = result["top_probabilities"][0]["probability"]
    probabilities = pd.DataFrame(result["top_probabilities"])
    probabilities["probability"] = probabilities["probability"].round(4)
    probabilities = probabilities.set_index("digit")

    col_a, col_b = st.columns(2)
    col_a.metric("Prediction", result["prediction"])
    if true_label is not None:
        col_b.metric("Dataset label", true_label)

    st.bar_chart(probabilities)
    if top_probability < 0.65:
        st.warning("Low confidence. Try a clearer, centered, single digit image.")


st.title("Handwritten Digit Classifier")

input_mode = st.sidebar.radio("Input", ["Dataset sample", "Upload image"])

if input_mode == "Dataset sample":
    gallery = load_sample_gallery()
    sample_index = st.sidebar.slider(
        "Sample",
        min_value=0,
        max_value=len(gallery["labels"]) - 1,
        value=min(12, len(gallery["labels"]) - 1),
    )
    pixels, true_label, image = load_digit_sample(sample_index)

    st.image(image, caption=f"Sample {sample_index}", width=220, clamp=True)
    show_prediction(pixels, true_label=true_label)

else:
    uploaded_file = st.sidebar.file_uploader("Image", type=["png", "jpg", "jpeg"])
    if uploaded_file is None:
        st.info("Upload a high-contrast image of one digit.")
        st.stop()

    uploaded_image = Image.open(uploaded_file)
    bundle = get_model_bundle()
    pixels = preprocess_uploaded_image(
        uploaded_image,
        input_shape=model_input_shape(bundle),
        pixel_range=model_pixel_range(bundle),
    )

    st.image(uploaded_image, caption="Uploaded image", width=260)
    show_prediction(pixels)
