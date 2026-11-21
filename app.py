"""Streamlit demo: upload an infrared image of a solar module, or try a sample.

Run with:  streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from solarfault.predict import classify, load_model

SAMPLES = Path(__file__).parent / "samples"

st.set_page_config(page_title="Solar module fault detection", layout="centered")
st.title("Solar module fault detection")
st.write(
    "A convolutional neural network that classifies 40 x 24 pixel infrared images of solar "
    "modules into 11 fault types or *No-Anomaly*. Trained on the public "
    "[InfraredSolarModules](https://github.com/RaptorMaps/InfraredSolarModules) dataset."
)


@st.cache_resource
def model():
    return load_model()


samples = sorted(SAMPLES.glob("*.jpg"))
choice = st.selectbox("Try a sample image", ["(upload your own)"] + [p.stem for p in samples])
uploaded = st.file_uploader("...or upload an infrared module image", type=["jpg", "jpeg", "png"])

image = None
if uploaded is not None:
    image = Image.open(uploaded)
elif choice != "(upload your own)":
    image = Image.open(SAMPLES / f"{choice}.jpg")

if image is not None:
    probs = classify(model(), image)
    top, p = next(iter(probs.items()))
    left, right = st.columns([1, 2])
    left.image(image.convert("L").resize((96, 160), Image.NEAREST), caption="Input (enlarged)")
    right.metric("Prediction", top, f"{p:.0%} confidence", delta_color="off")
    right.bar_chart(pd.Series(probs).head(5), horizontal=True)
    if choice != "(upload your own)" and uploaded is None:
        st.caption(f"True class of this sample: {choice.split('__')[0]}")
