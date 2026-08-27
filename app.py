import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="EcoVision AI",
    page_icon="♻️",
    layout="centered"
)


# --------------------------------------------------
# Application Title
# --------------------------------------------------

st.title("♻️ EcoVision AI")
st.subheader("AI-Powered Waste Classification")

st.write(
    "Upload an image of a waste item and EcoVision AI will "
    "predict its category using a MobileNetV2 deep learning model."
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "waste_classifier_mobilenetv2.keras"
    )


model = load_model()


# --------------------------------------------------
# Class Names
# --------------------------------------------------

class_names = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]


# --------------------------------------------------
# Image Upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a waste image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Resize image
    image_resized = image.resize((128, 128))

    # Convert image to NumPy array
    image_array = np.array(image_resized)

    # Normalize pixel values
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    predictions = model.predict(image_array, verbose=0)

    predicted_index = np.argmax(predictions[0])

    predicted_class = class_names[predicted_index]

    confidence = predictions[0][predicted_index] * 100


    # --------------------------------------------------
    # Display Result
    # --------------------------------------------------

    st.success(f"Prediction: {predicted_class.upper()}")

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )


    # --------------------------------------------------
    # Class Probabilities
    # --------------------------------------------------

    st.subheader("Prediction Probabilities")

    probabilities = predictions[0] * 100

    for class_name, probability in zip(
        class_names,
        probabilities
    ):
        st.write(
            f"**{class_name.capitalize()}**: "
            f"{probability:.2f}%"
        )

        st.progress(
            int(probability)
        )