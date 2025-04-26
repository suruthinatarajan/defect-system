import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load Pre-trained Model
@st.cache_resource
def load_model():
    st.write("Loading Model...")
    model = tf.keras.models.load_model("model/model.h5")
    st.write("Model Loaded Successfully!")
    return model

model = load_model()

# Define Class Labels
CLASS_NAMES = ['Defective', 'Non-Defective']

# Function to Process Image and Make Predictions
def preprocess_image(image):
    """Preprocess the image for model prediction."""
    image = image.resize((224, 224))  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict(image):
    """Predict if the image is defective or non-defective."""
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)

    # Debugging Logs
    st.write("Raw Model Output:", prediction)

    predicted_index = np.argmax(prediction)  # Get predicted class index
    confidence = np.max(prediction) * 100  # Get confidence score

    # Correctly map the prediction
    predicted_class = CLASS_NAMES[predicted_index]

    return predicted_class, confidence

# Streamlit UI
st.title("Defect Detection System")
st.write("Upload an image to check if it is defective or non-defective.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.write("Processing Image...")
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    st.write("Analyzing...")
    label, confidence = predict(image)

    st.write(f"Prediction: **{label}**")
    st.write(f"Confidence: **{confidence:.2f}%**")

    # Display results based on confidence
    if label == "Non-Defective" and confidence == 100.00:
        st.success("This product is Non-Defective!")
    else:
        # print("Non-defective")
        st.error("This product is Defective!")
