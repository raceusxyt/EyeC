import streamlit as st
import numpy as np
from PIL import Image
import os

# Set environment variables to prevent threading issues
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Import TensorFlow with error handling
try:
    import tensorflow as tf

    tf_available = True
except Exception as e:
    tf_available = False
    st.error(f"TensorFlow import error: {e}")


def load_labels():
    """Load labels from file"""
    labels = []
    try:
        with open("labels (1).txt", "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    # Remove number prefix (e.g., "1 cataract" -> "cataract")
                    parts = line.split(" ", 1)
                    if len(parts) > 1 and parts[0].isdigit():
                        labels.append(parts[1])
                    else:
                        labels.append(line)
    except Exception as e:
        st.error(f"Error loading labels: {e}")
    return labels


def load_tflite_model():
    """Load TensorFlow Lite model"""
    if not tf_available:
        return None

    try:
        interpreter = tf.lite.Interpreter(model_path="model.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None


def preprocess_image(image):
    """Preprocess image for model"""
    # Convert to RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize to 224x224
    image = image.resize((224, 224))

    # Convert to numpy array
    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    return img_array


def predict(interpreter, image_array):
    """Make prediction"""
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]["index"], image_array)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]["index"])
        return output[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


# Streamlit App
st.title("👁️ Eye Disease Detector")
st.write("Upload an eye image for AI analysis")

# Load model and labels
labels = load_labels()
interpreter = load_tflite_model()

if not labels:
    st.error("Could not load labels file")
    st.stop()

if not interpreter:
    st.error("Could not load TensorFlow Lite model")
    st.stop()

st.success("✅ Model and labels loaded successfully!")

# Show detectable conditions
st.subheader("Detectable Conditions:")
for i, label in enumerate(labels, 1):
    st.write(f"{i}. {label.title()}")

# File uploader
uploaded_file = st.file_uploader("Choose an eye image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    # Analyze button
    if st.button("Analyze Image"):
        with st.spinner("Analyzing..."):
            # Preprocess
            processed_image = preprocess_image(image)

            # Predict
            prediction = predict(interpreter, processed_image)

            if prediction is not None:
                # Get probabilities using softmax
                exp_pred = np.exp(prediction - np.max(prediction))
                probabilities = exp_pred / np.sum(exp_pred)

                # Top prediction
                top_idx = np.argmax(probabilities)
                top_label = labels[top_idx]
                top_confidence = probabilities[top_idx]

                # Display result
                st.subheader("🔬 Analysis Result")

                if top_label.lower() == "normal":
                    st.success(f"✅ **{top_label.upper()}**")
                else:
                    st.warning(f"⚠️ **{top_label.upper()}**")

                st.write(f"**Confidence:** {top_confidence:.1%}")

                # Show all predictions
                st.subheader("📊 All Predictions")
                for label, prob in zip(labels, probabilities):
                    st.write(f"**{label.title()}:** {prob:.1%}")
                    st.progress(float(prob))

                # Medical disclaimer
                st.warning(
                    "⚠️ **Disclaimer:** For educational purposes only. Not for medical diagnosis."
                )

else:
    st.info("Please upload an eye image to begin analysis")
