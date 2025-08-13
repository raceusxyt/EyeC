import streamlit as st
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Fix for macOS TensorFlow threading issues
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

import tensorflow as tf
import numpy as np
from PIL import Image


@st.cache_resource
def load_model():
    """Load the TensorFlow Lite model."""
    try:
        interpreter = tf.lite.Interpreter(model_path="model.tflite", num_threads=1)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def load_labels():
    """Load the labels from the text file."""
    try:
        with open("labels (1).txt", "r", encoding="utf-8") as f:
            labels = []
            for line in f:
                line = line.strip()
                if line:
                    # Remove the number prefix if it exists (e.g., "1 cataract" -> "cataract")
                    if " " in line and line.split(" ", 1)[0].isdigit():
                        labels.append(line.split(" ", 1)[1])
                    else:
                        labels.append(line)
    except UnicodeDecodeError:
        # Try with different encoding if utf-8 fails
        with open("labels (1).txt", "r", encoding="latin-1") as f:
            labels = []
            for line in f:
                line = line.strip()
                if line:
                    # Remove the number prefix if it exists
                    if " " in line and line.split(" ", 1)[0].isdigit():
                        labels.append(line.split(" ", 1)[1])
                    else:
                        labels.append(line)
    return labels


def preprocess_image(image):
    """Preprocess the image for model prediction."""
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize to model input size (typically 224x224 for many models)
    image = image.resize((224, 224))

    # Convert to numpy array and normalize
    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict(interpreter, image_array):
    """Make prediction using the TensorFlow Lite model."""
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set input tensor
    interpreter.set_tensor(input_details[0]["index"], image_array)

    # Run inference
    interpreter.invoke()

    # Get prediction
    prediction = interpreter.get_tensor(output_details[0]["index"])

    return prediction


def main():
    st.set_page_config(
        page_title="Eye Disease Detector", page_icon="=A", layout="centered"
    )

    st.title("=A Eye Disease Detector")
    st.write("Upload an eye image to detect potential eye diseases using AI")

    # Load model and labels
    interpreter = load_model()
    if interpreter is None:
        st.error(
            "Failed to load the TensorFlow Lite model. Please check if 'model.tflite' exists."
        )
        return

    try:
        labels = load_labels()
    except Exception as e:
        st.error(f"Error loading labels: {str(e)}")
        return

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an eye image...",
        type=["png", "jpg", "jpeg"],
        help="Upload a clear image of an eye for disease detection",
    )

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            st.image(image, caption="Uploaded eye image", use_column_width=True)

        with col2:
            st.subheader("Prediction Results")

            with st.spinner("Analyzing image..."):
                try:
                    # Preprocess image
                    processed_image = preprocess_image(image)

                    # Make prediction
                    prediction = predict(interpreter, processed_image)

                    # Get probabilities
                    probabilities = tf.nn.softmax(prediction[0]).numpy()

                    # Find top prediction
                    predicted_class_index = np.argmax(probabilities)
                    predicted_class = labels[predicted_class_index]
                    confidence = probabilities[predicted_class_index]

                    # Display results
                    st.success(f"**Prediction:** {predicted_class}")
                    st.info(f"**Confidence:** {confidence:.2%}")

                    # Show all probabilities
                    st.subheader("All Predictions")
                    for i, (label, prob) in enumerate(zip(labels, probabilities)):
                        st.write(f"{label}: {prob:.2%}")
                        st.progress(prob)

                    # Add medical disclaimer
                    st.warning(
                        """
                    � **Medical Disclaimer**: This AI model is for educational purposes only. 
                    It should not be used as a substitute for professional medical diagnosis. 
                    Always consult with a qualified healthcare professional for proper medical advice.
                    """
                    )

                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")

    else:
        st.info("Please upload an eye image to get started.")

        # Show example of expected image types
        st.subheader("9 How to use:")
        st.write(
            """
        1. Click 'Browse files' above
        2. Select a clear image of an eye (PNG, JPG, or JPEG format)
        3. Wait for the AI to analyze the image
        4. View the prediction results and confidence scores
        
        **Detectable conditions:**
        - Cataract
        - Crossed eyes
        - Diabetic retinopathy
        - Glaucoma
        - Normal (healthy eye)
        """
        )


if __name__ == "__main__":
    main()
