import streamlit as st
import os
import sys
import warnings

# Set environment variables before any imports
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Suppress all warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image

# Try to import TensorFlow with error handling
try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    st.error("TensorFlow not available. Please install: pip install tensorflow")


@st.cache_resource
def load_model():
    """Load the TensorFlow Lite model."""
    if not TF_AVAILABLE:
        return None

    try:
        # Simple model loading without threading configuration
        interpreter = tf.lite.Interpreter(model_path="model.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None


@st.cache_data
def load_labels():
    """Load labels from file."""
    try:
        labels = []
        with open("labels (1).txt", "r") as f:
            for line in f:
                line = line.strip()
                if line and " " in line:
                    # Extract label after number (e.g., "1 cataract" -> "cataract")
                    parts = line.split(" ", 1)
                    if parts[0].isdigit():
                        labels.append(parts[1])
                    else:
                        labels.append(line)
                elif line:
                    labels.append(line)
        return labels
    except Exception as e:
        st.error(f"Error loading labels: {e}")
        return []


def preprocess_image(image):
    """Preprocess image for model input."""
    try:
        # Convert to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize to 224x224 (common input size)
        image = image.resize((224, 224))

        # Convert to array and normalize
        img_array = np.array(image, dtype=np.float32) / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array
    except Exception as e:
        st.error(f"Image preprocessing error: {e}")
        return None


def predict_image(interpreter, image_array):
    """Make prediction using TensorFlow Lite model."""
    try:
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Set input
        interpreter.set_tensor(input_details[0]["index"], image_array)

        # Run inference
        interpreter.invoke()

        # Get output
        output = interpreter.get_tensor(output_details[0]["index"])
        return output[0]

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


def main():
    st.set_page_config(page_title="Eye Disease Detector", page_icon="👁️", layout="wide")

    st.title("👁️ Eye Disease Detection System")
    st.markdown("**AI-powered eye disease detection using deep learning**")

    # Load resources
    interpreter = load_model()
    labels = load_labels()

    if not interpreter or not labels:
        st.error("❌ Failed to load model or labels. Check if files exist:")
        st.code("- model.tflite\n- labels (1).txt")
        return

    st.success("✅ Model loaded successfully!")

    # Show detectable conditions
    st.subheader("🔍 Detectable Conditions:")
    cols = st.columns(len(labels))
    for i, label in enumerate(labels):
        with cols[i % len(cols)]:
            st.info(f"**{label.title()}**")

    st.markdown("---")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload an eye image",
        type=["png", "jpg", "jpeg"],
        help="Select a clear eye image for analysis",
    )

    if uploaded_file:
        try:
            image = Image.open(uploaded_file)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("📸 Input Image")
                st.image(image, caption="Uploaded image", use_column_width=True)
                st.info(f"Size: {image.size[0]} × {image.size[1]} pixels")

            with col2:
                st.subheader("🔬 Analysis")

                if st.button("🚀 Analyze Image", type="primary"):
                    with st.spinner("Analyzing..."):
                        # Preprocess
                        processed = preprocess_image(image)

                        if processed is not None:
                            # Predict
                            prediction = predict_image(interpreter, processed)

                            if prediction is not None:
                                # Get probabilities
                                if TF_AVAILABLE:
                                    probs = tf.nn.softmax(prediction).numpy()
                                else:
                                    # Manual softmax if TF not available
                                    exp_pred = np.exp(prediction - np.max(prediction))
                                    probs = exp_pred / np.sum(exp_pred)

                                # Top prediction
                                top_idx = np.argmax(probs)
                                top_label = labels[top_idx]
                                top_conf = probs[top_idx]

                                # Display result
                                if top_label.lower() == "normal":
                                    st.success(f"✅ **{top_label.upper()}**")
                                else:
                                    st.warning(f"⚠️ **{top_label.upper()}**")

                                st.metric("Confidence", f"{top_conf:.1%}")

                                # All predictions
                                st.subheader("📊 All Predictions")
                                for label, prob in zip(labels, probs):
                                    st.write(f"**{label.title()}**: {prob:.1%}")
                                    st.progress(float(prob))

                                # Disclaimer
                                st.warning(
                                    """
                                ⚠️ **Disclaimer**: This is for educational purposes only. 
                                Not a substitute for professional medical diagnosis.
                                """
                                )

        except Exception as e:
            st.error(f"Error: {e}")

    else:
        st.info("👆 Upload an eye image to begin analysis")

        with st.expander("📋 Instructions"):
            st.markdown(
                """
            1. **Upload**: Select a clear eye image (PNG/JPG/JPEG)
            2. **Analyze**: Click the analyze button
            3. **Results**: View predictions and confidence scores
            
            **Tips**:
            - Use well-lit, clear images
            - Ensure the eye is in focus
            - Higher resolution images work better
            """
            )


if __name__ == "__main__":
    main()
