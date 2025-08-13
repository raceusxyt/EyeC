import streamlit as st
import os
import sys
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Fix for macOS threading issues - must be set before importing tensorflow
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

try:
    import tensorflow as tf

    # Configure TensorFlow to use single thread
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
except ImportError:
    st.error(
        "TensorFlow is not installed. Please install it using: pip install tensorflow"
    )
    sys.exit(1)

import numpy as np
from PIL import Image


@st.cache_resource
def load_model():
    """Load the TensorFlow Lite model with error handling."""
    try:
        # Load the model with minimal threading
        interpreter = tf.lite.Interpreter(model_path="model.tflite", num_threads=1)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please ensure 'model.tflite' exists in the current directory.")
        return None


@st.cache_data
def load_labels():
    """Load the labels from the text file."""
    try:
        labels = []
        with open("labels (1).txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    # Remove number prefix if exists (e.g., "1 cataract" -> "cataract")
                    if " " in line and line.split(" ", 1)[0].isdigit():
                        labels.append(line.split(" ", 1)[1])
                    else:
                        labels.append(line)
        return labels
    except FileNotFoundError:
        st.error("Labels file 'labels (1).txt' not found.")
        return []
    except Exception as e:
        st.error(f"Error loading labels: {str(e)}")
        return []


def preprocess_image(image, target_size=(224, 224)):
    """Preprocess the image for model prediction."""
    try:
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize to model input size
        image = image.resize(target_size)

        # Convert to numpy array and normalize
        img_array = np.array(image, dtype=np.float32)
        img_array = img_array / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None


def predict(interpreter, image_array):
    """Make prediction using the TensorFlow Lite model."""
    try:
        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Set input tensor
        interpreter.set_tensor(input_details[0]["index"], image_array)

        # Run inference
        interpreter.invoke()

        # Get prediction
        prediction = interpreter.get_tensor(output_details[0]["index"])

        return prediction[0]
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None


def main():
    st.set_page_config(
        page_title="Eye Disease Detector",
        page_icon="👁️",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    st.title("👁️ Eye Disease Detection System")
    st.markdown("Upload an eye image to detect potential eye diseases using AI")

    # Load model and labels
    with st.spinner("Loading AI model..."):
        interpreter = load_model()
        labels = load_labels()

    if interpreter is None or not labels:
        st.error("Failed to load model or labels. Please check if the files exist.")
        st.stop()

    st.success("✅ Model loaded successfully!")

    # Display detectable conditions
    with st.expander("🔍 Detectable Conditions", expanded=False):
        for i, label in enumerate(labels, 1):
            st.write(f"{i}. **{label.title()}**")

    st.markdown("---")

    # File uploader
    uploaded_file = st.file_uploader(
        "📁 Choose an eye image...",
        type=["png", "jpg", "jpeg"],
        help="Upload a clear image of an eye for disease detection",
    )

    if uploaded_file is not None:
        try:
            # Display uploaded image
            image = Image.open(uploaded_file)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("📷 Uploaded Image")
                st.image(image, caption="Eye image for analysis", use_column_width=True)

                # Image info
                st.info(f"**Image size:** {image.size[0]} x {image.size[1]} pixels")

            with col2:
                st.subheader("🔬 Analysis Results")

                if st.button(
                    "🚀 Analyze Image", type="primary", use_container_width=True
                ):
                    with st.spinner("🔍 Analyzing image..."):
                        # Preprocess image
                        processed_image = preprocess_image(image)

                        if processed_image is not None:
                            # Make prediction
                            prediction = predict(interpreter, processed_image)

                            if prediction is not None:
                                # Apply softmax to get probabilities
                                probabilities = tf.nn.softmax(prediction).numpy()

                                # Find top prediction
                                predicted_class_index = np.argmax(probabilities)
                                predicted_class = labels[predicted_class_index]
                                confidence = probabilities[predicted_class_index]

                                # Display main result
                                if predicted_class.lower() == "normal":
                                    st.success(
                                        f"✅ **Result: {predicted_class.upper()}**"
                                    )
                                    st.success(f"**Confidence: {confidence:.1%}**")
                                else:
                                    st.warning(
                                        f"⚠️ **Detected: {predicted_class.upper()}**"
                                    )
                                    st.warning(f"**Confidence: {confidence:.1%}**")

                                # Show detailed results
                                st.markdown("### 📊 Detailed Analysis")

                                # Sort predictions by confidence
                                sorted_indices = np.argsort(probabilities)[::-1]

                                for i, idx in enumerate(sorted_indices):
                                    label = labels[idx]
                                    prob = probabilities[idx]

                                    # Color code based on probability
                                    if i == 0:  # Top prediction
                                        st.metric(
                                            label=f"🥇 {label.title()}",
                                            value=f"{prob:.1%}",
                                            delta=None,
                                        )
                                    else:
                                        st.write(f"**{label.title()}:** {prob:.1%}")

                                    # Progress bar
                                    st.progress(float(prob))

                                # Medical disclaimer
                                st.markdown("---")
                                st.warning(
                                    """
                                ⚠️ **Medical Disclaimer**: This AI tool is for educational and informational purposes only. 
                                It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. 
                                Always consult with a qualified healthcare provider for proper medical evaluation and diagnosis.
                                """
                                )

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

    else:
        st.info("👆 Please upload an eye image to get started.")

        # Instructions
        st.markdown("### 📋 How to Use")
        st.markdown(
            """
        1. **Upload Image**: Click the file uploader above and select a clear eye image
        2. **Supported Formats**: PNG, JPG, JPEG files
        3. **Image Quality**: Use well-lit, clear, high-resolution images for best results
        4. **Analysis**: Click 'Analyze Image' to get AI predictions
        5. **Results**: View the detected condition and confidence levels
        
        **💡 Tips for Best Results:**
        - Use images with good lighting
        - Ensure the eye is clearly visible and in focus
        - Avoid blurry or low-resolution images
        """
        )


if __name__ == "__main__":
    main()
