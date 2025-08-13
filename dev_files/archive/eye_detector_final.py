import streamlit as st
import numpy as np
from PIL import Image
import subprocess
import json
import tempfile
import os
from pathlib import Path


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


def predict_with_service(image):
    """Use standalone prediction service"""
    try:
        # Save image to temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            image.save(tmp_file.name, "JPEG")
            temp_image_path = tmp_file.name

        # Run prediction service
        result = subprocess.run(
            ["python", "prediction_service.py", "model.tflite", temp_image_path],
            capture_output=True,
            text=True,
            timeout=60,
        )

        # Clean up temp file
        os.unlink(temp_image_path)

        if result.returncode == 0:
            try:
                response = json.loads(result.stdout.strip())
                if "error" in response:
                    st.error(f"Prediction error: {response['error']}")
                    return None
                return np.array(response["probabilities"])
            except json.JSONDecodeError:
                print("error here 3")
                st.error("Failed to parse prediction results")
                return None
        else:
            st.error(f"Prediction service error: {result.stderr}")
            return None

    except subprocess.TimeoutExpired:
        st.error("Prediction timed out (60 seconds)")
        return None
    except Exception as e:
        st.error(f"Error running prediction service: {e}")
        return None


def main():
    st.set_page_config(
        page_title="Eye Disease Detector", page_icon="👁️", layout="centered"
    )

    st.title("👁️ Eye Disease Detection System")
    st.markdown("**AI-powered eye disease detection using deep learning**")

    # Check if required files exist
    model_exists = Path("model.tflite").exists()
    labels_exists = Path("labels (1).txt").exists()
    service_exists = Path("prediction_service.py").exists()

    if not all([model_exists, labels_exists, service_exists]):
        st.error("❌ Missing required files:")
        if not model_exists:
            st.write("- model.tflite")
        if not labels_exists:
            st.write("- labels (1).txt")
        if not service_exists:
            st.write("- prediction_service.py")
        st.stop()

    # Load labels
    labels = load_labels()
    if not labels:
        st.error("❌ Could not load labels")
        st.stop()

    st.success("✅ System ready!")

    # Show detectable conditions
    st.subheader("🔍 Detectable Eye Conditions:")
    cols = st.columns(len(labels))
    for i, label in enumerate(labels):
        with cols[i % len(cols)]:
            if label.lower() == "normal":
                st.success(f"✅ **{label.title()}**")
            else:
                st.info(f"⚠️ **{label.title()}**")

    st.markdown("---")

    # File uploader
    uploaded_file = st.file_uploader(
        "📁 Upload an eye image for analysis",
        type=["png", "jpg", "jpeg"],
        help="Select a clear, well-lit eye image",
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)

            # Display image and analysis side by side
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("📷 Uploaded Image")
                st.image(
                    image, caption="Eye image for analysis", use_container_width=True
                )

                # Image info
                st.info(f"**Size:** {image.size[0]} × {image.size[1]} pixels")
                st.info(f"**Format:** {image.format}")

            with col2:
                st.subheader("🔬 AI Analysis")

                if st.button(
                    "🚀 Analyze Image", type="primary", use_container_width=True
                ):
                    with st.spinner("🔍 Running AI analysis..."):
                        # Use prediction service
                        probabilities = predict_with_service(image)

                        if probabilities is not None:
                            # Find top prediction
                            top_idx = np.argmax(probabilities)
                            top_label = labels[top_idx]
                            top_confidence = probabilities[top_idx]

                            # Display main result with appropriate styling
                            st.markdown("### 🎯 Primary Detection")

                            if top_label.lower() == "normal":
                                st.success(f"✅ **{top_label.upper()}**")
                                st.success(f"**Confidence: {top_confidence:.1%}**")
                            else:
                                st.warning(f"⚠️ **{top_label.upper()}**")
                                st.warning(f"**Confidence: {top_confidence:.1%}**")

                            # Show detailed analysis
                            st.markdown("### 📊 Detailed Analysis")

                            # Create a dataframe-like display
                            for i, (label, prob) in enumerate(
                                zip(labels, probabilities)
                            ):
                                # Create columns for better layout
                                metric_col1, metric_col2 = st.columns([3, 1])

                                with metric_col1:
                                    if i == top_idx:
                                        st.write(f"🥇 **{label.title()}**")
                                    else:
                                        st.write(f"**{label.title()}**")

                                with metric_col2:
                                    st.write(f"**{prob:.1%}**")

                                # Progress bar
                                st.progress(float(prob))

                            # Medical disclaimer
                            st.markdown("---")
                            st.warning(
                                """
                            ⚠️ **Important Medical Disclaimer**
                            
                            This AI tool is designed for **educational and research purposes only**. 
                            It should **NOT** be used as a substitute for professional medical advice, 
                            diagnosis, or treatment. 
                            
                            **Always consult with qualified healthcare professionals** for proper 
                            medical evaluation and diagnosis of eye conditions.
                            """
                            )
                        else:
                            st.error(
                                "❌ Failed to analyze image. Please try again with a different image."
                            )

        except Exception as e:
            st.error(f"Error processing image: {e}")

    else:
        st.info("👆 Please upload an eye image to begin analysis")

        # Instructions and tips
        with st.expander("📋 How to Use This Tool", expanded=True):
            st.markdown(
                """
            ### Steps:
            1. **Upload Image**: Click the file uploader above and select an eye image
            2. **Supported Formats**: PNG, JPG, JPEG files
            3. **Click Analyze**: Press the "Analyze Image" button to run AI detection
            4. **View Results**: See the detected condition and confidence scores
            
            ### 💡 Tips for Best Results:
            - Use **clear, well-lit** images
            - Ensure the **eye is in focus** and centered
            - **Higher resolution** images generally work better
            - Avoid **blurry, dark, or low-quality** images
            - Take photos in **good lighting conditions**
            
            ### 🔬 About the AI Model:
            This system uses a TensorFlow Lite deep learning model trained to detect:
            - **Cataract** - Clouding of the eye's lens
            - **Crossed Eyes** - Eye misalignment (strabismus)
            - **Diabetic Retinopathy** - Diabetes-related retinal damage
            - **Glaucoma** - Optic nerve damage from eye pressure
            - **Normal** - Healthy eye condition
            """
            )


if __name__ == "__main__":
    main()
