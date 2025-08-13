import streamlit as st
import numpy as np
from PIL import Image
import subprocess
import json
import tempfile
import os


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


def create_prediction_script():
    """Create a separate Python script for TensorFlow operations"""
    script_content = """
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import json

def predict_image(model_path, image_path):
    try:
        # Load model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Load and preprocess image
        from PIL import Image
        image = Image.open(image_path)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize((224, 224))
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Make prediction
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        
        output = interpreter.get_tensor(output_details[0]['index'])
        
        # Apply softmax
        exp_output = np.exp(output[0] - np.max(output[0]))
        probabilities = exp_output / np.sum(exp_output)
        
        return probabilities.tolist()
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    
    result = predict_image(model_path, image_path)
    print(json.dumps(result))
"""

    with open("predict_helper.py", "w") as f:
        f.write(script_content)


def predict_with_subprocess(image):
    """Use subprocess to run TensorFlow prediction"""
    try:
        # Create prediction script if it doesn't exist
        if not os.path.exists("predict_helper.py"):
            create_prediction_script()

        # Save image to temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            image.save(tmp_file.name, "JPEG")
            temp_image_path = tmp_file.name

        # Run prediction in subprocess
        result = subprocess.run(
            ["python", "predict_helper.py", "model.tflite", temp_image_path],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Clean up temp file
        os.unlink(temp_image_path)

        if result.returncode == 0:
            try:
                probabilities = json.loads(result.stdout.strip())
                if isinstance(probabilities, dict) and "error" in probabilities:
                    st.error(f"Prediction error: {probabilities['error']}")
                    return None
                return np.array(probabilities)
            except json.JSONDecodeError:
                print("error here 1")
                st.error("Failed to parse prediction results")
                return None
        else:
            st.error(f"Subprocess error: {result.stderr}")
            return None

    except subprocess.TimeoutExpired:
        st.error("Prediction timed out")
        return None
    except Exception as e:
        st.error(f"Subprocess prediction error: {e}")
        return None


def main():
    st.set_page_config(
        page_title="Eye Disease Detector", page_icon="👁️", layout="centered"
    )

    st.title("👁️ Eye Disease Detection System")
    st.markdown("**AI-powered eye disease detection using TensorFlow Lite**")

    # Load labels
    labels = load_labels()

    if not labels:
        st.error("❌ Could not load labels file")
        st.stop()

    # Check if model file exists
    if not os.path.exists("model.tflite"):
        st.error("❌ Model file 'model.tflite' not found")
        st.stop()

    st.success("✅ Files loaded successfully!")

    # Show detectable conditions
    st.subheader("🔍 Detectable Conditions:")
    for i, label in enumerate(labels, 1):
        st.write(f"{i}. **{label.title()}**")

    st.markdown("---")

    # File uploader
    uploaded_file = st.file_uploader(
        "📁 Upload an eye image",
        type=["png", "jpg", "jpeg"],
        help="Select a clear eye image for analysis",
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("📷 Input Image")
                st.image(image, caption="Uploaded image", use_column_width=True)
                st.info(f"**Size:** {image.size[0]} × {image.size[1]} pixels")

            with col2:
                st.subheader("🔬 Analysis")

                if st.button("🚀 Analyze Image", type="primary"):
                    with st.spinner("🔍 Running AI analysis..."):
                        # Use subprocess for prediction
                        probabilities = predict_with_subprocess(image)

                        if probabilities is not None:
                            # Find top prediction
                            top_idx = np.argmax(probabilities)
                            top_label = labels[top_idx]
                            top_confidence = probabilities[top_idx]

                            # Display main result
                            if top_label.lower() == "normal":
                                st.success(f"✅ **Result: {top_label.upper()}**")
                            else:
                                st.warning(f"⚠️ **Detected: {top_label.upper()}**")

                            st.metric("Confidence", f"{top_confidence:.1%}")

                            # Show all predictions
                            st.subheader("📊 Detailed Results")

                            # Sort by confidence
                            sorted_indices = np.argsort(probabilities)[::-1]

                            for i, idx in enumerate(sorted_indices):
                                label = labels[idx]
                                prob = probabilities[idx]

                                if i == 0:  # Top prediction
                                    st.write(f"🥇 **{label.title()}**: {prob:.1%}")
                                else:
                                    st.write(f"**{label.title()}**: {prob:.1%}")

                                st.progress(float(prob))

                            # Medical disclaimer
                            st.markdown("---")
                            st.warning(
                                """
                            ⚠️ **Medical Disclaimer**: This AI tool is for educational purposes only. 
                            It should NOT be used as a substitute for professional medical advice, 
                            diagnosis, or treatment. Always consult with qualified healthcare 
                            professionals for proper medical evaluation.
                            """
                            )

        except Exception as e:
            st.error(f"Error processing image: {e}")

    else:
        st.info("👆 Please upload an eye image to begin analysis")

        with st.expander("📋 How to Use"):
            st.markdown(
                """
            1. **Upload**: Click above to select an eye image (PNG/JPG/JPEG)
            2. **Analyze**: Click the "Analyze Image" button
            3. **Results**: View the AI predictions and confidence scores
            
            **Tips for Best Results:**
            - Use clear, well-lit images
            - Ensure the eye is in focus
            - Higher resolution images work better
            - Avoid blurry or dark images
            """
            )


if __name__ == "__main__":
    main()
