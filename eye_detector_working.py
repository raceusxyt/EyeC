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


def create_simple_predictor():
    """Create a minimal TensorFlow script that avoids threading issues"""
    script_content = '''#!/usr/bin/env python3
import sys
import os
import json

# Set all environment variables before any imports
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

try:
    # Import with minimal configuration
    import tensorflow as tf
    
    # Disable GPU and set single threading
    tf.config.set_visible_devices([], 'GPU')
    
    import numpy as np
    from PIL import Image
    
    def predict_eye_disease(model_path, image_path):
        """Simple prediction function"""
        try:
            # Load model with minimal configuration
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            
            # Load and process image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to expected input size
            image = image.resize((224, 224))
            
            # Convert to array and normalize
            img_array = np.array(image, dtype=np.float32)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Get model details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Run prediction
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            
            # Get output
            output = interpreter.get_tensor(output_details[0]['index'])
            
            # Apply softmax manually to avoid TF operations
            output_flat = output.flatten()
            exp_values = np.exp(output_flat - np.max(output_flat))
            probabilities = exp_values / np.sum(exp_values)
            
            return {"success": True, "probabilities": probabilities.tolist()}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    if __name__ == "__main__":
        if len(sys.argv) != 3:
            print(json.dumps({"success": False, "error": "Invalid arguments"}))
            sys.exit(1)
        
        model_path = sys.argv[1]
        image_path = sys.argv[2]
        
        result = predict_eye_disease(model_path, image_path)
        print(json.dumps(result))

except Exception as e:
    print(json.dumps({"success": False, "error": f"Import error: {str(e)}"}))
'''

    with open("simple_predictor.py", "w") as f:
        f.write(script_content)


def predict_with_simple_service(image):
    """Use the simple prediction service"""
    try:
        # Create predictor if needed
        if not os.path.exists("simple_predictor.py"):
            create_simple_predictor()

        # Save image to temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            image.save(tmp_file.name, "JPEG", quality=95)
            temp_path = tmp_file.name

        # Prepare clean environment
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": os.getcwd(),
            "KMP_DUPLICATE_LIB_OK": "True",
            "TF_CPP_MIN_LOG_LEVEL": "3",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }

        # Run in subprocess with timeout
        result = subprocess.run(
            [sys.executable, "simple_predictor.py", "model.tflite", temp_path],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
            cwd=os.getcwd(),
        )

        # Clean up temp file
        os.unlink(temp_path)

        if result.returncode == 0:
            try:
                response = json.loads(result.stdout.strip())
                if response.get("success"):
                    return np.array(response["probabilities"]), None
                else:
                    return None, response.get("error", "Unknown error")
            except json.JSONDecodeError:
                return None, f"JSON decode error. Output: {result.stdout}"
        else:
            return None, f"Process failed: {result.stderr}"

    except subprocess.TimeoutExpired:
        return None, "Prediction timed out"
    except Exception as e:
        return None, f"Service error: {str(e)}"


def main():
    st.set_page_config(
        page_title="Eye Disease Detector", page_icon="👁️", layout="centered"
    )

    st.title("👁️ Eye Disease Detection System")
    st.markdown("**AI-powered eye disease screening using deep learning**")

    # Check required files
    if not Path("model.tflite").exists():
        st.error("❌ Model file 'model.tflite' not found")
        st.stop()

    if not Path("labels (1).txt").exists():
        st.error("❌ Labels file 'labels (1).txt' not found")
        st.stop()

    # Load labels
    labels = load_labels()
    if not labels:
        st.error("❌ Could not load labels")
        st.stop()

    st.success("✅ System Ready for Analysis")

    # Show detection capabilities
    st.subheader("🎯 AI Detection Capabilities")
    cols = st.columns(len(labels))
    for i, label in enumerate(labels):
        with cols[i]:
            if label.lower() == "normal":
                st.success(f"✅ **{label.title()}**")
            else:
                st.info(f"⚠️ **{label.title()}**")

    st.markdown("---")

    # File uploader
    uploaded_file = st.file_uploader(
        "📁 Upload Eye Image for Analysis",
        type=["png", "jpg", "jpeg"],
        help="Select a clear, well-lit eye image",
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)

            # Display uploaded image
            st.subheader("📷 Uploaded Image")

            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(image, caption="Eye image for analysis", use_column_width=True)
            with col2:
                st.info(f"**Size:** {image.size[0]} × {image.size[1]}")
                st.info(f"**Format:** {image.format}")
                st.info(f"**Mode:** {image.mode}")

            # Analysis section
            st.markdown("---")
            st.subheader("🔬 AI Analysis")

            if st.button(
                "🚀 Run Disease Detection", type="primary", use_container_width=True
            ):
                with st.spinner("🤖 AI is analyzing the eye image..."):
                    # Run prediction
                    probabilities, error = predict_with_simple_service(image)

                    if error:
                        st.error(f"❌ Analysis Failed")
                        st.error(f"**Error:** {error}")

                        # Troubleshooting tips
                        with st.expander("🔧 Troubleshooting Tips"):
                            st.markdown(
                                """
                            **Common Solutions:**
                            1. Try a different image format (PNG instead of JPG)
                            2. Ensure image is clear and well-lit
                            3. Check that model files are not corrupted
                            4. Restart the application
                            
                            **Technical Issues:**
                            - This error is related to TensorFlow threading on macOS
                            - The app uses subprocess isolation to minimize conflicts
                            - Consider using a different Python environment
                            """
                            )

                    elif probabilities is not None:
                        # Success! Display results
                        st.success("✅ Analysis Complete!")

                        # Get top prediction
                        top_idx = np.argmax(probabilities)
                        top_label = labels[top_idx]
                        top_confidence = probabilities[top_idx]

                        # Main result with visual emphasis
                        st.markdown("### 🎯 Primary Detection Result")

                        if top_label.lower() == "normal":
                            st.success(f"✅ **HEALTHY EYE DETECTED**")
                            st.balloons()  # Celebration for healthy result
                        else:
                            st.warning(f"⚠️ **{top_label.upper()} DETECTED**")

                        # Confidence display
                        st.metric("Confidence Level", f"{top_confidence:.1%}")

                        # Detailed analysis
                        st.markdown("### 📊 Detailed Analysis Results")

                        # Create a nice table-like display
                        for i, (label, prob) in enumerate(zip(labels, probabilities)):
                            col1, col2, col3 = st.columns([3, 1, 2])

                            with col1:
                                if i == top_idx:
                                    st.write(f"🏆 **{label.title()}**")
                                else:
                                    st.write(f"**{label.title()}**")

                            with col2:
                                st.write(f"**{prob:.1%}**")

                            with col3:
                                st.progress(float(prob))

                        # Medical disclaimer
                        st.markdown("---")
                        st.error(
                            """
                        🚨 **IMPORTANT MEDICAL DISCLAIMER**
                        
                        This AI tool is for **EDUCATIONAL AND RESEARCH PURPOSES ONLY**.
                        
                        **⚠️ CRITICAL WARNINGS:**
                        - NOT a medical device or diagnostic tool
                        - Cannot replace professional medical examination
                        - May produce false positives or false negatives
                        - Should NOT influence medical decisions
                        
                        **👨‍⚕️ ALWAYS consult qualified healthcare professionals for:**
                        - Proper medical diagnosis
                        - Treatment recommendations
                        - Emergency eye conditions
                        
                        **The developers assume NO responsibility for medical decisions based on this tool.**
                        """
                        )

        except Exception as e:
            st.error(f"Error processing image: {e}")

    else:
        # Welcome and instructions
        st.info("👆 Upload an eye image above to begin AI-powered analysis")

        with st.expander("📖 How to Use This Tool", expanded=True):
            st.markdown(
                """
            ### 🚀 Quick Start Guide:
            1. **Upload Image**: Click the file uploader and select an eye image
            2. **Supported Formats**: PNG, JPG, JPEG files
            3. **Run Analysis**: Click "Run Disease Detection" button
            4. **View Results**: See AI predictions and confidence scores
            
            ### 💡 Tips for Best Results:
            - Use **clear, high-resolution** images
            - Ensure **good lighting** conditions
            - Keep the **eye centered** and in focus
            - Avoid **blurry or dark** images
            
            ### 🔬 About the AI Model:
            - **Technology**: TensorFlow Lite deep learning model
            - **Training**: Trained on medical eye image datasets
            - **Accuracy**: Designed for preliminary screening only
            - **Limitations**: Not suitable for medical diagnosis
            """
            )

        with st.expander("⚠️ Important Disclaimers"):
            st.markdown(
                """
            ### Medical Disclaimer
            This tool is **NOT** a medical device and should **NOT** be used for:
            - Medical diagnosis or treatment decisions
            - Emergency medical situations
            - Replacing professional medical care
            
            ### Technical Disclaimer
            - AI predictions may be inaccurate
            - False positives and negatives are possible
            - Results depend on image quality
            - Professional validation required
            """
            )


if __name__ == "__main__":
    main()
