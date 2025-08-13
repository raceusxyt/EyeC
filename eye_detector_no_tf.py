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


def create_isolated_predictor():
    """Create an isolated prediction script that runs in a separate process"""
    script_content = """
import sys
import os
import json
import warnings

# Completely isolate TensorFlow
warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Use fork to completely isolate the process
if hasattr(os, 'fork'):
    pid = os.fork()
    if pid == 0:  # Child process
        try:
            import tensorflow as tf
            import numpy as np
            from PIL import Image
            
            def run_prediction(model_path, image_path):
                # Load model
                interpreter = tf.lite.Interpreter(model_path=model_path)
                interpreter.allocate_tensors()
                
                # Load and preprocess image
                image = Image.open(image_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image = image.resize((224, 224))
                img_array = np.array(image, dtype=np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Predict
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                interpreter.set_tensor(input_details[0]['index'], img_array)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])
                
                # Softmax
                exp_output = np.exp(output[0] - np.max(output[0]))
                probabilities = exp_output / np.sum(exp_output)
                
                return probabilities.tolist()
            
            model_path = sys.argv[1]
            image_path = sys.argv[2]
            result = run_prediction(model_path, image_path)
            print(json.dumps({"probabilities": result}))
            
        except Exception as e:
            print(json.dumps({"error": str(e)}))
        
        os._exit(0)  # Exit child process
    else:
        # Parent process waits for child
        os.waitpid(pid, 0)
else:
    # Fallback for systems without fork
    try:
        import tensorflow as tf
        import numpy as np
        from PIL import Image
        
        model_path = sys.argv[1]
        image_path = sys.argv[2]
        
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((224, 224))
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        exp_output = np.exp(output[0] - np.max(output[0]))
        probabilities = exp_output / np.sum(exp_output)
        
        print(json.dumps({"probabilities": probabilities.tolist()}))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))
"""

    with open("isolated_predictor.py", "w") as f:
        f.write(script_content)


def predict_image(image):
    """Predict using isolated subprocess"""
    try:
        # Create predictor script if needed
        if not os.path.exists("isolated_predictor.py"):
            create_isolated_predictor()

        # Save image to temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            image.save(tmp_file.name, "JPEG", quality=95)
            temp_path = tmp_file.name

        # Run prediction in completely isolated process
        env = os.environ.copy()
        env.update(
            {
                "KMP_DUPLICATE_LIB_OK": "True",
                "TF_CPP_MIN_LOG_LEVEL": "3",
                "OMP_NUM_THREADS": "1",
            }
        )

        result = subprocess.run(
            ["python", "isolated_predictor.py", "model.tflite", temp_path],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )

        # Clean up
        os.unlink(temp_path)

        if result.returncode == 0:
            try:
                print("Result\n", result)
                response = json.loads(result.stdout.strip())
                if "error" in response:
                    return None, response["error"]
                return np.array(response["probabilities"]), None
            except json.JSONDecodeError:
                print("error here 2")
                return None, "Failed to parse prediction results"
        else:
            return None, f"Subprocess error: {result.stderr}"

    except Exception as e:
        return None, str(e)


def main():
    st.set_page_config(
        page_title="Eye Disease Detector", page_icon="👁️", layout="centered"
    )

    st.title("👁️ Eye Disease Detection System")
    st.markdown("**Professional AI-powered eye disease screening tool**")

    # Check files
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

    st.success("✅ AI System Ready")

    # Show capabilities
    st.subheader("🎯 Detection Capabilities")
    for i, label in enumerate(labels, 1):
        icon = "✅" if label.lower() == "normal" else "⚠️"
        st.write(f"{i}. {icon} **{label.title()}**")

    st.markdown("---")

    # File upload
    uploaded_file = st.file_uploader(
        "📁 Upload Eye Image",
        type=["png", "jpg", "jpeg"],
        help="Upload a clear eye image for AI analysis",
    )

    if uploaded_file:
        try:
            image = Image.open(uploaded_file)

            # Display image
            st.subheader("📷 Uploaded Image")
            st.image(image, caption="Eye image for analysis", width=400)
            st.info(f"**Image Size:** {image.size[0]} × {image.size[1]} pixels")

            # Analysis button
            if st.button(
                "🔍 Run AI Analysis", type="primary", use_container_width=True
            ):
                with st.spinner("🤖 AI is analyzing the image..."):
                    probabilities, error = predict_image(image)

                    if error:
                        st.error(f"❌ Analysis failed: {error}")
                        st.info(
                            "💡 Try uploading a different image or check the model files"
                        )
                    elif probabilities is not None:
                        # Success! Show results
                        st.success("✅ Analysis Complete!")

                        # Top prediction
                        top_idx = np.argmax(probabilities)
                        top_label = labels[top_idx]
                        top_confidence = probabilities[top_idx]

                        # Main result
                        st.markdown("### 🎯 Detection Result")

                        if top_label.lower() == "normal":
                            st.success(f"✅ **NORMAL EYE DETECTED**")
                            st.success(f"**Confidence: {top_confidence:.1%}**")
                        else:
                            st.warning(f"⚠️ **{top_label.upper()} DETECTED**")
                            st.warning(f"**Confidence: {top_confidence:.1%}**")

                        # Detailed breakdown
                        st.markdown("### 📈 Confidence Breakdown")

                        # Sort by confidence
                        sorted_indices = np.argsort(probabilities)[::-1]

                        for rank, idx in enumerate(sorted_indices, 1):
                            label = labels[idx]
                            prob = probabilities[idx]

                            if rank == 1:
                                st.metric(f"🥇 {label.title()}", f"{prob:.1%}")
                            elif rank == 2:
                                st.metric(f"🥈 {label.title()}", f"{prob:.1%}")
                            elif rank == 3:
                                st.metric(f"🥉 {label.title()}", f"{prob:.1%}")
                            else:
                                st.write(f"**{label.title()}:** {prob:.1%}")

                            st.progress(float(prob))

                        # Medical disclaimer
                        st.markdown("---")
                        st.error(
                            """
                        🚨 **CRITICAL MEDICAL DISCLAIMER**
                        
                        This AI screening tool is for **EDUCATIONAL PURPOSES ONLY** and is **NOT** 
                        intended for medical diagnosis or treatment decisions.
                        
                        **⚠️ IMPORTANT:**
                        - This tool cannot replace professional medical examination
                        - False positives and false negatives are possible
                        - Seek immediate medical attention for eye problems
                        - Consult an ophthalmologist for proper diagnosis
                        
                        **The developers assume no responsibility for medical decisions 
                        based on this tool's output.**
                        """
                        )

        except Exception as e:
            st.error(f"Error: {e}")

    else:
        # Welcome message and instructions
        st.markdown("### 🚀 Get Started")
        st.info("Upload an eye image above to begin AI-powered disease detection")

        with st.expander("ℹ️ About This Tool"):
            st.markdown(
                """
            This application uses advanced deep learning to screen for common eye diseases.
            
            **Technology:**
            - TensorFlow Lite neural network
            - Trained on medical eye image datasets
            - Optimized for real-time inference
            
            **Use Cases:**
            - Educational demonstrations
            - Research and development
            - Preliminary screening (NOT diagnosis)
            
            **Limitations:**
            - Not a medical device
            - Requires professional validation
            - May have false positives/negatives
            """
            )


if __name__ == "__main__":
    main()
