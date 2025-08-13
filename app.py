import streamlit as st
import numpy as np
from PIL import Image
import os
from pathlib import Path

# TensorFlow will be imported dynamically to handle compatibility issues
tensorflow_available = None
tf = None


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


def check_tensorflow():
    """Check if TensorFlow can be imported safely"""
    global tensorflow_available, tf
    
    if tensorflow_available is None:
        try:
            # Suppress TensorFlow logging
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            import tensorflow as tf_module
            tf = tf_module
            tensorflow_available = True
        except Exception as e:
            tensorflow_available = False
            return False, str(e)
    
    return tensorflow_available, None


@st.cache_resource
def load_model():
    """Load the TensorFlow Lite model"""
    tf_available, error = check_tensorflow()
    if not tf_available:
        return None, f"TensorFlow not available: {error}"
    
    try:
        interpreter = tf.lite.Interpreter(model_path="model.tflite")
        interpreter.allocate_tensors()
        return interpreter, None
    except Exception as e:
        error_msg = str(e)
        if "mutex lock failed" in error_msg or "Invalid argument" in error_msg:
            return None, "macOS TensorFlow compatibility issue"
        else:
            return None, f"Model loading error: {error_msg}"


def preprocess_image(image):
    """Preprocess image for model input"""
    # Resize to 224x224 as expected by the model
    image = image.convert('RGB')
    image = image.resize((224, 224))
    
    # Convert to numpy array and normalize
    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def predict_disease(image, interpreter, labels):
    """Make prediction using the TensorFlow Lite model"""
    if interpreter is None:
        return None
        
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        
        # Run inference
        interpreter.invoke()
        
        # Get the output
        predictions = interpreter.get_tensor(output_details[0]['index'])
        
        # Apply softmax if needed (some models output raw logits)
        if tensorflow_available:
            probabilities = tf.nn.softmax(predictions[0]).numpy()
        else:
            # Manual softmax implementation
            exp_predictions = np.exp(predictions[0] - np.max(predictions[0]))
            probabilities = exp_predictions / np.sum(exp_predictions)
        
        return probabilities
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None


def fallback_prediction(image, labels):
    """
    Fallback prediction when TensorFlow fails (e.g., on macOS).
    Uses image analysis to provide reasonable demonstrations.
    """
    # Analyze image properties
    img_array = np.array(image.convert("RGB"))
    
    # Basic image characteristics for demo purposes
    brightness = np.mean(img_array)
    contrast = np.std(img_array)
    red_intensity = np.mean(img_array[:,:,0])
    
    # Create deterministic probabilities based on image characteristics
    np.random.seed(int(brightness + contrast + red_intensity) % 1000)
    
    # Generate base probabilities
    base_probs = np.random.dirichlet(np.ones(len(labels)) * 3.0)
    
    # Simple heuristics for demo (not medical analysis)
    for i, label in enumerate(labels):
        if "normal" in label.lower():
            if brightness > 140:  # Brighter images more likely normal
                base_probs[i] *= 1.3
        elif "cataract" in label.lower():
            if brightness < 100:  # Darker images might suggest cataract
                base_probs[i] *= 1.2
    
    # Normalize probabilities
    base_probs = base_probs / np.sum(base_probs)
    
    return base_probs


def main():
    st.set_page_config(
        page_title="Eye Disease Detector", page_icon="👁️", layout="centered"
    )

    st.title("👁️ Eye Disease Detection System")
    st.markdown("**AI-powered eye disease screening demonstration**")

    # Check required files
    model_exists = Path("model.tflite").exists()
    labels_exist = Path("labels (1).txt").exists()

    if not labels_exist:
        st.error("❌ Labels file 'labels (1).txt' not found")
        st.stop()

    # Load labels
    labels = load_labels()
    if not labels:
        st.error("❌ Could not load labels")
        st.stop()

    # Load model
    interpreter = None
    use_fallback = False
    if model_exists:
        interpreter, error = load_model()
        if interpreter is not None:
            st.success("✅ TensorFlow model loaded successfully")
        elif "macOS TensorFlow compatibility issue" in str(error) or "TensorFlow not available" in str(error):
            st.warning("⚠️ TensorFlow compatibility issue detected")
            st.info("🔧 Using fallback prediction system for demonstration")
            use_fallback = True
        else:
            st.error(f"❌ Failed to load model: {error}")
            st.stop()
    else:
        st.error("❌ Model file not found")
        st.stop()

    # Show detection capabilities
    st.subheader("🎯 AI Detection Capabilities")

    cols = st.columns(len(labels))
    for i, label in enumerate(labels):
        with cols[i]:
            if label.lower() == "normal":
                st.success(f"✅ **{label.title()}**")
            else:
                st.info(f"🔍 **{label.title()}**")

    st.markdown("---")

    # File uploader
    uploaded_file = st.file_uploader(
        "📁 Upload Eye Image for Analysis",
        type=["png", "jpg", "jpeg"],
        help="Select a clear, well-lit eye image for analysis",
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)

            # Display uploaded image
            st.subheader("📷 Uploaded Image")

            col1, col2 = st.columns([2, 1])

            with col1:
                st.image(
                    image, caption="Eye image for analysis", use_container_width=True
                )

            with col2:
                st.info(f"**Size:** {image.size[0]} × {image.size[1]} pixels")
                st.info(f"**Format:** {image.format}")
                st.info(f"**Mode:** {image.mode}")

            # Analysis section
            st.markdown("---")
            st.subheader("🔬 AI Analysis")

            if st.button(
                "🚀 Run Disease Detection", type="primary", use_container_width=True
            ):
                with st.spinner("🤖 Analyzing eye image..."):
                    if use_fallback:
                        # Use fallback prediction for compatibility
                        probabilities = fallback_prediction(image, labels)
                    else:
                        # Use real TensorFlow model for prediction
                        probabilities = predict_disease(image, interpreter, labels)

                if probabilities is not None:
                    # Display results
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

                    # Confidence metric
                    st.metric("Confidence Level", f"{top_confidence:.1%}")

                    # Detailed analysis
                    st.markdown("### 📊 Detailed Analysis Results")

                    # Sort predictions by confidence
                    sorted_indices = np.argsort(probabilities)[::-1]

                    for rank, idx in enumerate(sorted_indices, 1):
                        label = labels[idx]
                        prob = probabilities[idx]

                        col1, col2, col3 = st.columns([1, 3, 2])

                        with col1:
                            if rank == 1:
                                st.write("🥇")
                            elif rank == 2:
                                st.write("🥈")
                            elif rank == 3:
                                st.write("🥉")
                            else:
                                st.write(f"{rank}.")

                        with col2:
                            st.write(f"**{label.title()}**")

                        with col3:
                            st.write(f"**{prob:.1%}**")

                        # Progress bar
                        st.progress(float(prob))

                    # Technical and Medical disclaimers
                    st.markdown("---")
                    if use_fallback:
                        st.info(
                            """
                        🔧 **Technical Note**: Results shown are from a fallback prediction system 
                        due to TensorFlow compatibility issues on macOS. This demonstrates the 
                        interface functionality but does not use the actual trained model.
                        """
                        )
                    
                    st.warning(
                        """
                    ⚠️ **Medical Disclaimer**: This tool is for educational and demonstration 
                    purposes only. It should NOT be used for actual medical diagnosis or treatment 
                    decisions. Always consult qualified healthcare professionals for proper 
                    medical evaluation and diagnosis of eye conditions.
                    """
                    )
                else:
                    st.error("❌ Failed to analyze the image. Please try again with a different image.")

        except Exception as e:
            st.error(f"Error processing image: {e}")

    else:
        # Welcome and instructions
        st.info("👆 Upload an eye image above to begin AI analysis")

        with st.expander("📖 How to Use This Demo", expanded=True):
            st.markdown(
                """
            ### 🚀 Quick Start:
            1. **Upload Image**: Click the file uploader and select an eye image
            2. **Supported Formats**: PNG, JPG, JPEG files
            3. **Run Analysis**: Click "Run Disease Detection" button
            4. **View Results**: See AI predictions and confidence scores
            
            ### 💡 Image Guidelines:
            - Use **clear, high-resolution** images
            - Ensure **good lighting** conditions
            - Keep the **eye centered** and in focus
            - Avoid **blurry, dark, or low-quality** images
            
            ### 🔬 About This Demo:
            - **Purpose**: AI-powered eye disease detection system
            - **Technology**: TensorFlow Lite deep learning model
            - **Status**: Attempts real AI model, fallback on compatibility issues
            - **Educational**: Demonstrates AI medical screening workflow
            """
            )

        with st.expander("🔧 Technical Information"):
            st.markdown(
                """
            ### Model Architecture:
            - **Framework**: TensorFlow Lite for optimized inference
            - **Input Format**: 224×224 RGB images
            - **Preprocessing**: Normalization to [0, 1] range
            - **Output**: Probability distribution across disease classes
            
            ### Supported Conditions:
            - **Cataract**: Clouding of the eye's lens
            - **Crossed Eyes**: Misalignment of the eyes
            - **Diabetic Retinopathy**: Diabetes-related eye damage
            - **Glaucoma**: Optic nerve damage
            - **Normal**: Healthy eye condition
            
            ### Performance Considerations:
            - **Model Size**: Optimized TensorFlow Lite format
            - **Inference Speed**: Fast local processing
            - **Memory Usage**: Efficient resource utilization
            - **Platform Support**: Cross-platform compatibility
            """
            )


if __name__ == "__main__":
    main()
