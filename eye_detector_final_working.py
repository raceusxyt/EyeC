import streamlit as st
import numpy as np
from PIL import Image
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


def simulate_prediction(image, labels):
    """
    Simulate AI prediction results when TensorFlow is unavailable.
    This provides a working demo of the interface and functionality.
    """
    # Create realistic-looking probabilities based on image characteristics
    # In a real scenario, this would be replaced by actual model inference

    # Analyze image properties to make simulation more realistic
    img_array = np.array(image.convert("RGB"))

    # Simple heuristics for demo (not actual medical analysis)
    brightness = np.mean(img_array)
    contrast = np.std(img_array)

    # Generate probabilities that seem realistic
    np.random.seed(int(brightness + contrast) % 1000)  # Deterministic based on image

    # Create base probabilities
    base_probs = np.random.dirichlet(np.ones(len(labels)) * 2.0)

    # Adjust based on simple image characteristics (for demo purposes only)
    if brightness > 150:  # Bright image - more likely to be normal
        normal_idx = next(
            (i for i, label in enumerate(labels) if "normal" in label.lower()), 0
        )
        base_probs[normal_idx] *= 1.5

    # Normalize
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

    # System status
    if model_exists:
        st.success("✅ Model file detected")
        st.info(
            "ℹ️ **Note**: Due to TensorFlow compatibility issues on macOS, this demo uses simulated predictions that demonstrate the interface functionality."
        )
    else:
        st.warning("⚠️ Model file not found - running in demo mode")

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
                    # Use simulation since TensorFlow has compatibility issues
                    probabilities = simulate_prediction(image, labels)

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

                    # Technical note
                    st.markdown("---")
                    st.info(
                        """
                    🔧 **Technical Note**: This demo uses simulated AI predictions due to TensorFlow 
                    compatibility issues on macOS. The interface and functionality demonstrate 
                    how the real AI model would work when properly configured.
                    """
                    )

                    # Medical disclaimer
                    st.warning(
                        """
                    ⚠️ **Medical Disclaimer**: This tool is for educational and demonstration 
                    purposes only. It should NOT be used for actual medical diagnosis or treatment 
                    decisions. Always consult qualified healthcare professionals for proper 
                    medical evaluation and diagnosis of eye conditions.
                    """
                    )

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
            - **Purpose**: Demonstrates AI eye disease detection interface
            - **Technology**: TensorFlow Lite deep learning model
            - **Status**: Simulation mode due to macOS compatibility issues
            - **Educational**: Shows how real AI medical screening works
            """
            )

        with st.expander("🔧 Technical Information"):
            st.markdown(
                """
            ### TensorFlow Compatibility Issue:
            This demo runs in simulation mode due to a known macOS issue:
            - **Error**: "mutex lock failed: Invalid argument"
            - **Cause**: TensorFlow threading conflicts on macOS
            - **Solution**: Simulated predictions for demonstration
            
            ### For Production Use:
            - Deploy on Linux servers for full TensorFlow support
            - Use Docker containers for consistent environments
            - Consider cloud-based inference APIs
            - Test on different operating systems
            
            ### Model Information:
            - **File**: `model.tflite` (TensorFlow Lite format)
            - **Input**: 224×224 RGB images
            - **Output**: 5-class probability distribution
            - **Classes**: Cataract, Crossed Eyes, Diabetic Retinopathy, Glaucoma, Normal
            """
            )


if __name__ == "__main__":
    main()
