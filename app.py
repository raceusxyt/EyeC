import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import os
import time
import io

# -----------------------------
# Splash Screen (Logo once at start)
# -----------------------------
if "splash_shown" not in st.session_state:
    st.session_state.splash_shown = False

if not st.session_state.splash_shown:
    if os.path.exists("logo.png"):
        st.image("logo.png")  # show logo splash
    time.sleep(2)  # show splash for 2 seconds
    st.session_state.splash_shown = True
    st.rerun()
    
# -----------------------------
# Load TFLite Model
# -----------------------------
@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -----------------------------
# App UI (no duplicate logo)
# -----------------------------
st.title("👁 EyeC - Early Detection, Healthy Vision")

# -----------------------------
# Image Input (Upload or Camera)
# -----------------------------
st.markdown("### Choose how you want to provide an eye image:")

option = st.radio("Select Input Method", ["📁 Upload", "📷 Camera"])
image = None

if option == "📁 Upload":
    uploaded_file = st.file_uploader("Upload a fundus eye image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            if uploaded_file.size > 10 * 1024 * 1024:  # 10MB limit
                st.error("❌ File too large! Please upload an image under 10MB.")
            else:
                file_bytes = uploaded_file.read()
                image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                st.success(f"✅ Uploaded: {uploaded_file.name} ({uploaded_file.size/1024:.1f} KB)")
        except Exception as e:
            st.error(f"Error processing uploaded image: {e}")

elif option == "📷 Camera":
    camera_image = st.camera_input("Take a photo with your camera")
    if camera_image is not None:
        try:
            image = Image.open(io.BytesIO(camera_image.getvalue())).convert("RGB")
            st.success("✅ Camera image captured")
        except Exception as e:
            st.error(f"Error processing camera image: {e}")

# -----------------------------
# Prediction
# -----------------------------
if image is not None:
    try:
        st.markdown("### Preview:")
        st.image(image)

        # Preprocess (resize to model input)
        img_resized = image.resize((224, 224))  # adjust if needed
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Run model
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        # Example labels (update if you have different)
        labels = ["normal", "cataract", "glaucoma", "diabetic retinopathy"]

        pred_idx = int(np.argmax(predictions))
        pred_label = labels[pred_idx]
        pred_confidence = float(np.max(predictions))

        st.markdown("### 🔍 Prediction:")

        if pred_label == "normal":
            st.success(f"🎉 Congratulations! Your eyes seem healthy ({pred_confidence*100:.2f}%)")
        else:
            st.warning(f"Prediction: *{pred_label.capitalize()}* ({pred_confidence*100:.2f}%)")

        # Example medical guidance dictionary
        disease_info = {
                "cataract": {
                    "guidance": (
                        "Clouding of the eye’s lens may cause blurry vision and light sensitivity. "
                        "Protect your eyes from UV rays with sunglasses, eat antioxidant-rich foods, "
                        "and manage conditions like diabetes and hypertension."),
                    "consult": "Consult an ophthalmologist to evaluate cataract progression and discuss surgery if vision is affected."},
                "glaucoma": {
                    "guidance": (
                        "Glaucoma can damage the optic nerve if untreated. "
                        "Use prescribed eye drops regularly, avoid smoking, and limit caffeine. "
                        "Light exercise and healthy diet can help maintain eye pressure."),
                    "consult": "Seek medical consultation to monitor eye pressure and prevent irreversible vision loss."},
                "diabetic retinopathy": {
                    "guidance": (
                        "Caused by diabetes affecting retinal blood vessels. "
                        "Maintain strict blood sugar control, eat a balanced diet, and exercise regularly. "
                        "Avoid smoking and keep blood pressure and cholesterol under control."),
                    "consult": "Schedule regular eye check-ups with an ophthalmologist for early detection and laser/surgical treatment if needed."}}

        if pred_label in disease_info:
            st.markdown(f"**Medical Guidance:** {disease_info[pred_label]['guidance']}")
            st.markdown(f"**Consultation Advice:** {disease_info[pred_label]['consult']}")
        else:
            st.info("General Eye Care Tips:\n"
                    "- Maintain a healthy diet (Vitamin A, leafy greens, omega-3s).\n"
                    "- Follow the 20-20-20 rule: Every 20 minutes, look at something 20 feet away for 20 seconds.\n"
                    "- Get regular eye checkups (every 6–12 months).\n"
                    "- Protect your eyes from screens and UV light (use sunglasses).\n"
                    "- Stay hydrated and get proper sleep.")

    except Exception as e:
        st.error(f"Error processing image: {e}")
