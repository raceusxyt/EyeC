import streamlit as st
from PIL import Image
import numpy as np
import time
import os

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="EyeC - Eye Disease Detection",
    page_icon="👁",
    layout="centered"
)

# -----------------------------
# CSS Styling (theme + animation)
# -----------------------------
st.markdown(
    """
    <style>
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    .fade-in {
        animation: fadeIn 1.5s ease-in-out;
    }
    .stApp {
        background-color: #FFF9C4; /* light yellow */
        color: #0D47A1; /* dark blue */
    }
    .stButton>button {
        background-color: #0D47A1;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .splash {
        text-align: center;
        padding-top: 20%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Splash Screen
# -----------------------------
if "splash_shown" not in st.session_state:
    st.session_state.splash_shown = False

if not st.session_state.splash_shown:
    st.markdown("<div class='splash fade-in'>", unsafe_allow_html=True)

    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=False)
    else:
        st.markdown("<h1>👁 EyeC</h1>", unsafe_allow_html=True)

    st.markdown(
        """
        <p style='text-align: center; font-size: 22px; font-weight: bold; color:#0D47A1;'>
            Smart Eye Disease Detection
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)
    time.sleep(2)  # splash duration (safe for HF)
    st.session_state.splash_shown = True
    st.rerun()

# -----------------------------
# Disease Information
# -----------------------------
disease_info = {
    "cataract": {
        "guidance": "Cataract causes clouding of the eye lens, leading to blurred vision. Surgery is the only definitive treatment.",
        "consult": "Please consult an ophthalmologist within the next month for evaluation."
    },
    "glaucoma": {
        "guidance": "Glaucoma damages the optic nerve, often due to high eye pressure. Early detection can prevent vision loss.",
        "consult": "Urgently consult an ophthalmologist for eye pressure testing."
    },
    "diabetic retinopathy": {
        "guidance": "Caused by diabetes damaging retinal blood vessels. Can lead to blindness if untreated.",
        "consult": "Schedule an eye exam with a retina specialist soon."
    },
    "normal": {
        "guidance": "Congratulations! Your eyes appear healthy.",
        "consult": "Maintain regular check-ups every 1-2 years."
    }
}

# -----------------------------
# Load TFLite model
# -----------------------------
@st.cache_resource
def load_tflite_model(model_path="model.tflite"):
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter

    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details(), interpreter.get_output_details()

interpreter, input_details, output_details = load_tflite_model("model.tflite")

# -----------------------------
# Load labels
# -----------------------------
@st.cache_data
def load_labels(path="labels.txt"):
    if not os.path.exists(path):
        return ["normal", "cataract", "glaucoma", "diabetic retinopathy"]
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]

labels = load_labels("labels.txt")

# -----------------------------
# Title & Subtitle
# -----------------------------
st.markdown("<h1 class='fade-in'>👁 EyeC - Eye Disease Detection</h1>", unsafe_allow_html=True)
st.subheader("Upload an eye image or open your camera to predict disease")

# -----------------------------
# Image Input
# -----------------------------
uploaded_file = st.file_uploader("Upload an eye image", type=["jpg", "jpeg", "png"])
camera_image = st.camera_input("📷 Open Camera & Take Photo")

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
elif camera_image is not None:
    image = Image.open(camera_image).convert("RGB")

# -----------------------------
# Prediction
# -----------------------------
if image is not None:
    try:
        st.image(image, use_container_width=True)

        # Preprocess
        input_shape = input_details[0]['shape']
        img_resized = image.resize((input_shape[1], input_shape[2]))
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        # Results
        pred_index = np.argmax(output_data)
        pred_label = labels[pred_index].strip().lower()
        pred_confidence = output_data[pred_index]

        if pred_label == "normal":
            st.success(f"🎉 Congratulations! Your eyes seem healthy ({pred_confidence*100:.2f}%)")
        else:
            st.warning(f"Prediction: *{pred_label.capitalize()}* ({pred_confidence*100:.2f}%)")

        # Medical guidance
        if pred_label in disease_info:
            st.markdown(f"**Medical Guidance:** {disease_info[pred_label]['guidance']}")
            st.markdown(f"**Consultation Advice:** {disease_info[pred_label]['consult']}")
        else:
            st.info("No specific guidance available for this condition.")

    except Exception as e:
        st.error(f"Error processing image: {e}")