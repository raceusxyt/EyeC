import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path

# Prefer tflite-runtime; fallback to TF Lite only if available locally
try:
    from tflite_runtime.interpreter import Interpreter
    BACKEND = "tflite-runtime"
except Exception:
    try:
        from tensorflow.lite import Interpreter  # type: ignore
        BACKEND = "tensorflow.lite (fallback)"
    except Exception as e:
        BACKEND = None
        _IMPORT_ERR = e

# -------- Page / Theme tweaks --------
st.set_page_config(page_title="EyeC – Eye Disease Detection", page_icon="👁️", layout="centered")
st.markdown("""
<style>
/* Brand accents */
:root { --eyec-blue:#1e3a8a; --eyec-yellow:#f59e0b; }
/* Title badge */
.badge {display:inline-block;padding:.2rem .6rem;border-radius:9999px;background:var(--eyec-yellow);color:#111;font-weight:600;}
/* Cards */
.block-container {padding-top:2rem !important;}
/* Progress label alignment */
.prob-row {display:flex;align-items:center;gap:.5rem;margin:.25rem 0;}
.prob-label {min-width:150px;font-weight:600;}
/* Section header underline */
h3 {border-bottom:3px solid rgba(245,158,11,.25);padding-bottom:.25rem;}
</style>
""", unsafe_allow_html=True)

st.title("👁️ EyeC – Eye Disease Detection")
st.caption("Glaucoma • Cataract • Diabetic Retinopathy • Crossed Eyes • Normal  ")
st.markdown('<span class="badge">Demo • Not a medical device</span>', unsafe_allow_html=True)

# -------- Helpers --------
@st.cache_resource
def load_labels():
    """Load labels from labels.txt or 'labels (1).txt' and strip numeric prefixes."""
    for cand in ["labels.txt", "labels (1).txt"]:
        if Path(cand).exists():
            path = cand
            break
    else:
        raise FileNotFoundError("Labels file not found (expected labels.txt or 'labels (1).txt').")

    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split(" ", 1)
            if len(parts) > 1 and parts[0].isdigit():
                labels.append(parts[1].strip())
            else:
                labels.append(s)
    if not labels:
        raise ValueError("Labels file is empty.")
    return labels, path

@st.cache_resource
def load_interpreter(model_path="model.tflite"):
    if not BACKEND:
        raise RuntimeError(
            f"Could not import a TFLite interpreter. Last error: {_IMPORT_ERR}\n"
            "Install 'tflite-runtime' (recommended) or 'tensorflow' as fallback."
        )
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    return interpreter, inp, out, BACKEND

def compress_image(image: Image.Image, max_size_kb=350, quality=85) -> Image.Image:
    """JPEG/PNG compress to speed uploads (esp. mobile)."""
    img = image.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    while buf.tell()/1024 > max_size_kb and quality > 25:
        quality -= 5
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def preprocess(image: Image.Image, input_detail):
    """Resize & normalize to match model input; handles NHWC/NCHW and float/uint8."""
    shape = input_detail["shape"]  # [1,H,W,C] or [1,C,H,W]
    if len(shape) != 4:
        raise RuntimeError(f"Unexpected input shape: {shape}")

    nchw = (shape[1] in (1,3)) and (shape[-1] not in (1,3))
    if nchw:
        c, h, w = int(shape[1]), int(shape[2]), int(shape[3])
    else:
        h, w, c = int(shape[1]), int(shape[2]), int(shape[3])

    img = image.convert("RGB").resize((w, h))
    arr = np.array(img)

    if input_detail["dtype"] == np.uint8:
        tensor = arr.astype(np.uint8)  # quantized expects [0,255]
    else:
        tensor = arr.astype(np.float32) / 255.0  # float32 expects [0,1]

    if nchw:
        tensor = np.transpose(tensor, (2, 0, 1))  # HWC -> CHW
    tensor = np.expand_dims(tensor, 0)
    return tensor

def infer(image: Image.Image, interpreter, inp, out):
    x = preprocess(image, inp)
    interpreter.set_tensor(inp["index"], x)
    interpreter.invoke()
    y = interpreter.get_tensor(out["index"])[0]

    # Dequantize output if needed
    if out["dtype"] == np.uint8:
        scale, zp = out.get("quantization", (0.0, 0))
        y = (y.astype(np.float32) - zp) * (scale if scale else 1.0)

    # Softmax to probabilities
    y = y - np.max(y)
    probs = np.exp(y) / np.sum(np.exp(y))
    return probs

# -------- Load assets --------
try:
    labels, labels_path = load_labels()
except Exception as e:
    st.error(f"Labels error: {e}")
    st.stop()

try:
    interpreter, inp, out, backend = load_interpreter("model.tflite")
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

with st.expander("🔧 Model & App Info"):
    size_mb = Path("model.tflite").stat().st_size / (1024*1024)
    st.write(f"**Interpreter**: {backend}")
    st.write(f"**Model**: model.tflite ({size_mb:.2f} MB)")
    st.write(f"**Input**: shape {inp['shape']}, dtype `{inp['dtype']}`")
    st.write(f"**Output**: shape {out['shape']}, dtype `{out['dtype']}`")
    st.write(f"**Labels**: from `{labels_path}`")

# -------- UI: Upload / Camera --------
st.subheader("📸 Upload or Take a Photo")
uploaded = st.file_uploader(
    "Upload or take a photo of your eye",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False,
    help="On mobile, tap to use your camera or choose from your gallery."
)

if uploaded:
    try:
        img = Image.open(uploaded).convert("RGB")
        img = compress_image(img)  # faster on mobile
        c1, c2 = st.columns([2, 1])
        with c1:
            st.image(img, caption="Uploaded image", use_container_width=True)
        with c2:
            st.info(f"**Size:** {img.size[0]}×{img.size[1]}\n\n**Mode:** RGB")

        st.divider()
        if st.button("🚀 Run Disease Detection", type="primary", use_container_width=True):
            with st.spinner("Analyzing…"):
                probs = infer(img, interpreter, inp, out)

            idx = int(np.argmax(probs))
            name = labels[idx] if idx < len(labels) else f"Class {idx}"
            p = float(probs[idx])

            st.markdown("### 🎯 Primary Result")
            if name.lower() == "normal":
                st.success(f"✅ **HEALTHY EYE DETECTED**")
                st.metric("Confidence", f"{p:.1%}")
                st.balloons()
            else:
                st.warning(f"⚠️ **{name.upper()} DETECTED**")
                st.metric("Confidence", f"{p:.1%}")

            st.markdown("### 📊 Detailed Probabilities")
            order = np.argsort(probs)[::-1]
            for rank, j in enumerate(order, 1):
                label = labels[j] if j < len(labels) else f"Class {j}"
                pj = float(probs[j])
                st.markdown(
                    f'<div class="prob-row"><div class="prob-label">{("🥇" if rank==1 else "🥈" if rank==2 else "🥉" if rank==3 else str(rank)+".")} {label.title()}</div><div style="flex:1">{int(pj*1000)/10:.1f}%</div></div>',
                    unsafe_allow_html=True
                )
                st.progress(pj)

            st.divider()
            st.warning(
                "⚠️ **Medical Disclaimer**: EyeC is a research/education demo and not a medical device. "
                "For any symptoms or concerns, consult a licensed ophthalmologist."
            )
            if name.lower() != "normal":
                with st.expander("🩺 What to do next"):
                    st.markdown(
                        "- Schedule a comprehensive eye exam with an ophthalmologist.\n"
                        "- Bring this result as a reference; the doctor may perform OCT/fundus tests.\n"
                        "- Maintain good lighting and a sharp image for re-checks."
                    )
    except Exception as e:
        st.error(f"Error processing image: {e}")
else:
    st.info("👆 Tap above to take a photo or choose one from your gallery.")
