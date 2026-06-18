import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps
import io

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Digit Classifier",
    page_icon="✏️",
    layout="centered",
)
python3 << 'EOF'
import re

with open("app.py", "r") as f:
    content = f.read()

old = '''@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

# ── Styling'''

new = '''@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── Styling'''

content = content.replace(old, new)

with open("app.py", "w") as f:
    f.write(content)

print("Done")
EOF
# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
import joblib
...
def load_model():
    return joblib.load("model.pkl")

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .title { font-size: 2.4rem; font-weight: 800; color: #FFFFFF; margin-bottom: 0; }
    .subtitle { color: #9ca3af; font-size: 1rem; margin-top: 0.2rem; margin-bottom: 2rem; }
    .pred-box {
        background: linear-gradient(135deg, #1e3a5f, #1a2a4a);
        border: 1px solid #2563eb;
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        margin-top: 1rem;
    }
    .pred-digit { font-size: 5rem; font-weight: 900; color: #60a5fa; line-height: 1; }
    .pred-label { font-size: 0.9rem; color: #93c5fd; margin-top: 0.5rem; letter-spacing: 0.05em; }
    .conf-bar-label { font-size: 0.78rem; color: #9ca3af; margin-bottom: 0.2rem; }
    .stButton>button {
        width: 100%;
        background: #2563eb;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.65rem;
        font-size: 1rem;
        font-weight: 600;
        transition: background 0.2s;
    }
    .stButton>button:hover { background: #1d4ed8; }
    .warning-box {
        background: #422006;
        border: 1px solid #d97706;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        color: #fbbf24;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="title">✏️ Digit Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an image of a handwritten digit (0–9) and let the model predict it.</p>', unsafe_allow_html=True)

if not model_loaded:
    st.markdown("""
    <div class="warning-box">
    ⚠️ <strong>model.pkl not found.</strong><br>
    Run <code>python train_and_save.py</code> first to train and save the model, then restart Streamlit.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Helper: preprocess image ───────────────────────────────────────────────────
def preprocess_image(img: Image.Image) -> np.ndarray:
    """Convert any uploaded image into a (1, 784) float32 array matching MNIST format."""
    img = img.convert("L")                        # grayscale
    img = ImageOps.invert(img)                    # MNIST: white digit on black bg
    img = img.resize((28, 28), Image.LANCZOS)     # resize to 28×28
    arr = np.array(img, dtype=np.float32) / 255.0 # normalise
    return arr.reshape(1, 784)

# ── Layout: two columns ────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("#### 📂 Upload Image")
    uploaded = st.file_uploader(
        "Supported: PNG, JPG, BMP",
        type=["png", "jpg", "jpeg", "bmp"],
        label_visibility="collapsed",
    )

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Your image", use_container_width=True)

with col_right:
    st.markdown("#### 🔍 Prediction")

    if not uploaded:
        st.info("Upload an image on the left to get started.", icon="👈")
    else:
        processed = preprocess_image(img)
        probas = model.predict_proba(processed)[0]   # shape (10,)
        pred = int(np.argmax(probas))
        confidence = probas[pred] * 100

        # ── Prediction box ──────────────────────────────────────────────────
        st.markdown(f"""
        <div class="pred-box">
            <div class="pred-digit">{pred}</div>
            <div class="pred-label">PREDICTED DIGIT — {confidence:.1f}% confidence</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Confidence bars for all classes ─────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Confidence per digit**")
        for digit in range(10):
            prob = float(probas[digit]) * 100
            bar_col, label_col = st.columns([5, 1])
            with bar_col:
                st.progress(prob / 100)
            with label_col:
                st.markdown(
                    f"<div style='color:{'#60a5fa' if digit==pred else '#9ca3af'};"
                    f"font-weight:{'700' if digit==pred else '400'};padding-top:4px'>"
                    f"{digit}</div>",
                    unsafe_allow_html=True,
                )

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("""
    **Model:** Random Forest Classifier  
    **Dataset:** MNIST (60k train / 10k test)  
    **Estimators:** 20 trees  
    **Input:** 28 × 28 grayscale image, flattened to 784 features  
    """)
    st.divider()
    st.markdown("**Tips for best results:**")
    st.markdown("""
    - Draw a digit on white paper, photograph it, crop tightly  
    - Or screenshot a digit from any handwriting tool  
    - Avoid extra noise/background around the digit  
    """)
    st.divider()
    st.markdown("Built with 🌟 **Streamlit** + **scikit-learn**")
