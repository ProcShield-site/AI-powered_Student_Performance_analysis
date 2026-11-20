import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

CNN_PATH = "cnn_model.h5"
CNN_AVAILABLE = os.path.exists(CNN_PATH)
KNN_PATH = "knn_model.pkl"
SVM_PATH = "svm_model.pkl"

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(
    page_title="GradeGenius AI",
    page_icon="🎓",
    layout="centered",
)

BANNER_PATH = "banner.png"

# Colors used for each grade badge/chart
GRADE_COLORS = {
    "A": "#2ecc71",   # green
    "B": "#3498db",   # blue
    "C": "#f1c40f",   # yellow
    "D": "#e67e22",   # orange
    "F": "#e74c3c",   # red
}

st.markdown(
    """
    <style>
    /* ---------- animated, multi-colour pastel background ---------- */
    [data-testid="stApp"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    .stApp {
        background: linear-gradient(
            125deg,
            #e3f2fd 0%,
            #f3e5f5 20%,
            #fff3e0 40%,
            #e8f5e9 60%,
            #e0f7fa 80%,
            #e3f2fd 100%
        ) !important;
        background-size: 400% 400% !important;
        animation: bgShift 22s ease infinite;
    }
    @keyframes bgShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* ---------- floating, glowing banner image ---------- */
    div[data-testid="stImage"] img {
        border-radius: 20px;
        box-shadow: 0 12px 30px rgba(80, 60, 200, 0.25);
        animation: floatY 4s ease-in-out infinite;
    }
    @keyframes floatY {
        0%, 100% { transform: translateY(0px); }
        50%      { transform: translateY(-8px); }
    }

    /* ---------- gradient animated title ---------- */
    .gg-title {
        font-size: clamp(40px, 7vw, 76px);
        line-height: 1.15;
        font-weight: 900;
        text-align: center;
        margin-bottom: 0px;
        background: linear-gradient(90deg, #6a11cb, #2575fc, #11cbbf, #6a11cb);
        background-size: 300% auto;
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: titleShine 6s linear infinite;
    }
    @keyframes titleShine {
        0%   { background-position: 0% 50%; }
        100% { background-position: 300% 50%; }
    }
    .gg-subtitle {
        text-align: center;
        font-size: clamp(20px, 3.2vw, 32px);
        font-weight: 700;
        color: #4b3f72;
        margin-top: 6px;
    }
    .gg-tagline {
        text-align: center;
        font-size: clamp(14px, 1.6vw, 18px);
        color: #6b6b85;
        margin-top: 6px;
        margin-bottom: 10px;
    }

    /* ---------- section headers with colored pill ---------- */
    .gg-section {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 999px;
        background: linear-gradient(90deg, #6a11cb, #2575fc);
        color: white !important;
        font-weight: 700;
        font-size: 18px;
        margin-bottom: 10px;
        box-shadow: 0 4px 14px rgba(37, 117, 252, 0.35);
    }

    /* ---------- glowing, pulsing, 3D-tilt grade badge ---------- */
    .grade-badge {
        display: inline-block;
        padding: 18px 38px;
        border-radius: 18px;
        font-size: 46px;
        font-weight: 800;
        color: white;
        text-align: center;
        box-shadow: 0 0 25px 4px var(--glow, rgba(52,152,219,0.6)), 0 10px 20px rgba(0,0,0,0.15);
        animation: pulseGlow 2.2s ease-in-out infinite;
        transition: transform 0.25s ease;
        transform-style: preserve-3d;
    }
    .grade-badge:hover {
        transform: perspective(500px) rotateX(8deg) rotateY(-8deg) scale(1.05);
    }
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 18px 2px var(--glow, rgba(52,152,219,0.55)), 0 10px 20px rgba(0,0,0,0.15); }
        50%      { box-shadow: 0 0 32px 8px var(--glow, rgba(52,152,219,0.85)), 0 10px 20px rgba(0,0,0,0.15); }
    }

    /* ---------- colorful 3D-hover cards ---------- */
    .metric-card {
        background: white;
        border-radius: 14px;
        padding: 14px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.10);
        text-align: center;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .metric-card:hover {
        transform: perspective(600px) rotateX(4deg) translateY(-4px);
        box-shadow: 0 14px 28px rgba(106, 17, 203, 0.25);
    }

    /* ---------- gradient, lifting predict button ---------- */
    .stButton > button, .stFormSubmitButton > button {
        background: linear-gradient(90deg, #6a11cb, #2575fc, #11cbbf);
        background-size: 200% auto;
        color: white !important;
        font-weight: 700;
        border: none;
        border-radius: 999px;
        box-shadow: 0 6px 18px rgba(37, 117, 252, 0.4);
        transition: all 0.3s ease;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background-position: right center;
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 10px 24px rgba(106, 17, 203, 0.5);
    }

    /* ---------- colorful expander accents ---------- */
    div[data-testid="stExpander"] {
        border-radius: 12px !important;
        border-left: 5px solid #2575fc !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# Load saved model + preprocessing objects (cached, loads only once)
# ------------------------------------------------------------------
@st.cache_resource
def load_files():
    rf_model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("label_encoders.pkl")
    meta = joblib.load("meta.pkl")
    return rf_model, scaler, encoders, meta


@st.cache_resource
def load_cnn():
    from tensorflow.keras.models import load_model
    return load_model(CNN_PATH)


rf_model, scaler, encoders, meta = load_files()

feature_columns = meta["feature_columns"]
categorical = meta["categorical"]
numeric = meta["numeric"]
best_model_name = meta.get("best_model_name", "Random Forest")


def pretty(name: str) -> str:
    return name.replace("_", " ").title()


def hex_to_rgba(hex_color: str, alpha: float = 0.6) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
if os.path.exists(BANNER_PATH):
    st.image(BANNER_PATH, use_container_width=True)

st.markdown('<div class="gg-title">GradeGenius AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="gg-subtitle">Know Your Performance. Shape Your Future.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="gg-tagline">Fill in a student\'s details to instantly predict their final grade \u2014 '
    "compares Random Forest, KNN, SVM"
    + (" and CNN." if CNN_AVAILABLE else " (add the CNN model to compare all 4).")
    + "</div>",
    unsafe_allow_html=True,
)
st.divider()

# ------------------------------------------------------------------
# Model choice
# ------------------------------------------------------------------
model_options = ["Random Forest"]
if os.path.exists(KNN_PATH):
    model_options.append("KNN")
if os.path.exists(SVM_PATH):
    model_options.append("SVM")
if CNN_AVAILABLE:
    model_options.append("CNN")

chosen_model = st.radio("Choose a model", model_options, horizontal=True)
if not CNN_AVAILABLE:
    st.caption("CNN model not found — run `colab_cnn/train_cnn.py` in Colab and add "
               "`cnn_model.h5` to this folder to unlock it here.")

# ------------------------------------------------------------------
# Input form (built automatically from the training data)
# ------------------------------------------------------------------
with st.form("student_form"):
    st.markdown('<span class="gg-section">📋 Student Information</span>', unsafe_allow_html=True)

    left, right = st.columns(2)
    inputs = {}

    for i, col in enumerate(feature_columns):
        target = left if i % 2 == 0 else right
        with target:
            if col in categorical:
                inputs[col] = st.selectbox(pretty(col), categorical[col])
            else:
                info = numeric[col]
                inputs[col] = st.number_input(
                    pretty(col),
                    min_value=float(info["min"]),
                    max_value=float(info["max"]),
                    value=float(round(info["mean"], 1)),
                )

    submitted = st.form_submit_button("🔮 Predict Final Grade", use_container_width=True)

# ------------------------------------------------------------------
# Prediction (same preprocessing steps as training)
# ------------------------------------------------------------------
if submitted:
    row = pd.DataFrame([inputs])

    for col in categorical:
        row[col] = encoders[col].transform(row[col])

    row = row[feature_columns]
    row_scaled = scaler.transform(row)

    if chosen_model == "CNN":
        cnn = load_cnn()
        row_cnn = row_scaled.reshape(row_scaled.shape[0], row_scaled.shape[1], 1)
        proba_all = cnn.predict(row_cnn, verbose=0)[0]
        pred = int(np.argmax(proba_all))
        active_model = None  # CNN doesn't use sklearn's predict_proba path below
    else:
        if chosen_model == "KNN":
            active_model = joblib.load(KNN_PATH)
        elif chosen_model == "SVM":
            active_model = joblib.load(SVM_PATH)
        else:
            active_model = rf_model
        pred = active_model.predict(row_scaled)[0]
        proba_all = active_model.predict_proba(row_scaled)[0] if hasattr(active_model, "predict_proba") else None

    grade = encoders["final_grade"].inverse_transform([pred])[0]
    color = GRADE_COLORS.get(grade, "#3498db")
    glow = hex_to_rgba(color, 0.65)

    st.divider()
    st.markdown('<span class="gg-section">🏆 Result</span>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(
            f'<div class="grade-badge" style="background:{color}; --glow:{glow};">{grade}</div>',
            unsafe_allow_html=True,
        )
    with c2:
        if grade in ("D", "F"):
            st.warning("⚠️ This student is at risk. Consider academic support or intervention.")
        elif grade == "C":
            st.info("This student is performing adequately but has room to improve.")
        else:
            st.success("✅ This student is performing well.")

    if proba_all is not None:
        if chosen_model == "CNN":
            class_ids = np.arange(len(proba_all))
        else:
            class_ids = active_model.classes_
        labels = encoders["final_grade"].inverse_transform(class_ids)
        prob_df = pd.DataFrame({"Grade": labels, "Probability (%)": (proba_all * 100).round(2)})
        prob_df = prob_df.set_index("Grade").reindex(sorted(labels))

        st.markdown('<span class="gg-section">📊 Prediction Confidence</span>', unsafe_allow_html=True)
        bar_colors = [GRADE_COLORS.get(g, "#3498db") for g in prob_df.index]

        fig, ax = plt.subplots(figsize=(6, 3.2))
        bars = ax.barh(prob_df.index.astype(str), prob_df["Probability (%)"], color=bar_colors)
        for bar, val in zip(bars, prob_df["Probability (%)"]):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")
        ax.set_xlim(0, 110)
        ax.set_xlabel("Probability (%)")
        ax.invert_yaxis()
        fig.tight_layout()
        st.pyplot(fig)

        st.dataframe(prob_df, use_container_width=True)

st.divider()

model_list_label = "Random Forest vs KNN vs SVM vs CNN" if CNN_AVAILABLE else "Random Forest vs KNN vs SVM"
with st.expander(f"📈 View model comparison ({model_list_label})"):
    if os.path.exists("Model_Comparison.csv"):
        comp = pd.read_csv("Model_Comparison.csv")
        st.dataframe(comp.style.highlight_max(axis=0, subset=["Accuracy", "Precision", "Recall", "F1 Score"],
                                               color="#d4f7dc"), use_container_width=True)
        if os.path.exists("Model_Comparison_Grouped.png"):
            st.image("Model_Comparison_Grouped.png", use_container_width=True)
    else:
        st.write("Comparison file not found — add Model_Comparison.csv to this folder.")

with st.expander("🔍 View feature importance"):
    if os.path.exists("Feature_Importance.png"):
        st.image("Feature_Importance.png", use_container_width=True)
    else:
        st.write("Feature importance chart not found — add Feature_Importance.png to this folder.")

if CNN_AVAILABLE:
    with st.expander("🧠 View CNN training curves & confusion matrix"):
        for img in ["CNN_Training_Curves.png", "CNN_Confusion_Matrix.png"]:
            if os.path.exists(img):
                st.image(img, use_container_width=True)

st.caption("M.Tech Dissertation Project | Abhishek Chakraborty | Brainware University")
