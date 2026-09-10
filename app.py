import streamlit as st
import pandas as pd
import joblib


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon="🩺",
    layout="wide"
)


# -----------------------------
# Load Model & Preprocessor
# -----------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")


# -----------------------------
# App Title
# -----------------------------
st.title("🩺 Breast Cancer Classifier")

st.write(
    "Enter the tumor measurements below to predict "
    "whether the tumor is benign or malignant."
)


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("About the Model")

st.sidebar.info(
    "This application uses a Logistic Regression model "
    "trained on 10 selected features from the "
    "Breast Cancer dataset."
)


# -----------------------------
# Input Features
# -----------------------------
st.subheader("Tumor Features")

col1, col2 = st.columns(2)

with col1:
    mean_radius = st.number_input(
        "Mean Radius",
        min_value=0.0,
        value=14.0
    )

    mean_texture = st.number_input(
        "Mean Texture",
        min_value=0.0,
        value=19.0
    )

    mean_perimeter = st.number_input(
        "Mean Perimeter",
        min_value=0.0,
        value=90.0
    )

    mean_area = st.number_input(
        "Mean Area",
        min_value=0.0,
        value=650.0
    )

    mean_smoothness = st.number_input(
        "Mean Smoothness",
        min_value=0.0,
        value=0.10,
        format="%.4f"
    )


with col2:
    mean_compactness = st.number_input(
        "Mean Compactness",
        min_value=0.0,
        value=0.10,
        format="%.4f"
    )

    mean_concavity = st.number_input(
        "Mean Concavity",
        min_value=0.0,
        value=0.08,
        format="%.4f"
    )

    mean_concave_points = st.number_input(
        "Mean Concave Points",
        min_value=0.0,
        value=0.05,
        format="%.4f"
    )

    mean_symmetry = st.number_input(
        "Mean Symmetry",
        min_value=0.0,
        value=0.18,
        format="%.4f"
    )

    mean_fractal_dimension = st.number_input(
        "Mean Fractal Dimension",
        min_value=0.0,
        value=0.06,
        format="%.4f"
    )


# -----------------------------
# Prediction
# -----------------------------
st.divider()

if st.button("🔍 Predict", use_container_width=True):

    input_data = pd.DataFrame(
        [[
            mean_radius,
            mean_texture,
            mean_perimeter,
            mean_area,
            mean_smoothness,
            mean_compactness,
            mean_concavity,
            mean_concave_points,
            mean_symmetry,
            mean_fractal_dimension
        ]],
        columns=features
    )

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Probability
    probability = model.predict_proba(input_scaled)[0]

    # Display result
    if prediction == 0:
        st.error("⚠️ Prediction: Malignant")
    else:
        st.success("✅ Prediction: Benign")

    # Display probabilities
    st.subheader("Prediction Probability")

    benign_probability = probability[1]
    malignant_probability = probability[0]

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Benign",
            f"{benign_probability:.2%}"
        )

    with col2:
        st.metric(
            "Malignant",
            f"{malignant_probability:.2%}"
        )