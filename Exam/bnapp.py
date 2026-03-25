import joblib
import streamlit as st


st.set_page_config(page_title="BN Heart Disease", layout="centered")
st.title("BN Heart Disease")
st.caption("Loads bn_predictions.joblib saved from bn.ipynb")

payload = joblib.load("bn_predictions.joblib")
lookup = payload["lookup"]

age_group = st.selectbox("Age Group", payload.get("age_groups", ["young", "middle", "senior"]), index=1)
chol = st.selectbox(
    "Cholesterol Level",
    payload.get("chol_levels", ["normal", "borderline_high", "high"]),
    index=1,
)
bp = st.selectbox("BP Level", payload.get("bp_levels", ["normal", "elevated", "high"]), index=1)
chest_pain = st.selectbox(
    "Chest Pain Type",
    payload.get(
        "chest_pain_types",
        ["typical_angina", "atypical_angina", "non_anginal_pain", "asymptomatic"],
    ),
    index=2,
)
max_hr = st.selectbox("Max HR Level", payload.get("max_hr_levels", ["low", "normal", "high"]), index=1)
threshold = st.slider("Decision Threshold", 0.30, 0.80, 0.50, 0.05)

if st.button("Predict", type="primary"):
    key = (age_group, chol, bp, chest_pain, max_hr)
    p_yes = float(lookup[key])
    p_no = float(1.0 - p_yes)
    diagnosis = "Heart disease likely" if p_yes >= threshold else "Heart disease unlikely"

    st.metric("P(HeartDisease=1)", f"{p_yes:.4f}")
    st.caption(f"P(HeartDisease=0): {p_no:.4f}")
    st.info(diagnosis)
