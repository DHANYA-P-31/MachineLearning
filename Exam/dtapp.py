import joblib
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Heart Disease Predictor", page_icon="ML")
st.title("Decision Tree Heart Disease Predictor")
st.caption("Loads dt_model.joblib saved from decisionTree.ipynb")

artifact = joblib.load("dt_model.joblib")
model = artifact["model"]
features = artifact["features"]

st.subheader("Enter Patient Values")
user_input = {}
for feature in features:
	user_input[feature] = st.number_input(feature, value=0.0, step=1.0)

if st.button("Predict", type="primary"):
	input_df = pd.DataFrame([user_input], columns=features)
	pred = int(model.predict(input_df)[0])
	proba = model.predict_proba(input_df)[0]

	st.success("Prediction complete")
	st.write("Prediction:", "Heart Disease" if pred == 1 else "No Heart Disease")
	st.write(f"Probability (No Disease): {proba[0]:.2%}")
	st.write(f"Probability (Disease): {proba[1]:.2%}")
