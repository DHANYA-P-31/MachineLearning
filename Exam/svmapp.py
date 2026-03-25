
import joblib
import streamlit as st


st.set_page_config(page_title="SVM Spam Detector", page_icon=":email:")
st.title("SVM Spam Detector")
st.caption("Loads svm_model.joblib saved from svm.ipynb")

artifact = joblib.load("svm_model.joblib")
model = artifact["model"]
vectorizer = artifact["vectorizer"]

message = st.text_area(
    "Enter a message",
    value="Congratulations! You have won a free ticket. Reply WIN now.",
    height=140,
)

if st.button("Predict", type="primary"):
    if not message.strip():
        st.warning("Please enter some text.")
    else:
        vec = vectorizer.transform([message])
        pred = model.predict(vec)[0]
        label = "Spam" if pred == "spam" else "Ham"
        st.success(f"Prediction: {label}")
