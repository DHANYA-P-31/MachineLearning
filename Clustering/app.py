import joblib
import numpy as np
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Mall Customers K-Means Clustering", page_icon="🛍️", layout="centered")

st.title("Mall Customer Cluster Predictor (K-Means)")
st.write(
    "Enter customer values below. The app applies the same scaler used during training "
    "and predicts the cluster."
)

# Load saved artifacts
BASE_DIR = Path(__file__).resolve().parent


def load_artifact(filename: str):
    candidates = [BASE_DIR / filename, Path(filename)]
    for candidate in candidates:
        if candidate.exists():
            return joblib.load(candidate)

    st.error(
        f"Missing artifact: {filename}. Run notebook.ipynb to generate model files in the Clustering folder."
    )
    st.stop()


model = load_artifact("model.pkl")
scaler = load_artifact("scaler.pkl")
feature_cols = load_artifact("feature_columns.pkl")
cluster_profile = load_artifact("cluster_profile.pkl")

cluster_labels_path = BASE_DIR / "cluster_labels.pkl"
cluster_labels = joblib.load(cluster_labels_path) if cluster_labels_path.exists() else {}

st.subheader("Input Features")

# Defaults based on Mall Customers data ranges
annual_income = st.number_input("Annual Income (k$)", min_value=0.0, max_value=150.0, value=60.0, step=1.0)
spending_score = st.number_input("Spending Score (1-100)", min_value=1.0, max_value=100.0, value=50.0, step=1.0)

# Keep column order exactly same as training
input_map = {
    "Annual Income (k$)": annual_income,
    "Spending Score (1-100)": spending_score,
}

try:
    input_vector = np.array([[input_map[col] for col in feature_cols]])
except KeyError as exc:
    st.error(f"Feature mismatch between app and model artifacts: {exc}")
    st.stop()

if st.button("Predict Cluster"):
    scaled_input = scaler.transform(input_vector)
    pred_cluster = int(model.predict(scaled_input)[0])

    st.success(f"Predicted Cluster: {pred_cluster}")
    st.info(cluster_labels.get(pred_cluster, "Cluster label not available"))

    st.subheader("Cluster Profiling Summary")
    st.dataframe(cluster_profile)

st.caption("Tip: Run notebook.ipynb first if model files are missing.")
