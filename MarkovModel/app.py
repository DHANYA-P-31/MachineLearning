from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from hmmlearn.hmm import GaussianHMM
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / "data" / "heart.csv"
MODEL_PATH = ROOT / "model.pkl"


FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERICAL_COLS = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLS]


@st.cache_resource(show_spinner=False)
def load_or_train_artifact():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = df.dropna().copy()

    X_raw = df[FEATURE_COLUMNS]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), NUMERICAL_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )

    X_processed = preprocessor.fit_transform(X_raw)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    # Build pseudo-sequences by age sorting and fixed windows.
    sorted_idx = np.argsort(df["age"].values)
    X_seq = X_processed[sorted_idx]
    df_seq = df.iloc[sorted_idx].reset_index(drop=True)

    seq_len = 12
    lengths = []
    start = 0
    while start < len(X_seq):
        chunk = min(seq_len, len(X_seq) - start)
        lengths.append(chunk)
        start += chunk

    # Use 3 hidden states for deployment-friendly interpretation.
    model = GaussianHMM(
        n_components=3,
        covariance_type="diag",
        n_iter=400,
        random_state=42,
    )
    model.fit(X_seq, lengths)
    hidden = model.predict(X_seq, lengths)

    risk_df = df_seq.copy()
    risk_df["hidden"] = hidden

    risk_score = (
        0.20 * (risk_df["age"] / risk_df["age"].max())
        + 0.20 * (risk_df["chol"] / risk_df["chol"].max())
        + 0.20 * (risk_df["trestbps"] / risk_df["trestbps"].max())
        + 0.25 * (risk_df["oldpeak"] / (risk_df["oldpeak"].max() + 1e-8))
        + 0.15 * (1 - (risk_df["thalach"] / risk_df["thalach"].max()))
    )
    risk_df["risk_score"] = risk_score

    ordered = risk_df.groupby("hidden")["risk_score"].mean().sort_values().index.tolist()
    labels = ["Healthy", "At Risk", "Diseased"]
    state_name_map = {}
    for idx, st_id in enumerate(ordered):
        state_name_map[int(st_id)] = labels[min(idx, 2)]

    artifact = {
        "model": model,
        "preprocessor": preprocessor,
        "feature_columns": FEATURE_COLUMNS,
        "state_name_map": state_name_map,
    }

    joblib.dump(artifact, MODEL_PATH)
    return artifact


def to_feature_dataframe(user_values):
    return pd.DataFrame([user_values], columns=FEATURE_COLUMNS)


def main():
    st.set_page_config(page_title="Heart Disease HMM Predictor", page_icon="❤", layout="centered")

    st.title("Heart Disease Hidden State Predictor (HMM)")
    st.write(
        "Predicts latent condition state as **Healthy**, **At Risk**, or **Diseased** "
        "using a Gaussian Hidden Markov Model."
    )

    artifact = load_or_train_artifact()
    model = artifact["model"]
    preprocessor = artifact["preprocessor"]
    state_name_map = artifact["state_name_map"]

    st.subheader("Enter Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 20, 90, 54)
        sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1], index=1)
        cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3], index=0)
        trestbps = st.slider("Resting Blood Pressure (trestbps)", 80, 220, 130)
        chol = st.slider("Cholesterol (chol)", 100, 600, 240)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1], index=0)
        restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2], index=1)

    with col2:
        thalach = st.slider("Max Heart Rate (thalach)", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1], index=0)
        oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 7.0, 1.0, 0.1)
        slope = st.selectbox("Slope (slope)", [0, 1, 2], index=1)
        ca = st.selectbox("Major Vessels (ca)", [0, 1, 2, 3, 4], index=0)
        thal = st.selectbox("Thal (thal)", [0, 1, 2, 3], index=2)

    if st.button("Predict Heart Disease State", type="primary"):
        user_values = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
        }

        input_df = to_feature_dataframe(user_values)
        x = preprocessor.transform(input_df)
        if hasattr(x, "toarray"):
            x = x.toarray()

        predicted_state = int(model.predict(x, lengths=[1])[0])
        probabilities = model.predict_proba(x, lengths=[1])[0]

        predicted_label = state_name_map.get(predicted_state, f"State {predicted_state}")
        confidence = float(np.max(probabilities))

        st.success(f"Predicted State: {predicted_label}")
        st.info(f"Confidence (posterior max): {confidence:.3f}")

        prob_df = pd.DataFrame(
            {
                "State": [state_name_map.get(i, f"State {i}") for i in range(model.n_components)],
                "Probability": probabilities,
            }
        ).sort_values("Probability", ascending=False)

        st.subheader("State Probability Distribution")
        st.dataframe(prob_df, use_container_width=True)


if __name__ == "__main__":
    main()
