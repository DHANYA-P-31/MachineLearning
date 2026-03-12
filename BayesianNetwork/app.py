import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork


st.set_page_config(page_title="Heart Disease Bayesian Network", layout="centered")


@st.cache_data
def load_raw_data() -> pd.DataFrame:
    csv_path = Path(__file__).resolve().parent.parent / "data" / "heart.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find dataset at: {csv_path}")
    return pd.read_csv(csv_path)


@st.cache_data
def preprocess_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    selected_cols = ["age", "chol", "trestbps", "cp", "thalach", "target"]
    df = df_raw[selected_cols].copy()

    for col in ["age", "chol", "trestbps", "thalach"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    for col in ["cp", "target"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].mode().iloc[0]).astype(int)

    df["AgeGroup"] = pd.cut(
        df["age"],
        bins=[0, 40, 55, 120],
        labels=["young", "middle", "senior"],
        include_lowest=True,
    )
    df["CholesterolLevel"] = pd.cut(
        df["chol"],
        bins=[0, 200, 240, 1000],
        labels=["normal", "borderline_high", "high"],
        include_lowest=True,
    )
    df["BPLevel"] = pd.cut(
        df["trestbps"],
        bins=[0, 120, 140, 300],
        labels=["normal", "elevated", "high"],
        include_lowest=True,
    )
    df["MaxHRLevel"] = pd.cut(
        df["thalach"],
        bins=[0, 120, 160, 260],
        labels=["low", "normal", "high"],
        include_lowest=True,
    )

    cp_map = {
        0: "typical_angina",
        1: "atypical_angina",
        2: "non_anginal_pain",
        3: "asymptomatic",
    }
    df["ChestPainType"] = df["cp"].map(cp_map).fillna("asymptomatic")

    df_bn = df[[
        "AgeGroup",
        "CholesterolLevel",
        "BPLevel",
        "ChestPainType",
        "MaxHRLevel",
        "target",
    ]].copy()
    df_bn = df_bn.rename(columns={"target": "HeartDisease"})
    df_bn["HeartDisease"] = df_bn["HeartDisease"].astype(int)

    for col in ["AgeGroup", "CholesterolLevel", "BPLevel", "ChestPainType", "MaxHRLevel"]:
        if df_bn[col].isna().any():
            df_bn[col] = df_bn[col].fillna(df_bn[col].mode().iloc[0])

    return df_bn


@st.cache_resource
def train_model(df_bn: pd.DataFrame):
    model = BayesianNetwork([
        ("AgeGroup", "BPLevel"),
        ("AgeGroup", "CholesterolLevel"),
        ("AgeGroup", "HeartDisease"),
        ("BPLevel", "HeartDisease"),
        ("CholesterolLevel", "HeartDisease"),
        ("ChestPainType", "HeartDisease"),
        ("MaxHRLevel", "HeartDisease"),
    ])
    model.fit(df_bn, estimator=MaximumLikelihoodEstimator)
    inference = VariableElimination(model)
    return model, inference


def predict_heart_disease(inference: VariableElimination, evidence: dict, threshold: float):
    posterior = inference.query(variables=["HeartDisease"], evidence=evidence)
    states = posterior.state_names["HeartDisease"]

    if 1 in states:
        positive_index = states.index(1)
    elif "1" in states:
        positive_index = states.index("1")
    else:
        positive_index = int(np.argmax(states))

    p_yes = float(posterior.values[positive_index])
    p_no = float(1.0 - p_yes)
    diagnosis = "Heart disease likely" if p_yes >= threshold else "Heart disease unlikely"

    posterior_df = pd.DataFrame(
        {
            "State": [str(state) for state in states],
            "Probability": posterior.values,
        }
    )
    return p_yes, p_no, diagnosis, posterior_df


def main() -> None:
    st.title("Heart Disease Prediction Using Bayesian Network")
    st.write("Choose the patient conditions and get a simple risk prediction.")

    try:
        df_raw = load_raw_data()
        df_bn = preprocess_data(df_raw)
        _, inference = train_model(df_bn)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    age_group = st.selectbox("Age Group", ["young", "middle", "senior"], index=1)
    cholesterol = st.selectbox(
        "Cholesterol Level", ["normal", "borderline_high", "high"], index=1
    )
    bp_level = st.selectbox("BP Level", ["normal", "elevated", "high"], index=1)
    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["typical_angina", "atypical_angina", "non_anginal_pain", "asymptomatic"],
        index=2,
    )
    max_hr = st.selectbox("Max HR Level", ["low", "normal", "high"], index=1)
    threshold = st.slider("Decision Threshold", 0.30, 0.80, 0.50, 0.05)

    evidence = {
        "AgeGroup": age_group,
        "CholesterolLevel": cholesterol,
        "BPLevel": bp_level,
        "ChestPainType": chest_pain,
        "MaxHRLevel": max_hr,
    }

    p_yes, p_no, diagnosis, posterior_df = predict_heart_disease(inference, evidence, threshold)

    st.metric("Heart Disease Probability", f"{p_yes:.4f}")
    st.info(diagnosis)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(posterior_df["State"], posterior_df["Probability"], color=["#4f81bd", "#c0504d"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Posterior Distribution")
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)

    st.caption(f"P(HeartDisease = 0): {p_no:.4f} | Threshold: {threshold:.2f}")


if __name__ == "__main__":
    main()
