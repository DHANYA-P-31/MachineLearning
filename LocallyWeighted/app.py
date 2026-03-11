import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


st.set_page_config(page_title="Fish Market LWR", page_icon="🐟", layout="wide")


def locally_weighted_regression(X_design: np.ndarray, y: np.ndarray, tau: float, x_query: np.ndarray) -> float:
    """Predict a single point with Locally Weighted Regression."""
    diff = X_design - x_query
    sq_dist = np.sum(diff * diff, axis=1)
    w = np.exp(-sq_dist / (2 * tau**2))

    xtwx = X_design.T @ (X_design * w[:, None])
    xtwy = X_design.T @ (w * y)

    ridge = 1e-8 * np.eye(X_design.shape[1])
    theta = np.linalg.pinv(xtwx + ridge) @ xtwy
    return float(x_query @ theta)


def lwr_predict(X_design: np.ndarray, y: np.ndarray, X_query: np.ndarray, tau: float) -> np.ndarray:
    return np.array([
        locally_weighted_regression(X_design, y, tau, xq)
        for xq in X_query
    ])


@st.cache_data
def load_data() -> pd.DataFrame:
    # app.py is in LocallyWeighted/, dataset is in ../data/Fish.csv
    csv_path = Path(__file__).resolve().parent.parent / "data" / "Fish.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find dataset at: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df[["Length3", "Weight"]].dropna()
    df = df[df["Weight"] > 0].copy()
    return df


@st.cache_data
def build_training_arrays(df: pd.DataFrame):
    X = df[["Length3"]].values
    y = df["Weight"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_design = np.hstack((np.ones((X_scaled.shape[0], 1)), X_scaled))
    return X, y, X_scaled, X_design, scaler


def main() -> None:
    st.title("Locally Weighted Regression - Fish Market")
    st.write("Predict **fish weight** from **Length3** using Locally Weighted Regression (LWR).")

    try:
        fish_df = load_data()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    X_raw, y, X_scaled, X_design, scaler = build_training_arrays(fish_df)

    st.sidebar.header("Model Controls")
    tau = st.sidebar.slider("Tau (bandwidth)", min_value=0.05, max_value=2.50, value=0.10, step=0.01)

    length_min = float(np.min(X_raw))
    length_max = float(np.max(X_raw))
    length_default = float(np.median(X_raw))
    length3_input = st.sidebar.slider(
        "Length3 for prediction",
        min_value=length_min,
        max_value=length_max,
        value=length_default,
        step=0.1,
    )

    # Train/evaluate on full dataset for interactive demo
    y_pred = lwr_predict(X_design, y, X_design, tau)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    x_query_scaled = scaler.transform(np.array([[length3_input]], dtype=float))
    x_query_design = np.hstack((np.ones((1, 1)), x_query_scaled))[0]
    pred_weight = locally_weighted_regression(X_design, y, tau, x_query_design)

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Weight", f"{pred_weight:.2f}")
    c2.metric("MSE", f"{mse:.2f}")
    c3.metric("R²", f"{r2:.4f}")

    st.subheader("Prediction Plot")
    sorted_idx = X_scaled[:, 0].argsort()
    x_pred_scaled = X_scaled[sorted_idx, 0]
    y_pred_sorted = y_pred[sorted_idx]

    x_query_scalar = float(x_query_scaled[0, 0])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(X_scaled[:, 0], y, color="royalblue", alpha=0.6, s=35, label="Actual Data")
    ax.plot(x_pred_scaled, y_pred_sorted, color="crimson", linewidth=2.4, label="LWR Fit")
    ax.scatter([x_query_scalar], [pred_weight], color="black", s=100, marker="X", label="Predicted Point")
    ax.set_xlabel("Length3 (Standardized)")
    ax.set_ylabel("Weight")
    ax.set_title(f"Fish Market LWR Fit (tau = {tau:.2f})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    st.subheader("Dataset Preview")
    st.dataframe(fish_df.head(10), use_container_width=True)


if __name__ == "__main__":
    main()
