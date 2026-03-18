import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


def train_and_save():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    feature_cols = df.columns.tolist()

    # Defensive preprocessing for consistency with notebook workflow
    X = df.fillna(df.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    cluster_profile = pd.DataFrame(X).assign(cluster=labels).groupby("cluster").mean().round(2)

    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(feature_cols, "feature_columns.pkl")
    joblib.dump(cluster_profile, "cluster_profile.pkl")

    print("Saved: model.pkl, scaler.pkl, feature_columns.pkl, cluster_profile.pkl")


if __name__ == "__main__":
    train_and_save()
