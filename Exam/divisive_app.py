import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import BisectingKMeans
from sklearn.datasets import make_blobs


st.set_page_config(page_title="Divisive Clustering", layout="centered")
st.title("Divisive Clustering")
st.caption("Top-down clustering")

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)


k = st.slider("Number of clusters", 2, 6, 4, 1)

if st.button("Run", type="primary"):
    model = BisectingKMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X)

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=30)
    ax.set_title("Divisive Clustering (Top-Down)")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    fig.colorbar(scatter, ax=ax)
    st.pyplot(fig)
