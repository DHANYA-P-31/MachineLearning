import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score


st.set_page_config(page_title="Agglomerative Clustering", layout="centered")
st.title("Agglomerative Clustering")
st.caption("Simple hierarchical bottom-up clustering on Iris data")

iris = load_iris()
X = iris.data[:, :2]

n_clusters = st.slider("Number of clusters", 2, 6, 3, 1)

if st.button("Run", type="primary"):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)

    st.write("Silhouette Score:", round(float(score), 4))

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=35)
    ax.set_title("Agglomerative Clustering (Iris)")
    ax.set_xlabel("Sepal Length")
    ax.set_ylabel("Sepal Width")
    fig.colorbar(scatter, ax=ax)
    st.pyplot(fig)
