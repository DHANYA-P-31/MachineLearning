import streamlit as st
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


st.set_page_config(page_title="KNN Classifier", layout="centered")
st.title("KNN Classifier")
st.caption("Simple multiclass classification on Iris")

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

k = st.slider("Number of neighbors (k)", 1, 20, 5)

if st.button("Train and Evaluate", type="primary"):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.write("Accuracy:", round(float(acc), 4))
