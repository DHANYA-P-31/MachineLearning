import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


st.set_page_config(page_title="Perceptron Classifier", layout="centered")
st.title("Perceptron Classifier")
st.caption("Simple binary classification on Breast Cancer dataset")

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

max_iter = st.slider("Max Iterations", 100, 2000, 1000, 100)

if st.button("Train and Evaluate", type="primary"):
    model = Perceptron(max_iter=max_iter, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.write("Accuracy:", round(float(acc), 4))
