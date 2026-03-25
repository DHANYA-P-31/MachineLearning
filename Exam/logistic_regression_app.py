import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


st.set_page_config(page_title="Logistic Regression", layout="centered")
st.title("Logistic Regression")
st.caption("Simple binary classification")

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

c_val = st.slider("Regularization strength inverse (C)", 0.01, 10.0, 1.0)

if st.button("Train and Evaluate", type="primary"):
    model = LogisticRegression(C=c_val, max_iter=2000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.write("Accuracy:", round(float(acc), 4))
