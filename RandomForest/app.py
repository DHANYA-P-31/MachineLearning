import pathlib

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier


@st.cache_data
def load_data(csv_path: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


@st.cache_resource
def train_model(data: pd.DataFrame) -> RandomForestClassifier:
    X = data.drop("target", axis=1)
    y = data["target"]
    model = RandomForestClassifier(
        n_estimators=100,
        criterion="entropy",
        random_state=42,
    )
    model.fit(X, y)
    return model


def main() -> None:
    st.set_page_config(
        page_title="Heart Disease Prediction",
        layout="centered",
        initial_sidebar_state="auto",
        menu_items=None,
    )
    st.title("Heart Disease Prediction (Random Forest)")
    st.write(
        "Provide patient values to get a prediction. "
        "Inputs follow the same feature order as the dataset."
    )

    csv_path = pathlib.Path(__file__).with_name("heart.csv")
    data = load_data(csv_path)
    model = train_model(data)

    st.subheader("Patient Inputs")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("age", min_value=1, max_value=120, value=60)
        sex = st.selectbox("sex (1=Male, 0=Female)", options=[0, 1], index=1)
        cp = st.selectbox("cp (0-3)", options=[0, 1, 2, 3], index=0)
        trestbps = st.number_input("trestbps", min_value=80, max_value=220, value=130)
        chol = st.number_input("chol", min_value=100, max_value=600, value=250)
        fbs = st.selectbox("fbs (1=True, 0=False)", options=[0, 1], index=0)
        restecg = st.selectbox("restecg (0-2)", options=[0, 1, 2], index=1)

    with col2:
        thalach = st.number_input("thalach", min_value=60, max_value=220, value=150)
        exang = st.selectbox("exang (1=Yes, 0=No)", options=[0, 1], index=0)
        oldpeak = st.number_input("oldpeak", min_value=0.0, max_value=10.0, value=1.5)
        slope = st.selectbox("slope (0-2)", options=[0, 1, 2], index=1)
        ca = st.selectbox("ca (0-3)", options=[0, 1, 2, 3], index=1)
        thal = st.selectbox("thal (1=Normal, 2=Fixed, 3=Reversible)", options=[1, 2, 3], index=2)

    input_data = pd.DataFrame(
        {
            "age": [age],
            "sex": [sex],
            "cp": [cp],
            "trestbps": [trestbps],
            "chol": [chol],
            "fbs": [fbs],
            "restecg": [restecg],
            "thalach": [thalach],
            "exang": [exang],
            "oldpeak": [oldpeak],
            "slope": [slope],
            "ca": [ca],
            "thal": [thal],
        }
    )

    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]
        label = "Heart Disease" if prediction == 1 else "No Heart Disease"
        st.success(f"Prediction: {label}")
        st.write(f"Probability (No Disease): {proba[0]:.2f}")
        st.write(f"Probability (Heart Disease): {proba[1]:.2f}")

    with st.expander("Feature Reference"):
        st.markdown(
            """
            - age: Age in years
            - sex: 1 = Male, 0 = Female
            - cp: Chest pain type (0-3)
            - trestbps: Resting blood pressure (mm Hg)
            - chol: Serum cholesterol (mg/dl)
            - fbs: Fasting blood sugar > 120 mg/dl (1=True, 0=False)
            - restecg: Resting electrocardiographic results (0-2)
            - thalach: Maximum heart rate achieved
            - exang: Exercise induced angina (1=Yes, 0=No)
            - oldpeak: ST depression induced by exercise relative to rest
            - slope: Slope of the peak exercise ST segment (0-2)
            - ca: Number of major vessels (0-3) colored by fluoroscopy
            - thal: Thalassemia (1=Normal, 2=Fixed, 3=Reversible)
            """
        )


if __name__ == "__main__":
    main()
