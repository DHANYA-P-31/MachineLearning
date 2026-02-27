import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


st.set_page_config(page_title="SVM Spam Detector", page_icon=":email:", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600&family=Source+Sans+3:wght@400;600&display=swap');
    
    /* Force light background */
    .stApp {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 50%, #e9ecef 100%) !important;
    }
    
    html, body, [class*="css"], .main, .block-container  {
        font-family: 'Source Sans 3', sans-serif;
        background-color: transparent !important;
        color: #1a1a1a !important;
    }
    .title {
        font-family: 'Playfair Display', serif;
        font-size: 2.4rem;
        margin-bottom: 0.3rem;
    }
    .subtle {
        color: #495057;
        font-size: 1rem;
    }
    .card {
        background: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    .pill {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 0.85rem;
        border: 1px solid #ced4da;
        background: #f8f9fa;
        margin-right: 6px;
    }
    
    /* Override Streamlit's dark mode elements */
    .stTextArea textarea, .stTextInput input {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
        border: 1px solid #dee2e6 !important;
    }
    
    .stMarkdown, div[data-testid="stMarkdownContainer"] {
        color: #1a1a1a !important;
    }
    
    .stMetric, .stProgress {
        color: #1a1a1a !important;
    }
    
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def build_model():
    # Get the path to the data file relative to this script
    data_path = Path(__file__).parent.parent / "data" / "spamhamdata.csv"
    
    df = pd.read_csv(data_path, sep="\t", header=None, names=["label", "text"])
    df = df.dropna()

    X_train, _, y_train, _ = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)

    model = LinearSVC()
    model.fit(X_train_vec, y_train)

    return df, vectorizer, model


df, vectorizer, model = build_model()

st.markdown("<div class='title'>SVM Spam Detector</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtle'>Detect spam emails with a support vector machine trained on the spam/ham dataset.</div>",
    unsafe_allow_html=True,
)

st.markdown("<span class='pill'>Model: Linear SVM</span><span class='pill'>TF-IDF</span>", unsafe_allow_html=True)

col_left, col_right = st.columns([1.2, 0.8], gap="large")

with col_left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Email Content")
    st.caption("Paste a message to classify as spam or ham.")

    default_text = "Congratulations! You have won a free ticket. Reply WIN now."
    message = st.text_area("Message text", value=default_text, height=180)

    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Prediction")
    st.caption("The model returns the class and a confidence score.")

    if message.strip():
        vector = vectorizer.transform([message])
        pred = model.predict(vector)[0]
        score = float(model.decision_function(vector)[0])
        confidence = 1.0 / (1.0 + np.exp(-abs(score)))

        label = "Spam" if pred == "spam" else "Ham"
        color = "#dc3545" if pred == "spam" else "#28a745"  # Red for spam, green for ham
        
        st.markdown(f"""
            <div style="margin: 20px 0;">
                <div style="color: #6c757d; font-size: 0.875rem; margin-bottom: 4px;">Prediction</div>
                <div style="color: {color}; font-size: 2.5rem; font-weight: 600; margin-bottom: 10px;">{label}</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.progress(confidence, text=f"Confidence: {confidence:.2%}")
        st.write("Decision score:", score)
    else:
        st.info("Enter a message to get a prediction.")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

with st.expander("Dataset overview"):
    st.write("Rows:", len(df))
    st.write("Labels:", sorted(df["label"].unique().tolist()))
    st.dataframe(df.head())
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
