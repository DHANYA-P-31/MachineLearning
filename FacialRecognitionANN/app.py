import io
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
from PIL import Image


@st.cache_resource
def load_model():
    """Load the trained facial recognition model and related components"""
    model_dir = Path(__file__).parent
    
    try:
        mlp = joblib.load(model_dir / "facial_recognition_model.pkl")
        scaler = joblib.load(model_dir / "scaler.pkl")
        class_names = joblib.load(model_dir / "class_names.pkl")
        return mlp, scaler, class_names
    except FileNotFoundError:
        st.error("Model files not found. Please run the notebook first to train and save the model.")
        st.stop()


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Convert uploaded image to model input format"""
    # Convert to grayscale and resize to 64x64
    image_gray = image.convert('L').resize((64, 64))
    # Convert to array and normalize
    image_array = np.asarray(image_gray, dtype=np.float32) / 255.0
    # Flatten to 4096 features
    return image_array.reshape(-1)


def predict_face(image_data: np.ndarray, mlp, scaler, class_names, top_k=3):
    """Predict the person in the image"""
    # Scale the input
    image_scaled = scaler.transform(image_data.reshape(1, -1))
    
    # Get prediction and probabilities
    prediction = int(mlp.predict(image_scaled)[0])
    probabilities = mlp.predict_proba(image_scaled)[0] * 100
    
    # Get top K predictions
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    top_predictions = [
        (class_names[idx], probabilities[idx]) 
        for idx in top_indices
    ]
    
    return prediction, class_names[prediction], probabilities[prediction], top_predictions


def main():
    st.set_page_config(
        page_title="Face Recognition - ANN",
        page_icon="👤",
        layout="centered"
    )
    
    # Custom CSS for styling
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
        
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Make all Streamlit text white by default */
        .stApp, .stApp p, .stApp span, .stApp div, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
            color: white !important;
        }
        
        /* File uploader text */
        .stFileUploader label, .stFileUploader div, .stFileUploader span {
            color: #333 !important;
        }
        
        /* Button text */
        .stButton button {
            color: white !important;
            font-weight: 600 !important;
        }
        
        .main-title {
            font-family: 'Poppins', sans-serif;
            font-size: 2.5rem;
            font-weight: 700;
            text-align: center;
            color: white !important;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        .subtitle {
            font-family: 'Poppins', sans-serif;
            text-align: center;
            color: rgba(255,255,255,0.95) !important;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }
        
        .prediction-box {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            margin: 20px 0;
        }
        
        /* Override white text inside white boxes */
        .prediction-box *, .prediction-box h1, .prediction-box h2, .prediction-box h3, 
        .prediction-box h4, .prediction-box p, .prediction-box span, .prediction-box div {
            color: #333 !important;
        }
        
        .result-label {
            font-family: 'Poppins', sans-serif;
            font-size: 1.1rem;
            color: #666 !important;
            margin-bottom: 5px;
        }
        
        .result-value {
            font-family: 'Poppins', sans-serif;
            font-size: 2rem;
            font-weight: 700;
            color: #667eea !important;
            margin-bottom: 10px;
        }
        
        .confidence-bar {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            height: 30px;
            border-radius: 15px;
            margin: 10px 0;
            display: flex;
            align-items: center;
            padding: 0 15px;
            color: white !important;
            font-weight: 600;
        }
        
        .top-prediction {
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 12px 15px;
            margin: 8px 0;
            border-radius: 5px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .top-prediction * {
            color: #333 !important;
        }
        
        .upload-box {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            margin: 20px 0;
        }
        
        /* Override white text inside upload box */
        .upload-box *, .upload-box h1, .upload-box h2, .upload-box h3, 
        .upload-box h4, .upload-box p, .upload-box span, .upload-box div {
            color: #333 !important;
        }
        
        /* Info box styling */
        .stAlert {
            background-color: rgba(255, 255, 255, 0.95) !important;
        }
        
        .stAlert * {
            color: #333 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-title">👤 Face Recognition System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Powered by Artificial Neural Network</div>', unsafe_allow_html=True)
    
    # Load model
    mlp, scaler, class_names = load_model()
    
    # File uploader
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload Face Image")
    uploaded_file = st.file_uploader(
        "Choose a face image...",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown("#### 🖼️ Uploaded Image")
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown("#### 🔍 Preview (64x64)")
            preview_img = image.convert('L').resize((64, 64))
            st.image(preview_img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Predict button
        if st.button("🎯 Predict", use_container_width=True, type="primary"):
            with st.spinner("Analyzing face..."):
                # Preprocess and predict
                image_data = preprocess_image(image)
                pred_id, pred_name, confidence, top_predictions = predict_face(
                    image_data, mlp, scaler, class_names
                )
                
                # Display main prediction
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown("### 🎯 Prediction Result")
                st.markdown(f'<div class="result-label">Identified Person:</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="result-value">{pred_name}</div>', unsafe_allow_html=True)
                
                # Confidence bar
                st.markdown(f'<div class="confidence-bar" style="width: {confidence}%;">{confidence:.2f}%</div>', 
                           unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Top 3 predictions
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown("### 📊 Top 3 Predictions")
                for rank, (name, conf) in enumerate(top_predictions, 1):
                    medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
                    st.markdown(f'''
                        <div class="top-prediction">
                            <span><strong>{medal} {name}</strong></span>
                            <span style="color: #667eea; font-weight: 600;">{conf:.2f}%</span>
                        </div>
                    ''', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.info("👆 Please upload a face image to begin recognition")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: white; opacity: 0.8;'>
            <p>Built with Streamlit | ANN Model with 512-256 hidden layers</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
