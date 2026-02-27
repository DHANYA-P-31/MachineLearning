import streamlit as st
import numpy as np
import joblib
from PIL import Image
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="Character Recognition",
    page_icon="🔤",
    layout="centered",
    initial_sidebar_state="auto"
)

# Light mode styling
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
    
    .subtitle {
        color: #495057;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    .card {
        background: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin: 1rem 0;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #f0f4f8 0%, #ffffff 100%);
        border: 2px solid #dee2e6;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    
    .confidence-high {
        color: #28a745;
        font-weight: 600;
    }
    
    .confidence-medium {
        color: #ffc107;
        font-weight: 600;
    }
    
    .confidence-low {
        color: #dc3545;
        font-weight: 600;
    }
    
    .pill {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        font-size: 0.85rem;
        border: 1px solid #ced4da;
        background: #f8f9fa;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    
    /* Override Streamlit's default elements */
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
    
    .stButton > button {
        background-color: #007bff !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
    }
    
    .stButton > button:hover {
        background-color: #0056b3 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model():
    """Load the trained model bundle"""
    model_path = Path(__file__).parent / "mlp_character_recognition_bundle.pkl"
    bundle = joblib.load(model_path)
    return bundle

@st.cache_resource
def get_model_info():
    """Get model components"""
    bundle = load_model()
    return bundle

def predict_character(image, bundle, return_top_k=3):
    """
    Predict character class from an image
    
    Args:
        image: PIL Image object
        bundle: Dictionary containing model, pca, scaler, label encoder
        return_top_k: Return top K predictions
        
    Returns:
        Dictionary with prediction results
    """
    try:
        # Get model components
        model = bundle['model']
        pca = bundle['pca']
        scaler = bundle['scaler']
        le = bundle['label_encoder']

        def get_target_size(expected_features, aspect_ratio):
            """Pick image size (w, h) whose product matches expected_features."""
            if expected_features <= 0:
                raise ValueError("Invalid expected feature count")
            root = int(np.sqrt(expected_features))
            if root * root == expected_features:
                return (root, root)

            best_w, best_h = expected_features, 1
            best_delta = float("inf")
            for h in range(root, 0, -1):
                if expected_features % h == 0:
                    w = expected_features // h
                    ratio = w / h
                    delta = abs(ratio - aspect_ratio)
                    if delta < best_delta:
                        best_delta = delta
                        best_w, best_h = w, h
                    if best_delta == 0:
                        break
            return (best_w, best_h)
        
        # Preprocess image
        img = image.convert('L')

        expected_features = pca.n_features_in_
        aspect_ratio = img.width / img.height
        target_w, target_h = get_target_size(expected_features, aspect_ratio)

        # Resize image to match PCA training dimensions
        img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        
        img_array = np.array(img, dtype=np.float32).flatten()
        img_array /= 255.0
        
        # Transform with PCA and scale
        img_pca = pca.transform(img_array.reshape(1, -1))
        img_scaled = scaler.transform(img_pca)
        
        # Make prediction
        prediction = model.predict(img_scaled)[0]
        probabilities = model.predict_proba(img_scaled)[0]
        
        # Get top-k predictions
        top_k_indices = np.argsort(probabilities)[-return_top_k:][::-1]
        top_k_predictions = [
            (le.inverse_transform([idx])[0], probabilities[idx]) 
            for idx in top_k_indices
        ]
        
        return {
            'predicted_class': le.inverse_transform([prediction])[0],
            'confidence': float(probabilities[prediction]),
            'top_k_predictions': top_k_predictions,
            'all_probabilities': probabilities
        }
    except Exception as e:
        return {'error': str(e)}

# Main UI
st.markdown('<h1 class="title">🔤 Character Recognition</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an image to recognize the character</p>', unsafe_allow_html=True)

# Create columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp"],
        label_visibility="collapsed"
    )

with col2:
    st.subheader("⚙️ Settings")
    top_k = st.slider(
        "Show top K predictions",
        min_value=1,
        max_value=10,
        value=3
    )

# Process uploaded image
if uploaded_file is not None:
    # Display uploaded image
    st.subheader("📸 Uploaded Image")
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)
    
    # Make prediction
    bundle = get_model_info()
    
    with st.spinner("🔍 Analyzing image..."):
        result = predict_character(image, bundle, return_top_k=top_k)
    
    if 'error' not in result:
        # Display prediction results
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### Predicted Character")
            st.markdown(f"<h2 style='font-size: 3rem; margin: 0;'>{result['predicted_class']}</h2>", unsafe_allow_html=True)
        
        with col2:
            confidence_percent = result['confidence'] * 100
            if confidence_percent >= 80:
                confidence_class = "confidence-high"
            elif confidence_percent >= 60:
                confidence_class = "confidence-medium"
            else:
                confidence_class = "confidence-low"
            
            st.markdown("### Confidence")
            st.markdown(
                f"<p class='{confidence_class}' style='font-size: 1.5rem; margin: 0;'>{confidence_percent:.2f}%</p>",
                unsafe_allow_html=True
            )
        
        # Top K predictions
        if len(result['top_k_predictions']) > 1:
            st.subheader("🏆 Top Predictions")
            
            for idx, (char, confidence) in enumerate(result['top_k_predictions'], 1):
                col1, col2, col3 = st.columns([0.5, 2, 1])
                with col1:
                    st.markdown(f"**#{idx}**")
                with col2:
                    st.markdown(f"**{char}**")
                with col3:
                    st.progress(confidence, text=f"{confidence*100:.1f}%")
    else:
        st.error(f"❌ Error: {result['error']}")

# Footer
st.markdown('---')
st.markdown(
    """
    <div style='text-align: center; color: #495057; font-size: 0.9rem;'>
    <p>Character Recognition using MLP Neural Network</p>
    </div>
    """,
    unsafe_allow_html=True
)
