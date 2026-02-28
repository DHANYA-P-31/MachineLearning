# Facial Recognition Web Application

A web-based facial recognition system powered by Artificial Neural Network (ANN) using Streamlit.

## Features

- 🖼️ **Image Upload**: Upload face images in JPG, JPEG, PNG, or BMP format
- 🎯 **Real-time Prediction**: Get instant recognition results with confidence scores
- 📊 **Top 3 Predictions**: View the most likely matches with probability percentages
- 👁️ **Image Preview**: See both original and preprocessed (64x64 grayscale) versions
- 🎨 **Modern UI**: Clean, gradient-styled interface with intuitive controls

## Model Details

- **Architecture**: Multi-Layer Perceptron (MLP)
- **Layers**: Input (4096) → Hidden (512) → Hidden (256) → Output (31 classes)
- **Activation**: ReLU
- **Optimizer**: Adam with early stopping
- **Training Dataset**: Local faces from `data/Dataset/Faces`

## Setup Instructions

### 1. Train the Model

First, run the Jupyter notebook to train and save the model:

```bash
# Open the notebook
jupyter notebook facialRecognition.ipynb

# Run all cells up to and including "Save Model for Web Application"
```

This will create three files in the `FacialRecognitionANN` folder:
- `facial_recognition_model.pkl` - Trained ANN model
- `scaler.pkl` - Feature scaler
- `class_names.pkl` - Person name labels

### 2. Install Dependencies

```bash
pip install streamlit scikit-learn pillow joblib numpy
```

### 3. Run the Web Application

```bash
streamlit run FacialRecognitionANN/app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Usage

1. **Upload Image**: Click the upload button and select a face image
2. **Preview**: View the original and preprocessed image
3. **Predict**: Click the "🎯 Predict" button to identify the person
4. **Results**: See the predicted person name, confidence score, and top 3 matches

## File Structure

```
FacialRecognitionANN/
├── app.py                              # Streamlit web application
├── facialRecognition.ipynb             # Training notebook
├── README_APP.md                       # This file
├── facial_recognition_model.pkl        # Trained model (generated)
├── scaler.pkl                          # Feature scaler (generated)
└── class_names.pkl                     # Class labels (generated)
```

## Requirements

- Python 3.8+
- streamlit
- scikit-learn
- Pillow
- joblib
- numpy

## Technical Notes

- Images are automatically converted to 64x64 grayscale
- Input features are standardized using the saved scaler
- The model outputs probabilities for all 31 classes
- Top 3 predictions are displayed with confidence percentages

## Troubleshooting

**Model files not found error:**
- Make sure you've run the notebook and executed the "Save Model" cell
- Check that `.pkl` files exist in the `FacialRecognitionANN` folder

**Low accuracy:**
- Ensure the uploaded image clearly shows a face
- Try images similar to the training data quality
- Images are resized to 64x64, so quality may vary

## Future Enhancements

- [ ] Webcam integration for real-time recognition
- [ ] Batch image processing
- [ ] Model performance metrics display
- [ ] Image preprocessing options
- [ ] Export prediction results
