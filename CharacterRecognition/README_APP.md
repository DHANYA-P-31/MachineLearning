# Character Recognition UI

A light-themed web interface for character recognition using a trained MLP neural network.

## Features

- 🎨 **Clean Light Mode UI** - Modern, professional appearance
- 📸 **Image Upload** - Drag & drop or select image files (JPG, PNG, BMP)
- 🎯 **Character Prediction** - Real-time character recognition
- 📊 **Top-K Predictions** - View top predictions with confidence scores
- ⚙️ **Configurable** - Adjust number of predictions to display

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run app.py
```

The application will start on `http://localhost:8501`

## Usage

1. Open the application in your browser
2. Upload an image containing a character
3. Adjust the top-K setting if desired (default: 3)
4. View the prediction results with confidence scores
5. Check the top predictions to see alternative matches

## Model Details

- **Algorithm**: Multi-layer Perceptron (MLP)
- **Preprocessing**: PCA dimensionality reduction + StandardScaler normalization
- **Input**: Grayscale image (normalized to 0-1)
- **Output**: Predicted character class with probabilities

## Files

- `app.py` - Streamlit application
- `mlp_character_recognition_bundle.pkl` - Trained model bundle containing:
  - Trained MLP classifier
  - PCA transformer
  - StandardScaler
  - Label encoder
- `characterRecognition.ipynb` - Notebook with model training code
