# Facial Recognition with Artificial Neural Network

A deep learning implementation of face verification using Siamese Neural Networks on the Labeled Faces in the Wild (LFW) dataset.

## Overview

This project implements a face verification system that can determine whether two face images belong to the same person. The model uses a Siamese Neural Network architecture with shared weights to learn meaningful facial embeddings.

## Features

- **Siamese Neural Network Architecture**: Dual-pathway network with shared weights
- **Deep CNN Feature Extractor**: 3 convolutional blocks with batch normalization
- **128-dimensional Face Embeddings**: Compact representation of facial features
- **Euclidean Distance Metric**: Measures similarity between face embeddings
- **Comprehensive Training**: Includes early stopping, learning rate scheduling, and model checkpointing
- **Detailed Evaluation**: Accuracy, precision, recall, and confusion matrix analysis

## Dataset

- **Source**: Labeled Faces in the Wild (LFW) - Funneled version
- **Location**: `../data/lfw-funneled/lfw_funneled/`
- **Format**: Face pairs for verification task
- **Structure**: 
  - Matching pairs (same person)
  - Non-matching pairs (different people)
  - Images resized to 128x128 pixels

## Model Architecture

### Base Network (Feature Extractor)
```
- Block 1: 2x Conv2D (64 filters) + BatchNorm + MaxPooling + Dropout
- Block 2: 2x Conv2D (128 filters) + BatchNorm + MaxPooling + Dropout
- Block 3: 2x Conv2D (256 filters) + BatchNorm + MaxPooling + Dropout
- Dense Layers: 512 → 256 → 128 (embedding)
```

### Siamese Network
```
- Input: Two face images
- Processing: Both images through shared base network
- Distance: Euclidean distance between embeddings
- Output: Similarity score (0-1)
```

## Implementation Details

1. **Data Loading**: Parse LFW pairs file and load corresponding images
2. **Preprocessing**: Resize to 128x128, normalize to [0,1]
3. **Model Training**: 
   - Optimizer: Adam (lr=0.001)
   - Loss: Binary crossentropy
   - Batch size: 32
   - Epochs: Up to 50 (with early stopping)
4. **Evaluation**: Comprehensive metrics and visualizations

## Results

The model achieves:
- **High accuracy** in distinguishing same vs. different people
- **Clear separation** in embedding distance distributions
- **Robust performance** across the test set

## Usage

### Training
Run all cells in the notebook sequentially to:
1. Load and preprocess the dataset
2. Build the Siamese network
3. Train the model
4. Evaluate performance
5. Visualize results

### Inference
```python
# Load the saved model
model = keras.models.load_model('facial_recognition_siamese_model.h5')

# Verify two faces
is_same, confidence = verify_faces(img1_path, img2_path, model)
```

## Files Generated

- `facial_recognition_siamese_model.h5` - Complete trained model
- `facial_recognition_base_network.h5` - Feature extractor only
- `best_siamese_model.h5` - Best checkpoint during training

## Requirements

- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- OpenCV (cv2)
- scikit-learn

## Key Concepts

- **Siamese Networks**: Neural network architecture that learns to compare inputs
- **Face Embeddings**: Compact vector representations of faces
- **Metric Learning**: Learning a distance metric for similarity comparison
- **One-shot Learning**: Ability to verify faces with minimal training examples

## Applications

- Face verification systems
- Identity authentication
- Access control
- Duplicate face detection
- Security systems

## Notes

- The model uses a similarity threshold of 0.5 for verification
- Batch normalization helps stabilize training
- Dropout prevents overfitting
- Early stopping ensures optimal generalization
