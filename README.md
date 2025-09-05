# ASL Alphabet Recognition System

A complete machine learning application for American Sign Language (ASL) alphabet recognition using two different deep learning models: a custom CNN and VGG16 transfer learning.

## Overview

This project provides real-time ASL alphabet recognition through a web interface. Users can upload images of ASL hand signs and get predictions from two different AI models:

- **Custom CNN**: A lightweight, fast model trained from scratch
- **VGG16 Transfer Learning**: A more accurate model using pre-trained weights

## Project Structure

```
ProjetIA/
├── backend/           # FastAPI backend server
├── frontend/          # React TypeScript frontend
├── notebooks/         # Jupyter notebooks for model training
├── asl_dataset/       # Training and test datasets
├── scripts/           # Helper scripts for setup and running
├── tests/             # Test files
└── uploads/           # Uploaded images storage
```

## Features

- **Dual Model Prediction**: Choose between CNN or VGG16 models
- **Real-time Recognition**: Upload and get instant predictions
- **Model Comparison**: Compare results from both models
- **Prediction History**: Track previous predictions
- **Feedback System**: Provide feedback on prediction accuracy
- **REST API**: Complete API for integration with other applications

## Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **TensorFlow**: Deep learning framework
- **MongoDB**: Database for storing predictions and feedback
- **Uvicorn**: ASGI server

### Frontend
- **React**: JavaScript library for UI
- **TypeScript**: Type-safe JavaScript
- **Bootstrap**: CSS framework
- **Axios**: HTTP client

### Machine Learning
- **TensorFlow/Keras**: Model training and inference
- **OpenCV**: Image processing
- **NumPy**: Numerical computations

## Prerequisites

- Python 3.8+
- Node.js 16+
- MongoDB (local or cloud instance)
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/youcisla/ASL-AI.git
cd ProjetIA
```

### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install
```

### 4. Model Setup

Place your trained models in the `backend/models/` directory:

```
backend/models/
├── custom_cnn_asl_model.keras        # Custom CNN model
├── custom_cnn_labels.json            # CNN label mapping
├── vgg16_transfer_asl_final.keras    # VGG16 model
└── vgg16_transfer_labels.json        # VGG16 label mapping
```

### 5. Environment Configuration

Create a `.env` file in the root directory:

```env
# Database
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=asl_classifier

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173

# File Upload
MAX_FILE_SIZE_MB=10
UPLOADS_DIR=uploads

# Model Paths
ASL_MODEL_PATH=models/custom_cnn_asl_model.keras
LABELS_PATH=models/custom_cnn_labels.json
```

## Running the Application

### Option 1: Using Scripts (Recommended)

```bash
# Start backend
./scripts/start_backend.bat    # Windows
./scripts/start_backend.sh     # macOS/Linux

# Start frontend (in a new terminal)
./scripts/start_frontend.bat   # Windows
./scripts/start_frontend.sh    # macOS/Linux
```

### Option 2: Manual Commands

#### Start Backend Server

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Start Frontend Development Server

```bash
cd frontend
npm run dev
```

## Usage

1. **Access the Application**: Open your browser and go to `http://localhost:5173`

2. **Upload an Image**: 
   - Drag and drop an ASL alphabet image
   - Or click to select a file

3. **Choose Prediction Model**:
   - Click "Predict with CNN" for fast prediction
   - Click "Predict with VGG16" for more accurate prediction

4. **View Results**:
   - See top 3 predictions with confidence scores
   - View processing time
   - Provide feedback on accuracy

## API Endpoints

### Health Check
```
GET /health
```

### Predictions
```
POST /api/predict/cnn       # Predict using CNN model
POST /api/predict/vgg       # Predict using VGG16 model
POST /api/predict           # Default prediction (CNN)
```

### Labels
```
GET /api/labels             # Get all labels (CNN)
GET /api/labels/cnn         # Get CNN model labels
GET /api/labels/vgg         # Get VGG16 model labels
```

### Feedback and History
```
POST /api/feedback          # Submit prediction feedback
GET /api/history            # Get prediction history
```

## Model Training

The Jupyter notebooks for training models are located in the `notebooks/` directory. These notebooks contain the complete machine learning pipeline from data preprocessing to model deployment.

### Available Notebooks

#### 1. `from-scratch-vgg.ipynb` - Main Training Pipeline

This is the primary notebook containing the complete training process for both models. It includes:

**Data Loading and Preprocessing:**
- Loading ASL alphabet dataset (29 classes: A-Z, space, delete, nothing)
- Image preprocessing for both model types
- Data augmentation and normalization
- Train/validation split (80/20)

**Custom CNN Model:**
- Architecture: 3 convolutional layers (128→64→32 filters)
- Input: 60x60 grayscale images
- Layers: Conv2D + BatchNormalization + MaxPooling2D
- Dense layers: 256→128→64→29 neurons
- Dropout for regularization (0.2, 0.15)
- Training with callbacks (EarlyStopping, ModelCheckpoint)

**VGG16 Transfer Learning Model:**
- Pre-trained VGG16 base (ImageNet weights)
- Input: 224x224 RGB images
- Frozen feature extraction layers
- Custom classification head: GlobalAveragePooling2D + Dense layers
- Fine-tuning with lower learning rate (1e-4)

**Model Optimization:**
- EarlyStopping to prevent overfitting
- ReduceLROnPlateau for adaptive learning rate
- ModelCheckpoint to save best models
- Batch normalization for faster convergence

**Model Evaluation:**
- Accuracy plots for training/validation
- Performance comparison between models
- Prediction latency analysis

**Model Saving:**
- Automatic saving in multiple formats (.keras, .h5)
- Label mappings saved as JSON files
- Model metadata and specifications

#### 2. `asl-alphabet-sign.ipynb` - Data Exploration

**Dataset Analysis:**
- Statistical overview of the dataset
- Class distribution analysis
- Image quality assessment
- Sample visualization from each class

**Data Preprocessing Experiments:**
- Different image sizes comparison
- Color vs grayscale analysis
- Normalization techniques
- Augmentation strategies testing

#### 3. `projet-ia.ipynb` - Project Overview

**Methodology Documentation:**
- Problem definition and approach
- Literature review and related work
- Architecture design decisions
- Experimental setup and parameters

### Training Process Details

#### Step 1: Environment Setup
```python
# Required libraries
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
```

#### Step 2: Data Preparation
- **Dataset Structure**: 87,000 training images organized in class folders
- **Image Preprocessing**: Resize, normalize, and reshape for model input
- **Label Encoding**: Convert class names to categorical format
- **Data Augmentation**: Rotation, zoom, and shift for better generalization

#### Step 3: Model Architecture

**Custom CNN Architecture:**
```
Input Layer (60, 60, 1)
├── Conv2D(128, 3x3) + BatchNorm + ReLU + MaxPool
├── Conv2D(64, 3x3) + BatchNorm + ReLU + MaxPool  
├── Conv2D(32, 3x3) + BatchNorm + ReLU + MaxPool
├── Flatten
├── Dense(256) + ReLU + Dropout(0.2)
├── Dense(128) + ReLU + Dropout(0.15)
├── Dense(64) + ReLU
└── Dense(29) + Softmax
```

**VGG16 Transfer Learning:**
```
VGG16 Base (frozen)
├── GlobalAveragePooling2D
├── Dense(256) + ReLU + Dropout(0.3)
├── Dense(128) + ReLU + Dropout(0.2)
└── Dense(29) + Softmax
```

#### Step 4: Training Configuration

**Custom CNN Training:**
- Optimizer: Adam (default learning rate)
- Loss: Categorical Crossentropy
- Metrics: Accuracy
- Epochs: 10 (with early stopping)
- Batch Size: 32
- Validation Split: 20%

**VGG16 Training:**
- Optimizer: Adam (learning rate: 1e-4)
- Loss: Categorical Crossentropy
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Epochs: 10-12
- Batch Size: 32

#### Step 5: Model Evaluation

**Performance Metrics:**
- Training/Validation accuracy curves
- Loss curves analysis
- Confusion matrix for detailed class performance
- Top-3 accuracy for practical usage

**Model Comparison:**
| Metric | Custom CNN | VGG16 Transfer |
|--------|------------|----------------|
| Accuracy | ~85-90% | ~95-98% |
| Training Time | 30-45 min | 60-90 min |
| Model Size | ~50 MB | ~500 MB |
| Inference Speed | Fast | Moderate |

### Running the Training Notebooks

#### Prerequisites
```bash
# Install Jupyter and required packages
pip install jupyter
pip install tensorflow numpy opencv-python matplotlib seaborn
```

#### Execution Steps

1. **Start Jupyter Notebook:**
```bash
cd notebooks
jupyter notebook
```

2. **Configure Data Paths:**
   - Update data paths to point to your dataset location
   - Ensure dataset structure matches expected format

3. **Execute Cells Sequentially:**
   - Run data loading cells first
   - Execute preprocessing steps
   - Train Custom CNN model (cells 33-39)
   - Train VGG16 model (cells 40-47)

4. **Model Output:**
   - Models saved automatically to `backend/models/` directory
   - Training logs and plots displayed inline
   - Performance metrics calculated and displayed

#### Expected Output Files

After successful training, you'll have:
```
backend/models/
├── custom_cnn_asl_model.keras       # Custom CNN model
├── custom_cnn_asl_model.h5          # Custom CNN (legacy format)
├── custom_cnn_labels.json           # CNN label mappings
├── vgg16_transfer_asl_final.keras   # VGG16 final model
├── vgg16_transfer_asl_final.h5      # VGG16 (legacy format)
├── vgg16_transfer_asl_best.keras    # VGG16 best checkpoint
└── vgg16_transfer_labels.json       # VGG16 label mappings
```

### Notebook Customization

#### Modifying Model Architecture
```python
# Example: Adding more CNN layers
Model = Sequential([
    Conv2D(256, (3,3), activation='relu', input_shape=(60,60,1)),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    # Add your custom layers here
])
```

#### Hyperparameter Tuning
```python
# Experiment with different parameters
LEARNING_RATES = [1e-3, 1e-4, 1e-5]
BATCH_SIZES = [16, 32, 64]
DROPOUT_RATES = [0.1, 0.2, 0.3]
```

#### Data Augmentation
```python
# Custom augmentation pipeline
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False  # ASL signs shouldn't be flipped
)
```

### Troubleshooting Training

**Common Issues:**

1. **Out of Memory Error:**
   - Reduce batch size from 32 to 16 or 8
   - Use CPU training if GPU memory is limited

2. **Low Accuracy:**
   - Increase training epochs
   - Add more data augmentation
   - Adjust learning rate

3. **Overfitting:**
   - Increase dropout rates
   - Add more regularization
   - Use early stopping

4. **Training Too Slow:**
   - Use GPU acceleration
   - Reduce image resolution
   - Implement data pipeline optimization

## Development

### Running Tests

```bash
# Backend tests
cd backend
pytest

# API tests
python tests/test_api.py
```

### Code Formatting

```bash
# Frontend linting
cd frontend
npm run lint
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**:
   ```bash
   # Kill process on port 8000
   netstat -ano | findstr :8000    # Windows
   lsof -ti:8000 | xargs kill      # macOS/Linux
   ```

2. **MongoDB Connection Error**:
   - Ensure MongoDB is running
   - Check connection string in `.env` file

3. **Model Loading Error**:
   - Verify model files exist in `backend/models/`
   - Check file permissions

4. **Dependencies Issues**:
   ```bash
   # Recreate virtual environment
   rm -rf venv
   python -m venv venv
   pip install -r requirements.txt
   ```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Dataset

The project uses the ASL Alphabet dataset containing:
- 29 classes (A-Z, space, delete, nothing)
- 87,000 training images
- 29 test images

## Performance

### Model Specifications

| Model | Input Size | Architecture | Accuracy | Speed |
|-------|------------|--------------|----------|-------|
| Custom CNN | 60x60 grayscale | 3-layer CNN | ~85% | Fast |
| VGG16 Transfer | 224x224 RGB | VGG16 + Custom Head | ~95% | Moderate |

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- ASL Alphabet dataset from Kaggle
- TensorFlow and Keras communities
- FastAPI and React communities

## Contact

For questions or support, please open an issue on GitHub or contact the project maintainers.