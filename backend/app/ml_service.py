import json
import hashlib
import numpy as np
import tensorflow as tf
from PIL import Image
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import io
import logging
from app.config import settings

logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self):
        self.model = None
        self.labels = None
        self.model_loaded = False
    
    async def load_model(self):
        """Load the Keras model and labels"""
        try:
            # Load model
            model_path = Path(settings.asl_model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {settings.asl_model_path}")
            
            self.model = tf.keras.models.load_model(str(model_path))
            logger.info(f"Model loaded from {settings.asl_model_path}")
            
            # Load labels
            labels_path = Path(settings.labels_path)
            if not labels_path.exists():
                raise FileNotFoundError(f"Labels file not found: {settings.labels_path}")
            
            with open(labels_path, 'r') as f:
                self.labels = json.load(f)
            
            logger.info(f"Labels loaded: {len(self.labels)} classes")
            self.model_loaded = True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def read_labels(self) -> List[str]:
        """Get the list of class labels"""
        if self.labels is None:
            raise RuntimeError("Labels not loaded")
        return self.labels
    
    def is_loaded(self) -> bool:
        """Check if model and labels are loaded"""
        return self.model_loaded and self.model is not None and self.labels is not None


def safe_file_hash(file_bytes: bytes) -> str:
    """Generate SHA256 hash of file bytes"""
    return hashlib.sha256(file_bytes).hexdigest()


def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """Preprocess image for VGG16 model"""
    try:
        # Open image with PIL
        image = Image.open(io.BytesIO(file_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize((224, 224))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Apply VGG16 preprocessing
        img_array = tf.keras.applications.vgg16.preprocess_input(img_array.astype(np.float32))
        
        return img_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise ValueError(f"Could not process image: {e}")


def get_image_dimensions(file_bytes: bytes) -> Tuple[int, int]:
    """Get image width and height"""
    try:
        image = Image.open(io.BytesIO(file_bytes))
        return image.size  # (width, height)
    except Exception as e:
        logger.error(f"Error getting image dimensions: {e}")
        raise ValueError(f"Could not read image dimensions: {e}")


class MLService:
    def __init__(self):
        self.model_service = ModelService()
    
    async def initialize(self):
        """Initialize the ML service"""
        await self.model_service.load_model()
    
    def predict_topk(self, img_array: np.ndarray, k: int = 3) -> List[Dict[str, any]]:
        """Make prediction and return top-k results"""
        if not self.model_service.is_loaded():
            raise RuntimeError("Model not loaded")
        
        try:
            # Get predictions
            predictions = self.model_service.model.predict(img_array, verbose=0)[0]
            
            # Get top-k indices
            top_indices = np.argsort(predictions)[-k:][::-1]
            
            # Build results
            results = []
            for idx in top_indices:
                results.append({
                    "label": self.model_service.labels[idx],
                    "prob": float(predictions[idx])
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise RuntimeError(f"Prediction failed: {e}")
    
    def get_labels(self) -> List[str]:
        """Get all available labels"""
        return self.model_service.read_labels()
    
    def is_ready(self) -> bool:
        """Check if service is ready for predictions"""
        return self.model_service.is_loaded()
    
    def get_num_labels(self) -> int:
        """Get number of labels"""
        if self.model_service.labels:
            return len(self.model_service.labels)
        return 0

# Global ML service instance
ml_service = MLService()
