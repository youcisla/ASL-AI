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
            # Load labels first
            labels_path = Path(settings.labels_path)
            if not labels_path.exists():
                logger.warning(f"Labels file not found: {settings.labels_path}, using default labels")
                self.labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
                              "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
                              "del", "nothing", "space"]
            else:
                with open(labels_path, 'r') as f:
                    self.labels = json.load(f)
            
            logger.info(f"Labels loaded: {len(self.labels)} classes")
            
            # Try to load model
            model_path = Path(settings.asl_model_path)
            if not model_path.exists():
                logger.warning(f"Model file not found: {settings.asl_model_path}")
                logger.info("Running in DEMO MODE - using random predictions")
                self.model = None
                self.model_loaded = False
            else:
                self.model = tf.keras.models.load_model(str(model_path))
                logger.info(f"Model loaded from {settings.asl_model_path}")
                self.model_loaded = True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Falling back to DEMO MODE")
            self.model = None
            self.model_loaded = False
    
    def read_labels(self) -> List[str]:
        """Get the list of class labels"""
        if self.labels is None:
            return ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
                   "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
                   "del", "nothing", "space"]
        return self.labels
    
    def is_loaded(self) -> bool:
        """Check if model and labels are loaded"""
        return self.model_loaded and self.model is not None and self.labels is not None


def safe_file_hash(file_bytes: bytes) -> str:
    """Generate SHA256 hash of file bytes"""
    return hashlib.sha256(file_bytes).hexdigest()


def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """Preprocess image for ASL CNN model (64x64 grayscale)"""
    try:
        # Open image with PIL
        image = Image.open(io.BytesIO(file_bytes))
        
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to model input size (64x64)
        image = image.resize((64, 64))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Normalize to 0-1 range
        img_array = img_array.astype(np.float32) / 255.0
        
        # Reshape to (1, 64, 64, 1) for CNN input
        img_array = img_array.reshape(1, 64, 64, 1)
        
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
        try:
            labels = self.model_service.read_labels()
            
            if self.model_service.is_loaded() and self.model_service.model is not None:
                # Real model prediction
                predictions = self.model_service.model.predict(img_array, verbose=0)[0]
                top_indices = np.argsort(predictions)[-k:][::-1]
                
                results = []
                for idx in top_indices:
                    results.append({
                        "label": labels[idx],
                        "prob": float(predictions[idx])
                    })
            else:
                # Demo mode - generate random but realistic predictions
                import random
                logger.info("Using demo predictions (no model loaded)")
                
                # Select k random labels
                selected_labels = random.sample(labels, min(k, len(labels)))
                
                # Generate decreasing probabilities
                results = []
                base_prob = random.uniform(0.6, 0.9)  # Start with high confidence
                for i, label in enumerate(selected_labels):
                    prob = base_prob * (0.8 ** i)  # Decreasing confidence
                    results.append({
                        "label": label,
                        "prob": float(prob)
                    })
                
                # Sort by probability descending
                results.sort(key=lambda x: x["prob"], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise RuntimeError(f"Prediction failed: {e}")
    
    def get_labels(self) -> List[str]:
        """Get all available labels"""
        return self.model_service.read_labels()
    
    def is_ready(self) -> bool:
        """Check if service is ready for predictions"""
        return True  # Always ready, even in demo mode
    
    def get_num_labels(self) -> int:
        """Get number of labels"""
        labels = self.model_service.read_labels()
        return len(labels) if labels else 0

# Global ML service instance
ml_service = MLService()
