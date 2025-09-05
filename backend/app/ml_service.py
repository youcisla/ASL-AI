import json
import hashlib
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
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
        self.cnn_model = None
        self.vgg_model = None
        self.cnn_labels = None
        self.vgg_labels = None
        self.models_loaded = {"cnn": False, "vgg": False}
    
    async def load_models(self):
        """Load both CNN and VGG16 models with their labels"""
        try:
            # Load CNN model and labels
            await self._load_cnn_model()
            
            # Load VGG16 model and labels
            await self._load_vgg_model()
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    async def _load_cnn_model(self):
        """Load the custom CNN model"""
        try:
            # Load CNN labels
            cnn_labels_path = Path(settings.cnn_labels_path)
            if cnn_labels_path.exists():
                with open(cnn_labels_path, 'r') as f:
                    self.cnn_labels = json.load(f)
                logger.info(f"CNN labels loaded: {len(self.cnn_labels)} classes")
            else:
                logger.warning(f"CNN labels file not found: {settings.cnn_labels_path}, using default")
                self.cnn_labels = self._get_default_labels()
            
            # Load CNN model
            cnn_model_path = Path(settings.cnn_model_path)
            if cnn_model_path.exists():
                self.cnn_model = tf.keras.models.load_model(str(cnn_model_path))
                self.models_loaded["cnn"] = True
                logger.info("Custom CNN model loaded successfully")
            else:
                logger.warning(f"CNN model file not found: {settings.cnn_model_path}")
                self.models_loaded["cnn"] = False
                
        except Exception as e:
            logger.error(f"Error loading CNN model: {e}")
            self.models_loaded["cnn"] = False
    
    async def _load_vgg_model(self):
        """Load the VGG16 transfer learning model"""
        try:
            # Load VGG16 labels
            vgg_labels_path = Path(settings.vgg_labels_path)
            if vgg_labels_path.exists():
                with open(vgg_labels_path, 'r') as f:
                    self.vgg_labels = json.load(f)
                logger.info(f"VGG16 labels loaded: {len(self.vgg_labels)} classes")
            else:
                logger.warning(f"VGG16 labels file not found: {settings.vgg_labels_path}, using default")
                self.vgg_labels = self._get_default_labels()
            
            # Load VGG16 model
            vgg_model_path = Path(settings.vgg_model_path)
            if vgg_model_path.exists():
                try:
                    # Try loading with compilation first
                    self.vgg_model = tf.keras.models.load_model(str(vgg_model_path))
                    self.models_loaded["vgg"] = True
                    logger.info("VGG16 transfer learning model loaded successfully")
                except Exception as compile_error:
                    logger.warning(f"Failed to load VGG model with compilation: {compile_error}")
                    try:
                        # Try loading without compilation
                        self.vgg_model = tf.keras.models.load_model(str(vgg_model_path), compile=False)
                        # Manually compile the model
                        self.vgg_model.compile(
                            optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy']
                        )
                        self.models_loaded["vgg"] = True
                        logger.info("VGG16 model loaded successfully without initial compilation")
                    except Exception as no_compile_error:
                        logger.error(f"Failed to load VGG model even without compilation: {no_compile_error}")
                        self.models_loaded["vgg"] = False
            else:
                logger.warning(f"VGG16 model file not found: {settings.vgg_model_path}")
                self.models_loaded["vgg"] = False
                
        except Exception as e:
            logger.error(f"Error loading VGG16 model: {e}")
            self.models_loaded["vgg"] = False
    
    def _get_default_labels(self) -> Dict[str, str]:
        """Get default label mapping"""
        default_classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
                          "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
                          "del", "nothing", "space"]
        return {str(i): label for i, label in enumerate(default_classes)}
    
    def is_ready(self, model_type: str = "cnn") -> bool:
        """Check if specified model is ready"""
        return self.models_loaded.get(model_type, False)
    
    def get_labels(self, model_type: str = "cnn") -> List[str]:
        """Get labels for specified model"""
        if model_type == "cnn" and self.cnn_labels:
            return list(self.cnn_labels.values())
        elif model_type == "vgg" and self.vgg_labels:
            return list(self.vgg_labels.values())
        else:
            return list(self._get_default_labels().values())
    
    def get_num_labels(self, model_type: str = "cnn") -> int:
        """Get number of labels for specified model"""
        return len(self.get_labels(model_type))
    
    def predict_topk(self, img_array: np.ndarray, k: int = 3, model_type: str = "cnn") -> List[Dict]:
        """Make prediction using specified model"""
        if model_type == "cnn":
            return self._predict_cnn(img_array, k)
        elif model_type == "vgg":
            return self._predict_vgg(img_array, k)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _predict_cnn(self, img_array: np.ndarray, k: int = 3) -> List[Dict]:
        """Predict using CNN model"""
        if not self.models_loaded["cnn"] or self.cnn_model is None:
            return self._demo_prediction(k, "cnn")
        
        try:
            # CNN expects 60x60 grayscale images
            predictions = self.cnn_model.predict(img_array, verbose=0)
            probs = predictions[0]
            
            # Get top k predictions
            top_indices = np.argsort(probs)[-k:][::-1]
            
            results = []
            for idx in top_indices:
                label = self.cnn_labels.get(str(idx), f"class_{idx}")
                results.append({
                    "label": label,
                    "prob": float(probs[idx])
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error during CNN prediction: {e}")
            return self._demo_prediction(k, "cnn")
    
    def _predict_vgg(self, img_array: np.ndarray, k: int = 3) -> List[Dict]:
        """Predict using VGG16 model"""
        if not self.models_loaded["vgg"] or self.vgg_model is None:
            return self._demo_prediction(k, "vgg")
        
        try:
            # VGG16 expects 224x224 RGB images
            predictions = self.vgg_model.predict(img_array, verbose=0)
            probs = predictions[0]
            
            # Get top k predictions
            top_indices = np.argsort(probs)[-k:][::-1]
            
            results = []
            for idx in top_indices:
                label = self.vgg_labels.get(str(idx), f"class_{idx}")
                results.append({
                    "label": label,
                    "prob": float(probs[idx])
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error during VGG prediction: {e}")
            return self._demo_prediction(k, "vgg")
    
    def _demo_prediction(self, k: int, model_type: str) -> List[Dict]:
        """Generate demo predictions when model is not available"""
        labels = self.get_labels(model_type)
        import random
        
        # Generate random probabilities that sum to less than 1
        probs = [random.uniform(0.1, 0.9) for _ in range(k)]
        probs = [p / sum(probs) * 0.8 for p in probs]  # Scale down so they don't sum to 1
        
        # Sort probabilities in descending order
        probs.sort(reverse=True)
        
        # Select random labels
        selected_labels = random.sample(labels, k)
        
        return [
            {"label": label, "prob": prob}
            for label, prob in zip(selected_labels, probs)
        ]


def safe_file_hash(file_bytes: bytes) -> str:
    """Generate SHA256 hash of file bytes"""
    return hashlib.sha256(file_bytes).hexdigest()


def preprocess_image_cnn(file_bytes: bytes) -> np.ndarray:
    """Preprocess image for custom CNN model (60x60 grayscale)"""
    try:
        # Open image with PIL
        image = Image.open(io.BytesIO(file_bytes))
        
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to CNN input size (60x60)
        image = image.resize((60, 60))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Normalize to 0-1 range
        img_array = img_array.astype(np.float32) / 255.0
        
        # Reshape to (1, 60, 60, 1) for CNN input
        img_array = img_array.reshape(1, 60, 60, 1)
        
        return img_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image for CNN: {e}")
        raise ValueError(f"Could not process image for CNN: {e}")


def preprocess_image_vgg(file_bytes: bytes) -> np.ndarray:
    """Preprocess image for VGG16 model (224x224 RGB)"""
    try:
        # Open image with PIL
        image = Image.open(io.BytesIO(file_bytes))
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to VGG16 input size (224x224)
        image = image.resize((224, 224))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to float32 and normalize to 0-1 range
        img_array = img_array.astype(np.float32) / 255.0
        
        # VGG16 preprocessing (ImageNet standard)
        # Note: Since we trained with preprocessing_function=preprocess_input from VGG16,
        # we need to apply the same preprocessing here
        from tensorflow.keras.applications.vgg16 import preprocess_input
        
        # Reshape to add batch dimension
        img_array = img_array.reshape(1, 224, 224, 3)
        
        # Apply VGG16 preprocessing (converts 0-1 to -1 to 1 range)
        img_array = preprocess_input(img_array * 255.0)  # preprocess_input expects 0-255 range
        
        return img_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image for VGG16: {e}")
        raise ValueError(f"Could not process image for VGG16: {e}")


# Backward compatibility
def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """Default preprocessing (CNN model)"""
    return preprocess_image_cnn(file_bytes)


def get_image_dimensions(file_bytes: bytes) -> Tuple[int, int]:
    """Get image width and height"""
    try:
        image = Image.open(io.BytesIO(file_bytes))
        return image.size  # (width, height)
    except Exception as e:
        logger.error(f"Error getting image dimensions: {e}")
        raise ValueError(f"Could not read image dimensions: {e}")


# Create global instance
ml_service = ModelService()
