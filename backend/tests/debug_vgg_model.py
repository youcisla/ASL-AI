#!/usr/bin/env python3
"""
Debug script to analyze the VGG16 model loading issue.
"""

import sys
import os
from pathlib import Path

# Add parent directory (backend) to path so we can import app
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Change working directory to backend so model paths work correctly
os.chdir(backend_dir)

import tensorflow as tf
from tensorflow import keras

def analyze_vgg_model():
    """Analyze the VGG16 model to understand the architecture issue"""
    print("🔍 Analyzing VGG16 model...")
    
    model_path = "./models/vgg16_transfer_asl_final.keras"
    
    try:
        # Try to load the model
        print(f"Loading model from: {model_path}")
        model = keras.models.load_model(model_path)
        
        print("✅ Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        print(f"Number of layers: {len(model.layers)}")
        
        # Print model summary
        print("\n📋 Model Summary:")
        model.summary()
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        
        # Try to inspect the model file
        try:
            print("\n🔧 Attempting to inspect model architecture...")
            
            # Load without compilation to avoid issues
            model = keras.models.load_model(model_path, compile=False)
            print("✅ Model loaded without compilation!")
            
            print(f"Model input shape: {model.input_shape}")
            print(f"Model output shape: {model.output_shape}")
            
            # Print layer information
            print("\n📋 Layer Information:")
            for i, layer in enumerate(model.layers):
                print(f"Layer {i}: {layer.name} - {type(layer).__name__}")
                if hasattr(layer, 'input_shape'):
                    print(f"  Input shape: {layer.input_shape}")
                if hasattr(layer, 'output_shape'):
                    print(f"  Output shape: {layer.output_shape}")
            
        except Exception as e2:
            print(f"❌ Error inspecting model: {e2}")
            return False
    
    return False

def check_tensorflow_version():
    """Check TensorFlow version compatibility"""
    print(f"🔧 TensorFlow version: {tf.__version__}")
    print(f"🔧 Keras version: {keras.__version__}")

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 VGG16 MODEL DEBUG")
    print("=" * 60)
    
    check_tensorflow_version()
    print()
    analyze_vgg_model()
