#!/usr/bin/env python3
"""
Test script to verify that both models can be loaded and make predictions.
Run this script to test your model setup before starting the main application.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from app.ml_service import ml_service, preprocess_image_cnn, preprocess_image_vgg
from app.config import settings
import numpy as np
from PIL import Image
import io


def create_test_image(size=(224, 224), mode='RGB'):
    """Create a test image for validation"""
    # Create a simple test image with some patterns
    img = Image.new(mode, size, color='white')
    # Add some simple patterns to make it more realistic
    pixels = img.load()
    for i in range(size[0]):
        for j in range(size[1]):
            # Create a simple gradient pattern
            value = int((i + j) % 256)
            if mode == 'RGB':
                pixels[i, j] = (value, value // 2, value // 3)
            else:  # Grayscale
                pixels[i, j] = value
    return img


async def test_model_loading():
    """Test loading both models"""
    print("=" * 60)
    print("TESTING MODEL LOADING")
    print("=" * 60)
    
    # Load models
    print("Loading models...")
    await ml_service.load_models()
    
    # Check if models are loaded
    cnn_loaded = ml_service.is_ready("cnn")
    vgg_loaded = ml_service.is_ready("vgg")
    
    print(f"CNN Model loaded: {cnn_loaded}")
    print(f"VGG16 Model loaded: {vgg_loaded}")
    
    if cnn_loaded:
        cnn_labels = ml_service.get_labels("cnn")
        print(f"CNN Labels count: {len(cnn_labels)}")
        print(f"CNN Sample labels: {list(cnn_labels)[:5]}...")
    
    if vgg_loaded:
        vgg_labels = ml_service.get_labels("vgg")
        print(f"VGG16 Labels count: {len(vgg_labels)}")
        print(f"VGG16 Sample labels: {list(vgg_labels)[:5]}...")
    
    return cnn_loaded, vgg_loaded


def test_preprocessing():
    """Test image preprocessing functions"""
    print("\n" + "=" * 60)
    print("TESTING IMAGE PREPROCESSING")
    print("=" * 60)
    
    try:
        # Test CNN preprocessing
        print("Testing CNN preprocessing...")
        cnn_test_img = create_test_image(size=(100, 100), mode='RGB')
        cnn_img_bytes = io.BytesIO()
        cnn_test_img.save(cnn_img_bytes, format='PNG')
        cnn_img_bytes.seek(0)
        
        cnn_processed = preprocess_image_cnn(cnn_img_bytes.getvalue())
        print(f"CNN processed shape: {cnn_processed.shape}")
        print(f"CNN processed range: [{cnn_processed.min():.3f}, {cnn_processed.max():.3f}]")
        
        # Test VGG preprocessing
        print("Testing VGG16 preprocessing...")
        vgg_test_img = create_test_image(size=(300, 300), mode='RGB')
        vgg_img_bytes = io.BytesIO()
        vgg_test_img.save(vgg_img_bytes, format='PNG')
        vgg_img_bytes.seek(0)
        
        vgg_processed = preprocess_image_vgg(vgg_img_bytes.getvalue())
        print(f"VGG16 processed shape: {vgg_processed.shape}")
        print(f"VGG16 processed range: [{vgg_processed.min():.3f}, {vgg_processed.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"Preprocessing test failed: {e}")
        return False


async def test_predictions():
    """Test making predictions with both models"""
    print("\n" + "=" * 60)
    print("TESTING MODEL PREDICTIONS")
    print("=" * 60)
    
    # Create test images
    cnn_test_img = create_test_image(size=(100, 100), mode='RGB')
    cnn_img_bytes = io.BytesIO()
    cnn_test_img.save(cnn_img_bytes, format='PNG')
    cnn_img_bytes.seek(0)
    
    vgg_test_img = create_test_image(size=(300, 300), mode='RGB')
    vgg_img_bytes = io.BytesIO()
    vgg_test_img.save(vgg_img_bytes, format='PNG')
    vgg_img_bytes.seek(0)
    
    try:
        # Test CNN prediction
        if ml_service.is_ready("cnn"):
            print("Testing CNN prediction...")
            cnn_processed = preprocess_image_cnn(cnn_img_bytes.getvalue())
            cnn_results = ml_service.predict_topk(cnn_processed, k=3, model_type="cnn")
            print(f"CNN Top 3 predictions:")
            for i, result in enumerate(cnn_results[:3]):
                print(f"  {i+1}. {result['label']}: {result['prob']:.3f}")
        else:
            print("CNN model not available for prediction testing")
        
        # Test VGG16 prediction
        if ml_service.is_ready("vgg"):
            print("\nTesting VGG16 prediction...")
            vgg_processed = preprocess_image_vgg(vgg_img_bytes.getvalue())
            vgg_results = ml_service.predict_topk(vgg_processed, k=3, model_type="vgg")
            print(f"VGG16 Top 3 predictions:")
            for i, result in enumerate(vgg_results[:3]):
                print(f"  {i+1}. {result['label']}: {result['prob']:.3f}")
        else:
            print("VGG16 model not available for prediction testing")
            
        return True
        
    except Exception as e:
        print(f"Prediction test failed: {e}")
        return False


def test_model_files():
    """Test if model files exist and are accessible"""
    print("\n" + "=" * 60)
    print("TESTING MODEL FILES")
    print("=" * 60)
    
    files_to_check = [
        ("CNN Model", settings.cnn_model_path),
        ("CNN Labels", settings.cnn_labels_path),
        ("VGG16 Model", settings.vgg_model_path),
        ("VGG16 Labels", settings.vgg_labels_path),
    ]
    
    all_exist = True
    for name, path in files_to_check:
        file_path = Path(path)
        exists = file_path.exists()
        size = file_path.stat().st_size if exists else 0
        print(f"{name:15} | {'✓' if exists else '✗'} | {path} | {size/1024/1024:.1f} MB")
        if not exists:
            all_exist = False
    
    return all_exist


async def main():
    """Run all tests"""
    print("ASL MODEL TESTING SUITE")
    print("Testing your trained models and backend setup...\n")
    
    # Test 1: Check if model files exist
    files_ok = test_model_files()
    
    # Test 2: Test preprocessing functions
    preprocessing_ok = test_preprocessing()
    
    # Test 3: Test model loading
    cnn_loaded, vgg_loaded = await test_model_loading()
    
    # Test 4: Test predictions
    predictions_ok = await test_predictions()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Model files exist:     {'✓' if files_ok else '✗'}")
    print(f"Preprocessing works:   {'✓' if preprocessing_ok else '✗'}")
    print(f"CNN model loads:       {'✓' if cnn_loaded else '✗'}")
    print(f"VGG16 model loads:     {'✓' if vgg_loaded else '✗'}")
    print(f"Predictions work:      {'✓' if predictions_ok else '✗'}")
    
    if all([files_ok, preprocessing_ok, cnn_loaded or vgg_loaded, predictions_ok]):
        print("\n🎉 All tests passed! Your backend is ready to run.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the issues above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
