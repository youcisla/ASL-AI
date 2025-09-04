#!/usr/bin/env python3
"""
Test script to validate the ASL Classifier API endpoints.
"""

import asyncio
import json
import requests
import time
from pathlib import Path
from PIL import Image
import io
import numpy as np

API_BASE = "http://localhost:8000"

def create_test_image():
    """Create a simple test image"""
    # Create a 224x224 RGB image with random colors
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return img_bytes.getvalue()

def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{API_BASE}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Health check passed")
            print(f"  Status: {data['status']}")
            print(f"  Model loaded: {data['model_loaded']}")
            print(f"  Number of labels: {data['num_labels']}")
            print(f"  Database: {data['db']}")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False

def test_labels():
    """Test labels endpoint"""
    print("\nTesting /api/labels endpoint...")
    try:
        response = requests.get(f"{API_BASE}/api/labels")
        if response.status_code == 200:
            labels = response.json()
            print(f"✓ Labels retrieved successfully")
            print(f"  Number of labels: {len(labels)}")
            print(f"  First 5 labels: {labels[:5]}")
            return labels
        else:
            print(f"✗ Labels request failed: {response.status_code}")
            if response.status_code == 503:
                print("  Model not loaded - check model files")
            return None
    except Exception as e:
        print(f"✗ Labels request error: {e}")
        return None

def test_prediction():
    """Test prediction endpoint"""
    print("\nTesting /api/predict endpoint...")
    try:
        # Create test image
        img_bytes = create_test_image()
        
        # Send prediction request
        files = {"image": ("test.png", img_bytes, "image/png")}
        start_time = time.time()
        response = requests.post(f"{API_BASE}/api/predict", files=files)
        request_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Prediction successful")
            print(f"  Request time: {request_time:.0f}ms")
            print(f"  Upload ID: {data['upload_id']}")
            print(f"  Model latency: {data['latency_ms']:.0f}ms")
            print(f"  Top prediction: {data['top3'][0]['label']} ({data['top3'][0]['prob']:.3f})")
            print(f"  Top 3 predictions:")
            for i, pred in enumerate(data['top3']):
                print(f"    {i+1}. {pred['label']}: {pred['prob']:.3f}")
            return data
        else:
            print(f"✗ Prediction failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"  Error: {error_data.get('detail', 'Unknown error')}")
            except:
                print(f"  Response: {response.text}")
            return None
    except Exception as e:
        print(f"✗ Prediction error: {e}")
        return None

def test_feedback(upload_id):
    """Test feedback endpoint"""
    print("\nTesting /api/feedback endpoint...")
    try:
        feedback_data = {
            "upload_id": upload_id,
            "is_correct": True,
            "notes": "Test feedback from automated script"
        }
        
        response = requests.post(f"{API_BASE}/api/feedback", json=feedback_data)
        
        if response.status_code == 200:
            print(f"✓ Feedback submitted successfully")
            return True
        else:
            print(f"✗ Feedback failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"  Error: {error_data.get('detail', 'Unknown error')}")
            except:
                print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Feedback error: {e}")
        return False

def test_history():
    """Test history endpoint"""
    print("\nTesting /api/history endpoint...")
    try:
        response = requests.get(f"{API_BASE}/api/history?limit=5")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ History retrieved successfully")
            print(f"  Total predictions: {data['total']}")
            print(f"  Items in response: {len(data['items'])}")
            if data['items']:
                latest = data['items'][0]
                print(f"  Latest prediction: {latest['top1']['label']} ({latest['top1']['prob']:.3f})")
            return True
        else:
            print(f"✗ History failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ History error: {e}")
        return False

def test_invalid_image():
    """Test with invalid image"""
    print("\nTesting with invalid file...")
    try:
        # Send text file as image
        files = {"image": ("test.txt", b"not an image", "text/plain")}
        response = requests.post(f"{API_BASE}/api/predict", files=files)
        
        if response.status_code == 400:
            print(f"✓ Invalid file correctly rejected")
            return True
        else:
            print(f"✗ Invalid file not rejected: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Invalid file test error: {e}")
        return False

def main():
    """Run all tests"""
    print("ASL Classifier API Test Suite")
    print("=" * 40)
    
    tests_passed = 0
    total_tests = 6
    
    # Test 1: Health check
    if test_health():
        tests_passed += 1
    
    # Test 2: Labels
    labels = test_labels()
    if labels:
        tests_passed += 1
    
    # Test 3: Prediction
    prediction_result = test_prediction()
    if prediction_result:
        tests_passed += 1
        
        # Test 4: Feedback (depends on prediction)
        if test_feedback(prediction_result['upload_id']):
            tests_passed += 1
    else:
        print("\nSkipping feedback test (prediction failed)")
    
    # Test 5: History
    if test_history():
        tests_passed += 1
    
    # Test 6: Invalid image
    if test_invalid_image():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 40)
    print(f"Test Results: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! API is working correctly.")
        return 0
    else:
        print(f"❌ {total_tests - tests_passed} test(s) failed.")
        return 1

if __name__ == "__main__":
    exit(main())
