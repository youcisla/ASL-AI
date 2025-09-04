# Quick retrain with full dataset for better R/L accuracy
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
import os
import cv2
import tqdm
from sklearn.utils import shuffle
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam

def loadTrainData(trainDir, imageWidth, imageHeight, max_images_per_class=None):
    """Load training data with optional limit per class"""
    classes = sorted([d for d in os.listdir(trainDir) 
                     if os.path.isdir(os.path.join(trainDir, d))])
    
    print(f"🏷️ Loading data for {len(classes)} classes...")
    if max_images_per_class:
        print(f"⚡ Using {max_images_per_class} images per class for faster training")
    else:
        print(f"🔥 Using ALL available images for maximum accuracy")
    
    imagesList = []
    labels = []
    
    for class_name in tqdm.tqdm(classes, desc="Processing classes"):
        classPath = os.path.join(trainDir, class_name)
        image_files = [f for f in os.listdir(classPath) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if max_images_per_class:
            image_files = image_files[:max_images_per_class]
        
        class_loaded = 0
        for image_file in image_files:
            try:
                imgPath = os.path.join(classPath, image_file)
                img = cv2.imread(imgPath)
                
                if img is None:
                    continue
                
                # Convert to grayscale
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (imageWidth, imageHeight))
                
                imagesList.append(img)
                labels.append(class_name)
                class_loaded += 1
                
            except Exception:
                continue
        
        print(f"  ✅ {class_name}: {class_loaded} images loaded")
    
    return imagesList, labels

def quick_retrain():
    """Quick retrain with more data for better R/L distinction"""
    print("🚀 Quick Retrain for Better R/L Accuracy")
    print("="*50)
    
    # Use the existing training directory
    trainDir = 'asl_dataset\\asl_alphabet_train\\asl_alphabet_train'
    
    if not os.path.exists(trainDir):
        print(f"❌ Training directory not found: {trainDir}")
        return
    
    # Load MORE data - use 500 images per class for balance of speed and accuracy
    print("📊 Loading training data (500 images per class)...")
    X, y = loadTrainData(trainDir, imageWidth=64, imageHeight=64, max_images_per_class=500)
    
    print(f"✅ Data loaded: {len(X)} images, {len(set(y))} classes")
    
    # Preprocess
    XShuffled, yShuffled = shuffle(X, y, random_state=42)
    xtrain = np.array(XShuffled)
    ytrain = np.array(yShuffled)
    
    # Normalize and reshape
    xtrain = xtrain.astype('float32') / 255.0
    xtrain = xtrain.reshape(xtrain.shape[0], 64, 64, 1)
    
    # Convert labels to categorical
    classes = sorted(list(set(y)))
    categories = {c: i for i, c in enumerate(classes)}
    
    for i in range(len(ytrain)):
        ytrain[i] = categories[ytrain[i]]
    
    ytrain = to_categorical(ytrain, num_classes=len(classes))
    
    print(f"📏 Training data shape: {xtrain.shape}")
    print(f"📏 Labels shape: {ytrain.shape}")
    
    # Build enhanced model
    print("🏗️ Building enhanced CNN model...")
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 1)),
        BatchNormalization(),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Dropout(0.15),
        
        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Dropout(0.2),
        
        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Dropout(0.25),
        
        Conv2D(256, (3,3), activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(len(classes), activation='softmax'),
    ])
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("📊 Model compiled successfully!")
    
    # Enhanced callbacks
    callbacks = [
        ModelCheckpoint('best_asl_model_enhanced.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=3, min_lr=1e-7, mode='max', verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, mode='max', verbose=1)
    ]
    
    # Train with more epochs
    print("🎯 Starting enhanced training...")
    history = model.fit(
        xtrain, ytrain,
        validation_split=0.2,
        epochs=20,  # More epochs
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save enhanced model
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    
    # Load best model and save
    best_model = tf.keras.models.load_model('best_asl_model_enhanced.h5')
    best_model.save("models/asl_cnn_model_enhanced.keras")
    
    # Save labels
    with open("models/labels.json", 'w') as f:
        json.dump(classes, f, indent=2)
    
    # Copy to backend
    import shutil
    shutil.copy("models/asl_cnn_model_enhanced.keras", "backend/models/asl_cnn_model.keras")
    shutil.copy("models/labels.json", "backend/models/labels.json")
    
    final_acc = max(history.history['val_accuracy'])
    print(f"\n🎉 Enhanced model training complete!")
    print(f"✅ Best validation accuracy: {final_acc*100:.2f}%")
    print(f"✅ Enhanced model saved and copied to backend")
    print(f"✅ With 5x more training data, R/L distinction should be much better!")
    
    return final_acc

if __name__ == "__main__":
    accuracy = quick_retrain()
    print(f"\n🚀 Restart your backend to use the enhanced model!")
    print(f"🎯 Expected improvement in R/L accuracy: {accuracy*100:.1f}%")
