# Models Directory

This directory contains the trained machine learning models and associated metadata for the ASL Classifier.

## Required Files

### 1. vgg16_asl_final.keras
The trained VGG16 transfer learning model saved in Keras format.

**Requirements:**
- Input shape: (224, 224, 3)
- Preprocessing: `tf.keras.applications.vgg16.preprocess_input`
- Output: Softmax probabilities for ASL classes

### 2. labels.json
JSON array containing the class labels in the same order as the model's output.

**Format:**
```json
["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
```

## How to Generate These Files

1. **Train your model** using the provided Jupyter notebook (`projet-ia.ipynb`)

2. **Run the final cell** in the notebook to save both files:
   ```python
   # The notebook contains code to save:
   model.save("./models/vgg16_asl_final.keras")
   
   # And export labels:
   with open("./models/labels.json", "w") as f:
       json.dump(class_names, f, indent=2)
   ```

3. **Verify the files** are created in this directory

## File Sizes

- `vgg16_asl_final.keras`: ~60-80 MB (depends on exact architecture)
- `labels.json`: <1 KB

## Model Performance

The model uses VGG16 transfer learning with:
- Frozen base layers (except block5 for fine-tuning)
- Global average pooling
- Dense layer with L2 regularization
- Dropout for regularization
- Label smoothing during training

Expected performance:
- Training accuracy: >95%
- Validation accuracy: >90%
- Inference time: ~50-200ms per image

## Troubleshooting

### Model file not found
```
FileNotFoundError: Model file not found: ./models/vgg16_asl_final.keras
```
**Solution:** Run the training notebook and execute the final cell to save the model.

### Labels file not found
```
FileNotFoundError: Labels file not found: ./models/labels.json
```
**Solution:** Run the training notebook and execute the final cell to save the labels.

### Model loading errors
If you get TensorFlow/Keras loading errors:
1. Ensure you're using compatible TensorFlow versions
2. Try loading with `tf.keras.models.load_model()`
3. Check the model was saved properly

### Wrong number of classes
If the API reports wrong number of classes:
1. Check that `labels.json` contains the correct number of classes
2. Ensure the labels match your training data
3. Verify the model output shape matches the number of labels
