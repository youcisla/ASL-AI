# Quick script to retrain with FULL dataset for better R/L distinction
import os
import sys
sys.path.append('.')

# Let's check if we can quickly retrain with more data
print("🔄 Analyzing training options for better R/L distinction...")

# Check how much training data we have
train_dir = "asl_dataset\\asl_alphabet_train\\asl_alphabet_train"

if os.path.exists(train_dir):
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    print(f"📊 Available classes: {len(classes)}")
    
    # Check R and L specifically
    r_path = os.path.join(train_dir, "R")
    l_path = os.path.join(train_dir, "L")
    
    if os.path.exists(r_path):
        r_images = [f for f in os.listdir(r_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"📈 R images available: {len(r_images)}")
    
    if os.path.exists(l_path):
        l_images = [f for f in os.listdir(l_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"📈 L images available: {len(l_images)}")
    
    print(f"\n💡 Current model was trained with only 100 images per class")
    print(f"💡 For better R/L distinction, we should retrain with ALL available data")
    print(f"💡 This would give us ~{len(r_images) if 'r_images' in locals() else 'unknown'} R images and ~{len(l_images) if 'l_images' in locals() else 'unknown'} L images")
else:
    print("❌ Training directory not found")

print(f"\n🔧 Options to improve R/L distinction:")
print(f"1. ✅ Retrain with full dataset (recommended)")
print(f"2. ✅ Add data augmentation for better generalization") 
print(f"3. ✅ Increase training epochs for better learning")
print(f"4. ✅ Add more validation data")

print(f"\n⚡ Quick solution: Modify the notebook to use max_images_per_class=None")
print(f"   This will use ALL available training data instead of just 100 per class")
