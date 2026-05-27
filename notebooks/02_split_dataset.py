"""
02_split_dataset.py
Splits the chess dataset into train/val/test folders.
"""

import os
import shutil
import random
from pathlib import Path

# Robust paths (work regardless of where you run from)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Where the original dataset lives
SOURCE_DIR = PROJECT_ROOT / "data" / "synthetic_digital"
# Where the new split structure will be created
OUTPUT_DIR = PROJECT_ROOT / "data" / "synthetic_split"

# Split ratios
TRAIN_RATIO = 0.70  # 70% for training
VAL_RATIO = 0.15    # 15% for validation
TEST_RATIO = 0.15   # 15% for test (the rest)

# Seed for reproducibility (same shuffle every time)
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

print("=" * 50)
print("DATASET SPLITTING")
print("=" * 50)
print(f"Source: {SOURCE_DIR}")
print(f"Output: {OUTPUT_DIR}\n")

# Get list of classes (Bishop, King, Knight, etc.)
classes = sorted(os.listdir(SOURCE_DIR))
print(f"Classes: {classes}\n")

# Create the output folder structure
# data/chess_split/train/Bishop, data/chess_split/train/King, etc.
for split in ["train", "val", "test"]:
    for cls in classes:
        folder = OUTPUT_DIR / split / cls
        folder.mkdir(parents=True, exist_ok=True)

print("✅ Folder structure created.\n")


# For each class, shuffle and split images
print("Splitting images...")
print("-" * 50)

summary = {}  # to track how many files went where

for cls in classes:
    src_class_dir = SOURCE_DIR / cls
    
    # Get all image files in this class
    images = [f for f in os.listdir(src_class_dir) 
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Shuffle them randomly (reproducible thanks to seed)
    random.shuffle(images)
    
    # Calculate split sizes
    n_total = len(images)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    # Test gets whatever's left (avoids rounding issues)
    
    # Split the list
    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]
    
    # Copy files to their destinations
    for img in train_imgs:
        shutil.copy(src_class_dir / img, OUTPUT_DIR / "train" / cls / img)
    for img in val_imgs:
        shutil.copy(src_class_dir / img, OUTPUT_DIR / "val" / cls / img)
    for img in test_imgs:
        shutil.copy(src_class_dir / img, OUTPUT_DIR / "test" / cls / img)
    
    # Save summary
    summary[cls] = (len(train_imgs), len(val_imgs), len(test_imgs))
    print(f"  {cls:10s} → train: {len(train_imgs):3d}, "
          f"val: {len(val_imgs):3d}, test: {len(test_imgs):3d}")

print("-" * 50)



# Print final summary
total_train = sum(s[0] for s in summary.values())
total_val = sum(s[1] for s in summary.values())
total_test = sum(s[2] for s in summary.values())
total_all = total_train + total_val + total_test

print(f"\n📊 SUMMARY:")
print(f"  Train: {total_train:4d} images ({total_train/total_all*100:.1f}%)")
print(f"  Val:   {total_val:4d} images ({total_val/total_all*100:.1f}%)")
print(f"  Test:  {total_test:4d} images ({total_test/total_all*100:.1f}%)")
print(f"  TOTAL: {total_all:4d} images")
print(f"\n✅ Dataset split successfully!")
print(f"📁 Output location: {OUTPUT_DIR}")