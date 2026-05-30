"""
Split the chess dataset into training, validation, and test folders.

This script organizes the raw synthetic chess dataset by copying images
into separate directories for each split. The script maintains class
subdirectories within each split to keep the data organized by piece type.
"""

import os
import shutil
import random
from pathlib import Path

# Configure directory paths relative to this script location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Define where the raw data is located
SOURCE_DIR = PROJECT_ROOT / "data" / "synthetic_digital"
# Define where the split data will be saved
OUTPUT_DIR = PROJECT_ROOT / "data" / "synthetic_split"

# Configure how to split the dataset between training, validation, and testing
TRAIN_RATIO = 0.70  # 70% of images for training
VAL_RATIO = 0.15    # 15% of images for validation
TEST_RATIO = 0.15   # 15% of images for testing

# Fixed seed for repeatable splits
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

print("=" * 50)
print("DATASET SPLITTING")
print("=" * 50)
print(f"Source: {SOURCE_DIR}")
print(f"Output: {OUTPUT_DIR}\n")

# Class names
classes = sorted(os.listdir(SOURCE_DIR))
print(f"Classes: {classes}\n")

# Create the output folders
for split in ["train", "val", "test"]:
    for cls in classes:
        folder = OUTPUT_DIR / split / cls
        folder.mkdir(parents=True, exist_ok=True)

print("✅ Folder structure created.\n")


# Split each class folder
print("Splitting images...")
print("-" * 50)

summary = {}

for cls in classes:
    src_class_dir = SOURCE_DIR / cls
    
    # Collect image files
    images = [f for f in os.listdir(src_class_dir) 
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Shuffle the file order
    random.shuffle(images)
    
    # Work out split sizes
    n_total = len(images)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    
    # Split the list
    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]
    
    # Copy files to each split
    for img in train_imgs:
        shutil.copy(src_class_dir / img, OUTPUT_DIR / "train" / cls / img)
    for img in val_imgs:
        shutil.copy(src_class_dir / img, OUTPUT_DIR / "val" / cls / img)
    for img in test_imgs:
        shutil.copy(src_class_dir / img, OUTPUT_DIR / "test" / cls / img)
    
    # Save counts for the report
    summary[cls] = (len(train_imgs), len(val_imgs), len(test_imgs))
    print(f"  {cls:10s} → train: {len(train_imgs):3d}, "
          f"val: {len(val_imgs):3d}, test: {len(test_imgs):3d}")

print("-" * 50)
# Final summary
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
