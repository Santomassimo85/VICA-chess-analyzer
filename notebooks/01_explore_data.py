"""
01_explore_data.py
Script to explore the chess dataset
"""

import os
import cv2
import matplotlib.pyplot as plt


DATASET_PATH = "data/Chessman-image-dataset/Chess"

print("=" * 50)
print("CHESS DATASET EXPLORATION")
print("=" * 50)

classes = os.listdir(DATASET_PATH)
print(f"\nClasses found: {classes}")
print(f"Number of classes: {len(classes)}")

print("\nImage count per class:")
print("-" * 30)

total = 0
for cls in classes:
    cls_path = os.path.join(DATASET_PATH, cls)
    n_images = len(os.listdir(cls_path))
    print(f"  {cls:10s} → {n_images:4d} images")
    total += n_images

print("-" * 30)
print(f"  TOTAL      → {total:4d} images\n")

print("Showing 1 example image per class...")
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for idx, cls in enumerate(classes):
    cls_path = os.path.join(DATASET_PATH, cls)
    first_img_name = os.listdir(cls_path)[0]
    img_path = os.path.join(cls_path, first_img_name)
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    axes[idx].imshow(img)
    axes[idx].set_title(cls, fontsize=14)
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig("notebooks/sample_pieces.png", dpi=100)
plt.show()
print("\n✅ Done! Image saved in notebooks/sample_pieces.png")











