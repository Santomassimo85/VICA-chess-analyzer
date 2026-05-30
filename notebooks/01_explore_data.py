"""
Explore Chess Dataset

This notebook examines the structure of our chess piece dataset. We load all
chess piece classes, count how many images we have for each piece type, and
visualize one example from each class to get a feel for the data quality and
variety before moving into model training.
"""

import os
import cv2
import matplotlib.pyplot as plt


# Path to the chess dataset
DATASET_PATH = "data/Chessman-image-dataset/Chess"

print("=" * 50)
print("CHESS DATASET EXPLORATION")
print("=" * 50)

# Load all piece classes from the dataset directory
classes = os.listdir(DATASET_PATH)
print(f"\nClasses found: {classes}")
print(f"Number of classes: {len(classes)}")

# Count and display the number of images in each class
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

# Create a visualization grid with one sample image from each class
print("Showing 1 example image per class...")
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for idx, cls in enumerate(classes):
    cls_path = os.path.join(DATASET_PATH, cls)
    first_img_name = os.listdir(cls_path)[0]
    img_path = os.path.join(cls_path, first_img_name)
    
    # Read and convert image from BGR to RGB color space
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Display the image with its class label
    axes[idx].imshow(img)
    axes[idx].set_title(cls, fontsize=14)
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig("notebooks/sample_pieces.png", dpi=100)
plt.show()
print("\nDone! Sample image grid saved to notebooks/sample_pieces.png")











