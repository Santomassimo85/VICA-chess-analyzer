"""
piece_classifier.py
Loads the trained ResNet18 and predicts chess pieces from images.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

# Constants (must match training!)
IMAGE_SIZE = 224
NUM_CLASSES = 6

# ImageNet normalization (same as training)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


class PieceClassifier:
    """Classifies a chess piece image into one of 6 classes."""

    def __init__(self, model_path, device=None):
        """
        Load the trained model from a .pth file.

        Args:
            model_path: path to chess_classifier.pth
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        # Auto-detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        print(f"🔧 Using device: {self.device}")

        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.class_names = checkpoint['class_names']
        self.test_accuracy = checkpoint.get('test_accuracy', 'N/A')

        # Rebuild the model architecture (same as training!)
        self.model = models.resnet18(weights=None)  # no pre-trained, we load our own
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, NUM_CLASSES)

        # Load our trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()  # IMPORTANT: inference mode

        # Preprocessing pipeline (same as training!)
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])

        print(f"✅ Model loaded! Classes: {self.class_names}")
        print(f"   Model test accuracy: {self.test_accuracy}")

    def _predict_tensor(self, image):
        """
        Internal helper: runs the model on a PIL image.
        Used by both predict() and predict_pil().

        Args:
            image: a PIL.Image object

        Returns:
            dict with 'class_name', 'confidence', 'all_probabilities'
        """
        # Make sure it's RGB
        image = image.convert('RGB')

        # Apply preprocessing (same as training)
        input_tensor = self.transform(image)

        # Add batch dimension: (3, 224, 224) -> (1, 3, 224, 224)
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        # Forward pass (no gradients needed)
        with torch.no_grad():
            outputs = self.model(input_batch)
            probabilities = torch.softmax(outputs, dim=1)

        # Get the best class
        confidence, predicted_idx = probabilities.max(1)
        predicted_class = self.class_names[predicted_idx.item()]

        # Build readable output
        all_probs = {
            self.class_names[i]: float(probabilities[0][i])
            for i in range(len(self.class_names))
        }

        return {
            'class_name': predicted_class,
            'confidence': float(confidence.item()),
            'all_probabilities': all_probs
        }

    def predict(self, image_path):
        """
        Predict the chess piece from an image FILE.

        Args:
            image_path: path to the image file (str or Path)

        Returns:
            dict with 'class_name', 'confidence', 'all_probabilities'
        """
        image = Image.open(image_path)
        return self._predict_tensor(image)

    def predict_pil(self, pil_image):
        """
        Predict the chess piece from a PIL image IN MEMORY.
        Used by board_analyzer.py to avoid saving 64 files to disk.

        Args:
            pil_image: a PIL.Image object

        Returns:
            dict with 'class_name', 'confidence', 'all_probabilities'
        """
        return self._predict_tensor(pil_image)


# Test block — runs only when you execute THIS file directly
if __name__ == "__main__":
    # Robust path
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent

    MODEL_PATH = PROJECT_ROOT / "models" / "chess_classifier.pth"

    # Test image: take the first image from test/Knight as a test
    TEST_IMAGE = PROJECT_ROOT / "data" / "chess_split" / "test" / "Knight"
    test_img_name = list(TEST_IMAGE.glob("*"))[0]  # first image in folder

    print(f"📂 Loading model from: {MODEL_PATH}")
    print(f"🖼️  Testing on: {test_img_name.name}\n")

    # Create classifier
    classifier = PieceClassifier(MODEL_PATH)

    # Predict
    result = classifier.predict(test_img_name)

    print("\n🎯 PREDICTION:")
    print(f"   Class: {result['class_name']}")
    print(f"   Confidence: {result['confidence']*100:.2f}%")
    print(f"\n📊 All probabilities:")
    for cls, prob in sorted(result['all_probabilities'].items(),
                            key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 50)
        print(f"   {cls:10s} {prob*100:5.2f}%  {bar}")