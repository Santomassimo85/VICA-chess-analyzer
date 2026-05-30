"""
piece_classifier.py

Utility for loading a trained ResNet-18 model and using it to predict
which chess piece is present in a given image. The module provides a
PieceClassifier class that wraps model loading, preprocessing and
inference, and returns human-friendly results including probabilities.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

# Constants (must match the values used during training)
IMAGE_SIZE = 224
NUM_CLASSES = 6

# ImageNet normalization values used for model training
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


class PieceClassifier:
    """Classifier for single-piece images.

    The class encapsulates a PyTorch model and preprocessing steps so
    callers can load a checkpoint once and use the same object to make
    repeated predictions on either image files or PIL Image objects.
    """

    def __init__(self, model_path, device=None):
        """Initialize the classifier and load model weights.

        Parameters
        ----------
        model_path : str or pathlib.Path
            Path to a checkpoint file produced during training. The
            checkpoint is expected to contain the keys 'model_state_dict'
            and 'class_names'.
        device : str, optional
            Torch device to use for inference ('cuda' or 'cpu'). If None,
            the constructor will pick an available GPU when possible.
        """
        # Choose a device for running the model
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        # Note: printing helps when running scripts interactively
        print(f"Using device: {self.device}")

        # Load the training checkpoint from disk
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.class_names = checkpoint['class_names']
        self.test_accuracy = checkpoint.get('test_accuracy', 'N/A')

        # Recreate the network architecture used during training
        self.model = models.resnet18(weights=None)  # no pre-trained, we load our own
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, NUM_CLASSES)

        # Load the trained parameter values into the model
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()  # IMPORTANT: inference mode

        # Image preprocessing pipeline. Must match the transforms used
        # during training for correct results.
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])

        # Informative message about the loaded model
        print(f"Model loaded. Classes: {self.class_names}")
        print(f"Model test accuracy: {self.test_accuracy}")

    def _predict_tensor(self, image):
        """Run the model on a single PIL Image and return results.

        The returned dictionary contains:
        - 'class_name': the predicted class label (string)
        - 'confidence': probability for the top prediction (float)
        - 'all_probabilities': mapping from class name to probability
        """
        # Ensure the image has three color channels
        image = image.convert('RGB')

        # Apply the preprocessing transforms defined in __init__
        input_tensor = self.transform(image)

        # Add a batch dimension because the model expects batches
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        # Perform a forward pass; gradients are not required for inference
        with torch.no_grad():
            outputs = self.model(input_batch)
            probabilities = torch.softmax(outputs, dim=1)

        # Determine the predicted class and its confidence
        confidence, predicted_idx = probabilities.max(1)
        predicted_class = self.class_names[predicted_idx.item()]

        # Build a simple mapping of class -> probability for the caller
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
        """Predict from an image file given its path.

        The image is opened with PIL and forwarded to the model.
        """
        image = Image.open(image_path)
        return self._predict_tensor(image)

    def predict_pil(self, pil_image):
        """Predict from a PIL Image object already in memory.

        This avoids the overhead of saving temporary files when
        performing multiple predictions (for example, when scanning a
        chessboard and classifying each square).
        """
        return self._predict_tensor(pil_image)


# When executed as a script, run a small self-test using a sample image
if __name__ == "__main__":
    # Build paths relative to this script for portability
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent

    MODEL_PATH = PROJECT_ROOT / "models" / "chess_classifier.pth"

    # Test image: take the first image from test/Knight as a test
    TEST_IMAGE = PROJECT_ROOT / "data" / "chess_split" / "test" / "Knight"
    test_img_name = list(TEST_IMAGE.glob("*"))[0]  # first image in folder

    print(f"Loading model from: {MODEL_PATH}")
    print(f"Testing on: {test_img_name.name}\n")

    # Create classifier
    classifier = PieceClassifier(MODEL_PATH)

    # Predict
    result = classifier.predict(test_img_name)

    print("\nPREDICTION:")
    print(f"   Class: {result['class_name']}")
    print(f"   Confidence: {result['confidence']*100:.2f}%")
    print("\nAll probabilities:")
    for cls, prob in sorted(result['all_probabilities'].items(),
                            key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 50)
        print(f"   {cls:10s} {prob*100:5.2f}%  {bar}")
