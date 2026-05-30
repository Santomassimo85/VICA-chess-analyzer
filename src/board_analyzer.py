"""
board_analyzer.py

This module converts a chess board image into an 8x8 board representation.

The analysis process follows these steps:
1. Remove borders: Crops uniform borders around the board so it fills the frame
2. Split into squares: Divides the board image into 64 individual square images
3. Detect pieces: Uses computer vision to identify occupied squares
4. Classify pieces: Runs a trained neural network to identify piece types

The output consists of two 8x8 matrices:
- Board matrix: Contains piece names (or "empty" for vacant squares)
- Confidence matrix: Contains the classifier's confidence scores for each square
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image

from piece_classifier import PieceClassifier

def auto_crop_borders(img, tolerance=15):
    """
    Remove a uniform border around the image if present.

    Many screenshots or photos include a solid-colour frame around the
    chess board (for example a black or white margin). This routine
    inspects the outer rows and columns for very low variance. When a
    contiguous low-variance region is found on an edge it is removed.

    The function limits cropping to at most 25% from each side and also
    rejects the crop if it would discard more than half of the image in
    either dimension (safety check).

    Args:
        img: uint8 BGR image (numpy array)
        tolerance: maximum grayscale standard deviation to treat a row/
                   column as uniform (lower -> stricter)

    Returns:
        Cropped image if a border was removed, otherwise the original
        image. The function never modifies the input shape drastically
        thanks to conservative limits.
    """
    import numpy as np

    h, w = img.shape[:2]

    # Work on a grayscale copy to measure variance per line
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Track how many rows/cols from each side look uniform. We limit the
    # scan to 25% of the image to avoid removing board content.
    top = 0
    for y in range(h // 4):
        if gray[y, :].std() > tolerance:
            top = y
            break

    bottom = h
    for y in range(h - 1, h - h // 4, -1):
        if gray[y, :].std() > tolerance:
            bottom = y + 1
            break

    left = 0
    for x in range(w // 4):
        if gray[:, x].std() > tolerance:
            left = x
            break

    right = w
    for x in range(w - 1, w - w // 4, -1):
        if gray[:, x].std() > tolerance:
            right = x + 1
            break

    cropped = img[top:bottom, left:right]

    # If the crop would remove too much, don't apply it.
    if cropped.shape[0] < h * 0.5 or cropped.shape[1] < w * 0.5:
        return img

    if top > 0 or left > 0 or bottom < h or right < w:
        print(f"   🪚 Auto-cropped borders: top={top}, bottom={h-bottom}, "
              f"left={left}, right={w-right} px")

    return cropped


class BoardAnalyzer:
    """Analyzes a full chess board image into a position matrix."""
    
    
    

    def __init__(self, model_path, confidence_threshold=0.55,
                 edge_threshold=0.020, color_std_threshold=18.0):
        # Load our trained classifier
        self.classifier = PieceClassifier(model_path)
        # Below this confidence -> treat square as empty
        self.confidence_threshold = confidence_threshold
        # Occupancy detection thresholds
        self.edge_threshold = edge_threshold
        self.color_std_threshold = color_std_threshold
        print(f"🎯 Confidence threshold: {confidence_threshold}")
        print(f"🔍 Edge threshold: {edge_threshold}")
        print(f"🎨 Color std threshold: {color_std_threshold}")

    def split_into_squares(self, board_image):
        """
        Split an aligned board image into 64 squares.
        Returns list of 64 dicts: {row, col, image}.
        """
        h, w = board_image.shape[:2]
        square_h = h // 8
        square_w = w // 8

        squares = []
        for row in range(8):
            for col in range(8):
                y1, y2 = row * square_h, (row + 1) * square_h
                x1, x2 = col * square_w, (col + 1) * square_w
                square_img = board_image[y1:y2, x1:x2]
                squares.append({
                    'row': row,
                    'col': col,
                    'image': square_img
                })
        return squares

    def is_square_occupied(self, square_image):
        """
        Decide if a square contains a piece, using classical CV.
        An empty square is smooth and uniform; an occupied one
        has edges (the piece outline) and color variation.

        Returns True if occupied, False if empty.
        """
        # Crop the center 70% of the square to ignore border artifacts
        h, w = square_image.shape[:2]
        margin_h, margin_w = int(h * 0.15), int(w * 0.15)
        center = square_image[margin_h:h - margin_h, margin_w:w - margin_w]

        # --- Check 1: Edge density ---
        gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / edges.size

        # --- Check 2: Color variation (std deviation) ---
        color_std = float(np.std(gray))

        # Occupied if EITHER signal is high enough
        occupied = (edge_density > self.edge_threshold or
                    color_std > self.color_std_threshold)
        return occupied

    def classify_square(self, square_image):
        """
        Classify one square.
        First checks occupancy; only runs the CNN if occupied.
        Returns (piece_name, confidence).
        """
        # Step 1: Is the square even occupied?
        if not self.is_square_occupied(square_image):
            return ('empty', 1.0)

        # Step 2: Occupied -> classify with the CNN
        rgb = cv2.cvtColor(square_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        result = self.classifier.predict_pil(pil_image)

        piece = result['class_name']
        conf = result['confidence']

        # Extra safety: if CNN is very unsure, call it empty
        if conf < self.confidence_threshold:
            return ('empty', conf)
        return (piece, conf)

    def analyze_board(self, board_image):
        """
        Analyze a full board image.
        Returns an 8x8 matrix of piece names (or 'empty')
        and an 8x8 matrix of confidences.
        """
        
        board_image = auto_crop_borders(board_image)
        
        squares = self.split_into_squares(board_image)

        board_matrix = [['empty' for _ in range(8)] for _ in range(8)]
        confidence_matrix = [[0.0 for _ in range(8)] for _ in range(8)]

        print("\n🔍 Analyzing 64 squares...")
        piece_count = 0

        for sq in squares:
            row, col = sq['row'], sq['col']
            piece, conf = self.classify_square(sq['image'])
            board_matrix[row][col] = piece
            confidence_matrix[row][col] = conf
            if piece != 'empty':
                piece_count += 1

        print(f"✅ Found {piece_count} pieces on the board\n")
        return board_matrix, confidence_matrix

    def print_board(self, board_matrix):
        """Print the board matrix in a readable way."""
        symbols = {
            'empty': '.', 'Pawn': 'P', 'Knight': 'N', 'Bishop': 'B',
            'Rook': 'R', 'Queen': 'Q', 'King': 'K'
        }
        print("  +" + "---+" * 8)
        for row in range(8):
            line = f"{8-row} |"
            for col in range(8):
                piece = board_matrix[row][col]
                line += f" {symbols.get(piece, '?')} |"
            print(line)
            print("  +" + "---+" * 8)
        print("    a   b   c   d   e   f   g   h")


# Test block
if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent

    MODEL_PATH = PROJECT_ROOT / "models" / "chess_classifier_digital.pth"
    TEST_IMAGE = PROJECT_ROOT / "data" / "test_photos" / "screenshot1.png"

    print(f"📂 Model: {MODEL_PATH}")
    print(f"🖼️  Board: {TEST_IMAGE}\n")

    board_img = cv2.imread(str(TEST_IMAGE))
    if board_img is None:
        print("❌ Could not load board image! Check the path.")
        exit()

    analyzer = BoardAnalyzer(MODEL_PATH)
    board_matrix, conf_matrix = analyzer.analyze_board(board_img)

    print("📋 DETECTED BOARD:")
    analyzer.print_board(board_matrix)
