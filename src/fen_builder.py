"""
fen_builder.py

Generate a FEN (Forsyth–Edwards Notation) string from a detected
chess board. The board detector or classifier provides the piece
type for each square; this module determines piece color from the
square image and assembles the final FEN placement string.

The color detection is a simple, classical image processing approach:
it isolates the piece region in the square and decides whether the
piece body is predominantly light (white pieces) or dark (black
pieces). This file does not perform piece recognition; it only
combines type and color into a standard FEN.
"""

import cv2
import numpy as np


# Map piece type to its FEN letter. The mapping here uses the
# canonical uppercase letters; the case is changed later based on
# detected piece color (uppercase for White, lowercase for Black).
PIECE_TO_FEN = {
    'Pawn': 'P', 'Knight': 'N', 'Bishop': 'B',
    'Rook': 'R', 'Queen': 'Q', 'King': 'K'
}


def detect_piece_color(square_image):
    """
    Determine whether a piece in a square image is white or black.

    This function crops the central area of the square (where the
    piece body is expected), converts it to grayscale, and finds
    edges to localize the piece. It then inspects pixel intensities
    within the localized region to decide if the piece body is
    predominantly light or dark.

    Returns:
        'white' or 'black'
    """
    # Crop the central region of the square where a piece is most
    # likely to appear. This reduces influence from square borders
    # and background.
    h, w = square_image.shape[:2]
    m_h, m_w = int(h * 0.20), int(w * 0.20)
    center = square_image[m_h:h - m_h, m_w:w - m_w]

    gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)

    # Step 1: find the piece pixels by detecting edges. The piece
    # typically contrasts with the board background, so edge
    # detection helps locate its outline.
    edges = cv2.Canny(gray, 40, 120)
    # Dilate the edge map to form a filled-like mask for the piece
    # region (so we can sample interior pixels reliably).
    kernel = np.ones((7, 7), np.uint8)
    piece_region = cv2.dilate(edges, kernel, iterations=2)

    # Step 2: extract grayscale pixels that fall inside the piece mask
    piece_pixels = gray[piece_region > 0]

    # Safety fallback: if the edge-based mask yields too few pixels
    # (for example the piece is small or edges failed), use a simple
    # brightness comparison on the cropped center instead.
    if piece_pixels.size < 20:
        # Fallback: compare extremes in the whole center
        bright = np.count_nonzero(gray > 170)
        dark = np.count_nonzero(gray < 90)
        return 'white' if bright >= dark else 'black'

    # Step 3: within the piece region, compare counts of bright and
    # dark pixels. Many white pieces have a lighter interior while
    # black pieces have darker interiors. The outline can be dark
    # for both colors, so this focuses on interior pixels.
    bright_in_piece = np.count_nonzero(piece_pixels > 150)
    dark_in_piece = np.count_nonzero(piece_pixels < 100)

    if bright_in_piece >= dark_in_piece:
        return 'white'
    return 'black'


def build_color_matrix(board_matrix, square_images):
    """
    Create an 8x8 matrix describing the detected color for each
    occupied square.

    For each square where board_matrix indicates a piece (not
    'empty'), the function runs detect_piece_color on the
    corresponding cropped square image. Empty squares are left as
    None.

    Returns an 8x8 list with values 'white', 'black' or None.
    """
    color_matrix = [[None] * 8 for _ in range(8)]
    for row in range(8):
        for col in range(8):
            if board_matrix[row][col] != 'empty':
                color_matrix[row][col] = detect_piece_color(
                    square_images[row][col]
                )
    return color_matrix

def build_fen(board_matrix, square_images, side_to_move='w'):
    """
    Assemble a full FEN string from detected piece types and
    square images.

    Parameters:
        board_matrix: 8x8 list where each entry is a piece name
            (e.g. 'Pawn', 'Rook') or the string 'empty'. Row 0 is the
            top of the board (rank 8).
        square_images: 8x8 list of the corresponding cropped square
            images in BGR color order. These images are used to
            determine piece color.
        side_to_move: 'w' or 'b' for which side is to move.

    Returns:
        A FEN string including placement, side to move and default
        values for castling/en-passant/halfmove/fullmove.
    """
    fen_rows = []

    for row in range(8):
        fen_row = ""
        empty_count = 0

        for col in range(8):
            piece = board_matrix[row][col]

            if piece == 'empty':
                empty_count += 1
            else:
                # Flush any pending empty squares
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0

                # Get the FEN letter for this piece type
                letter = PIECE_TO_FEN.get(piece, '?')

                # Detect color and adjust case
                color = detect_piece_color(square_images[row][col])
                if color == 'white':
                    fen_row += letter.upper()
                else:
                    fen_row += letter.lower()

        # Flush trailing empty squares at end of row
        if empty_count > 0:
            fen_row += str(empty_count)

        fen_rows.append(fen_row)

    # Join ranks into the placement field using '/' as the separator
    placement = "/".join(fen_rows)

    # Construct the remainder of the FEN. For simplicity this code
    # uses conservative defaults for castling rights and en-passant
    # (no castling, no en-passant target) and sets the move counters
    # to zero/one. Adjust these fields elsewhere if you have that
    # information available.
    fen = f"{placement} {side_to_move} - - 0 1"
    return fen


# ---------- TEST BLOCK ----------
if __name__ == "__main__":
    from pathlib import Path
    from board_analyzer import BoardAnalyzer

    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent

    MODEL_PATH = PROJECT_ROOT / "models" / "chess_classifier_digital.pth"
    TEST_IMAGE = PROJECT_ROOT / "data" / "test_photos" / "screenshot1.png"

    print(f"📂 Model: {MODEL_PATH}")
    print(f"🖼️  Board: {TEST_IMAGE}\n")

    board_img = cv2.imread(str(TEST_IMAGE))
    if board_img is None:
        print("❌ Could not load board image!")
        exit()

    # Analyze the board
    analyzer = BoardAnalyzer(MODEL_PATH)
    board_matrix, conf_matrix = analyzer.analyze_board(board_img)

    print("📋 DETECTED BOARD:")
    analyzer.print_board(board_matrix)

    # Get the 64 square images (needed for color detection)
    squares = analyzer.split_into_squares(board_img)
    # Rearrange flat list into 8x8 grid
    square_grid = [[None] * 8 for _ in range(8)]
    for sq in squares:
        square_grid[sq['row']][sq['col']] = sq['image']

    # Build FEN
    fen = build_fen(board_matrix, square_grid, side_to_move='w')

    print(f"\n♟️  GENERATED FEN:")
    print(f"   {fen}")
    print(f"\n💡 Paste this FEN into lichess.org/analysis to verify!")
