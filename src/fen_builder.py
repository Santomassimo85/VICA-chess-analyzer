"""
fen_builder.py
Converts a detected board (8x8 matrix + square images)
into a FEN string for chess engines.

The CNN gives piece TYPE; this module adds piece COLOR
via a classical brightness check, then assembles FEN.
"""

import cv2
import numpy as np


# Map piece type -> FEN letter (uppercase = white, we lowercase for black)
PIECE_TO_FEN = {
    'Pawn': 'P', 'Knight': 'N', 'Bishop': 'B',
    'Rook': 'R', 'Queen': 'Q', 'King': 'K'
}


def detect_piece_color(square_image):
    """
    Decide if a piece is white or black.

    Strategy: a piece has a black outline and a colored body.
    We isolate the piece from the square background, then check
    whether the piece BODY is mostly light (white) or dark (black).

    Returns 'white' or 'black'.
    """
    # Crop the center 60% (the piece body, ignore square corners)
    h, w = square_image.shape[:2]
    m_h, m_w = int(h * 0.20), int(w * 0.20)
    center = square_image[m_h:h - m_h, m_w:w - m_w]

    gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)

    # Step 1: find the piece pixels (the piece differs from the
    # uniform background). Use edges to locate the piece region.
    edges = cv2.Canny(gray, 40, 120)
    # Dilate so the edge outline becomes a filled-ish region
    kernel = np.ones((7, 7), np.uint8)
    piece_region = cv2.dilate(edges, kernel, iterations=2)

    # Step 2: get the pixels INSIDE the piece region
    piece_pixels = gray[piece_region > 0]

    # Safety: if we found almost no piece pixels, fall back
    if piece_pixels.size < 20:
        # Fallback: compare extremes in the whole center
        bright = np.count_nonzero(gray > 170)
        dark = np.count_nonzero(gray < 90)
        return 'white' if bright >= dark else 'black'

    # Step 3: within the piece, count bright vs dark pixels.
    # A white piece body is light grey/white; a black piece
    # body is dark grey/black. The outline is dark for both,
    # so we look at the brightest part of the piece.
    bright_in_piece = np.count_nonzero(piece_pixels > 150)
    dark_in_piece = np.count_nonzero(piece_pixels < 100)

    if bright_in_piece >= dark_in_piece:
        return 'white'
    return 'black'


def build_color_matrix(board_matrix, square_images):
    """
    Build an 8x8 matrix of piece colors ('white'/'black'/None).
    None means the square is empty.

    Args:
        board_matrix: 8x8 list of piece types
        square_images: 8x8 list of cropped square images (BGR)

    Returns:
        8x8 list of 'white' / 'black' / None
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
    Build a FEN string from the board.

    Args:
        board_matrix: 8x8 list of piece names ('Pawn', 'empty', ...).
                      Row 0 = top of board (rank 8).
        square_images: 8x8 list of the cropped square images (BGR).
        side_to_move: 'w' or 'b'.

    Returns:
        A FEN string.
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

    # Join rows with '/'
    placement = "/".join(fen_rows)

    # Full FEN: placement + side + castling + en passant + halfmove + fullmove
    # We use safe defaults for the extra fields.
    fen = f"{placement} {side_to_move} KQkq - 0 1"
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