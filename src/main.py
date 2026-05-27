"""
main.py
The complete Chess Position Analyzer pipeline.

Pipeline:
  image -> board detection -> classification
  -> rule-based correction -> FEN -> Stockfish
"""

import cv2
import sys
from pathlib import Path

from board_analyzer import BoardAnalyzer
from fen_builder import build_fen, build_color_matrix
from chess_advisor import ChessAdvisor
from rule_checker import check_position, correct_position


def ask_choice(question, options):
    """Ask the user to pick one option. Returns the chosen string."""
    while True:
        print(f"\n{question}")
        for i, opt in enumerate(options, 1):
            print(f"  {i}. {opt}")
        choice = input("Enter the number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("❌ Invalid choice, try again.")


def flip_board(board_matrix):
    """Flip the board 180 degrees (for Black-at-bottom screenshots)."""
    return [row[::-1] for row in board_matrix[::-1]]


def print_fen_board(fen):
    """Print the board from a FEN string, showing colors."""
    placement = fen.split()[0]
    rows = placement.split('/')
    print("  +" + "---+" * 8)
    for i, row in enumerate(rows):
        line = f"{8-i} |"
        for ch in row:
            if ch.isdigit():
                line += " . |" * int(ch)
            else:
                line += f" {ch} |"
        print(line)
        print("  +" + "---+" * 8)
    print("    a   b   c   d   e   f   g   h")
    print("\n  (UPPERCASE = White pieces, lowercase = black pieces)")


def analyze_chess_image(image_path, model_path, engine_path,
                        side_to_move, black_at_bottom):
    """Run the full pipeline on a chess board image."""
    print("=" * 55)
    print("♟️  CHESS POSITION ANALYZER")
    print("=" * 55)

    # ---- STAGE 1: Load the image ----
    print(f"\n📂 Loading image: {image_path}")
    board_img = cv2.imread(str(image_path))
    if board_img is None:
        return {'success': False, 'error': 'Could not load image.'}
    print(f"   ✅ Image loaded ({board_img.shape[1]}x{board_img.shape[0]})")

    # ---- STAGE 2: Detect & classify the board ----
    print("\n🔍 Stage 2: Analyzing the board...")
    analyzer = BoardAnalyzer(model_path)
    board_matrix, conf_matrix = analyzer.analyze_board(board_img)

    # Get the 64 square images
    squares = analyzer.split_into_squares(board_img)
    square_grid = [[None] * 8 for _ in range(8)]
    for sq in squares:
        square_grid[sq['row']][sq['col']] = sq['image']

    # ---- STAGE 2b: Handle board orientation ----
    if black_at_bottom:
        print("   🔄 Black is at the bottom -> flipping to standard view")
        board_matrix = flip_board(board_matrix)
        conf_matrix = flip_board(conf_matrix)
        square_grid = [row[::-1] for row in square_grid[::-1]]

    print("\n📋 Detected position (piece types only):")
    analyzer.print_board(board_matrix)

    # ---- STAGE 3: Rule-based post-processing ----
    print("\n🔧 Stage 3: Rule-based validation & correction...")
    color_matrix = build_color_matrix(board_matrix, square_grid)

    # Check for rule violations
    warnings = check_position(board_matrix, color_matrix)
    if warnings:
        print("   ⚠️  Rule violations detected:")
        for w in warnings:
            print(f"      - {w}")
    else:
        print("   ✅ No rule violations detected.")

    # Apply automatic corrections
    board_matrix, corrections = correct_position(
        board_matrix, color_matrix, conf_matrix
    )
    if corrections:
        print("   🔧 Corrections applied:")
        for c in corrections:
            print(f"      - {c}")
        # Rebuild color matrix after correction
        color_matrix = build_color_matrix(board_matrix, square_grid)

    # ---- STAGE 4: Build the FEN ----
    print("\n♟️  Stage 4: Building FEN notation...")
    fen = build_fen(board_matrix, square_grid, side_to_move=side_to_move)
    print(f"   FEN: {fen}")

    print("\n🎨 Final position WITH colors:")
    print_fen_board(fen)

    # ---- STAGE 5: Stockfish analysis ----
    print("\n🐟 Stage 5: Consulting Stockfish...")
    advisor = ChessAdvisor(engine_path, think_time=1.0)
    result = advisor.analyze(fen)

    # ---- STAGE 6: Present results ----
    print("\n" + "=" * 55)
    print("📊 ANALYSIS RESULT")
    print("=" * 55)

    if not result['valid']:
        print(f"⚠️  {result['error']}")
        print("\n💡 Some pieces were likely misclassified.")
        print("   Check the detected board above against your screenshot.")
        return {'success': False, 'error': result['error'], 'fen': fen}

    print(f"   Turn:        {result['turn']} to move")
    print(f"   Evaluation:  {result['eval_text']}")
    print(f"   Best move:   {result['best_move_san']} "
          f"({result['best_move']})")
    print("=" * 55)
    print("\n💡 Tip: paste the FEN into lichess.org/analysis to verify!")

    return {
        'success': True,
        'fen': fen,
        'board_matrix': board_matrix,
        'turn': result['turn'],
        'evaluation': result['eval_text'],
        'best_move': result['best_move_san']
    }


# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent

    MODEL_PATH = PROJECT_ROOT / "models" / "chess_classifier_digital.pth"
    ENGINE_PATH = PROJECT_ROOT / "engine" / "stockfish.exe"

    print("=" * 55)
    print("  Welcome to Chess Position Analyzer!")
    print("=" * 55)

    # --- Get the image path ---
    if len(sys.argv) > 1:
        IMAGE_PATH = Path(sys.argv[1])
    else:
        print("\n📂 Place your screenshot in: data/test_photos/")
        filename = input("Enter the screenshot filename "
                          "(e.g. screenshot1.png): ").strip()
        IMAGE_PATH = PROJECT_ROOT / "data" / "test_photos" / filename

    if not IMAGE_PATH.exists():
        print(f"❌ File not found: {IMAGE_PATH}")
        sys.exit()

    # --- Ask whose turn it is ---
    turn_choice = ask_choice("Whose turn is it to move?",
                             ["White", "Black"])
    side = 'w' if turn_choice == "White" else 'b'

    # --- Ask the board orientation ---
    orient_choice = ask_choice(
        "Which color is at the BOTTOM of your screenshot?",
        ["White (standard)", "Black (flipped)"])
    black_bottom = (orient_choice == "Black (flipped)")

    # --- Run the pipeline ---
    analyze_chess_image(
        image_path=IMAGE_PATH,
        model_path=MODEL_PATH,
        engine_path=ENGINE_PATH,
        side_to_move=side,
        black_at_bottom=black_bottom
    )