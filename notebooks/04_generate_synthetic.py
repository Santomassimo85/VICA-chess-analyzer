"""
04_generate_synthetic.py
Generates a synthetic dataset of digital chess pieces.

FIXED: properly extracts BOTH white and black variants
of every piece type (previous version dropped white
pieces for Knight/Bishop/Rook since each side has 2).
"""

import random
from pathlib import Path
from PIL import Image, ImageEnhance

# ---------- CONFIGURATION ----------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

SCREENSHOT = PROJECT_ROOT / "data" / "test_photos" / "screenshot1.png"
OUTPUT_DIR = PROJECT_ROOT / "data" / "synthetic_digital"

IMAGES_PER_CLASS = 80
IMG_SIZE = 224

CLASSES = ["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"]

random.seed(42)


# ---------- STEP 1: CUT PIECE-SQUARES FROM SCREENSHOT ----------
def extract_pieces_from_screenshot(screenshot_path):
    """
    Cut piece-squares from a starting position.
    Grabs BOTH a white and a black variant for each class.
    Returns dict: {'Pawn': [white_img, black_img], ...}
    """
    board = Image.open(screenshot_path).convert("RGB")
    w, h = board.size
    sq = w // 8

    def get_square(row, col):
        left, top = col * sq, row * sq
        return board.crop((left, top, left + sq, top + sq))

    # We pick ONE specific column per piece type.
    # Back rank order: a=Rook b=Knight c=Bishop d=Queen
    #                  e=King  f=Bishop g=Knight h=Rook
    # column index ->  0       1        2        3
    #                  4       5        6        7
    piece_columns = {
        "Rook": 0,     # column a
        "Knight": 1,   # column b
        "Bishop": 2,   # column c
        "Queen": 3,    # column d
        "King": 4,     # column e
        # Pawn handled separately (whole row of pawns)
    }

    pieces = {cls: [] for cls in CLASSES}

    for cls, col in piece_columns.items():
        black_piece = get_square(0, col)   # row 0 = black back rank
        white_piece = get_square(7, col)   # row 7 = white back rank
        pieces[cls].append(white_piece)
        pieces[cls].append(black_piece)

    # Pawns: row 1 = black pawns, row 6 = white pawns
    pieces["Pawn"].append(get_square(6, 0))  # white pawn
    pieces["Pawn"].append(get_square(1, 0))  # black pawn

    print("  Extracted variants (should be 2 each: white + black):")
    for cls in CLASSES:
        print(f"    {cls:8s}: {len(pieces[cls])} variants")

    return pieces, sq


# ---------- STEP 2: AUGMENT A PIECE-SQUARE ----------
def augment(piece_square):
    """Apply small random variations. Returns RGB IMG_SIZE image."""
    img = piece_square.convert("RGB")

    brightness = random.uniform(0.80, 1.20)
    img = ImageEnhance.Brightness(img).enhance(brightness)

    contrast = random.uniform(0.85, 1.15)
    img = ImageEnhance.Contrast(img).enhance(contrast)

    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    w, h = img.size
    crop_frac = random.uniform(0.0, 0.12)
    cw, ch = int(w * crop_frac), int(h * crop_frac)
    img = img.crop((cw, ch, w - cw, h - ch))

    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    return img


# ---------- MAIN ----------
def main():
    print("=" * 55)
    print("SYNTHETIC DIGITAL CHESS PIECE GENERATOR (FIXED v2)")
    print("=" * 55)

    if not SCREENSHOT.exists():
        print(f"❌ Screenshot not found: {SCREENSHOT}")
        return

    print(f"📂 Input screenshot: {SCREENSHOT}\n")
    print("✂️  Step 1: Extracting piece-squares...")
    pieces, sq = extract_pieces_from_screenshot(SCREENSHOT)
    print(f"   (each square is {sq}x{sq} pixels)\n")

    print("🎨 Step 2: Generating augmented images...")
    total = 0
    for cls in CLASSES:
        cls_dir = OUTPUT_DIR / cls
        cls_dir.mkdir(parents=True, exist_ok=True)

        # Alternate white/black so each class is balanced
        for i in range(IMAGES_PER_CLASS):
            variant = pieces[cls][i % len(pieces[cls])]
            img = augment(variant)
            img.save(cls_dir / f"{cls}_{i:03d}.png")
            total += 1

        print(f"  {cls:8s}: {IMAGES_PER_CLASS} images created")

    print(f"\n✅ Done! Generated {total} synthetic images.")
    print(f"📁 Saved in: {OUTPUT_DIR}")
    print("\n⚠️  CHECK: open data/synthetic_digital/Knight/ and confirm")
    print("    you now see BOTH white and black knights, all SOLID.")


if __name__ == "__main__":
    main()