"""Create a synthetic dataset of chess pieces from a chessboard screenshot."""

import random
from pathlib import Path
from PIL import Image, ImageEnhance

# Paths and generation settings.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

SCREENSHOT = PROJECT_ROOT / "data" / "test_photos" / "screenshot1.png"
OUTPUT_DIR = PROJECT_ROOT / "data" / "synthetic_digital"

IMAGES_PER_CLASS = 80
IMG_SIZE = 224

CLASSES = ["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"]

random.seed(42)


# Extract one example square for each piece type.
def extract_pieces_from_screenshot(screenshot_path):
    """Return sample squares for each piece type from the board image."""
    board = Image.open(screenshot_path).convert("RGB")
    w, h = board.size
    sq = w // 8

    def get_square(row, col):
        left, top = col * sq, row * sq
        return board.crop((left, top, left + sq, top + sq))

    # The back rank layout follows standard chess notation:
    # a=Rook, b=Knight, c=Bishop, d=Queen, e=King, and so on.
    piece_columns = {
        "Rook": 0,
        "Knight": 1,
        "Bishop": 2,
        "Queen": 3,
        "King": 4,
    }

    pieces = {cls: [] for cls in CLASSES}

    for cls, col in piece_columns.items():
        black_piece = get_square(0, col)   # row 0 = black back rank
        white_piece = get_square(7, col)   # row 7 = white back rank
        pieces[cls].append(white_piece)
        pieces[cls].append(black_piece)

    # Pawns are taken from the second rank on each side.
    pieces["Pawn"].append(get_square(6, 0))
    pieces["Pawn"].append(get_square(1, 0))

    print("  Extracted variants:")
    for cls in CLASSES:
        print(f"    {cls:8s}: {len(pieces[cls])} variants")

    return pieces, sq


# Apply simple visual variations to each square.
def augment(piece_square):
    """Add a few light random adjustments and resize the image."""
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


# Script entry point.
def main():
    print("=" * 55)
    print("SYNTHETIC DIGITAL CHESS PIECE GENERATOR")
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

        # Alternate between the available examples for each class.
        for i in range(IMAGES_PER_CLASS):
            variant = pieces[cls][i % len(pieces[cls])]
            img = augment(variant)
            img.save(cls_dir / f"{cls}_{i:03d}.png")
            total += 1

        print(f"  {cls:8s}: {IMAGES_PER_CLASS} images created")

    print(f"\n✅ Done! Generated {total} synthetic images.")
    print(f"📁 Saved in: {OUTPUT_DIR}")
    print("\n⚠️  Check data/synthetic_digital/Knight/ to make sure both colors are present.")


if __name__ == "__main__":
    main()
