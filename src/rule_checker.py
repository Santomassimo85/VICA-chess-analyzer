"""
rule_checker.py

This module validates and corrects chess positions detected by the CNN classifier.
It applies the fundamental rules of chess to catch and fix common classification errors.

The main functions check for:
- Incorrect piece counts (each player must have exactly one king)
- Piece limits (e.g., max 8 pawns per color)
- Impossible pawn placements (pawns cannot exist on the first or eighth rank)

When errors are detected, the module can automatically correct obvious mistakes,
such as converting misclassified pawns on back ranks to rooks.
"""


# Maximum allowed count per piece type, per color
MAX_COUNTS = {
    'Pawn': 8, 'Knight': 10, 'Bishop': 10,
    'Rook': 10, 'Queen': 9, 'King': 1
}


def check_position(board_matrix, color_matrix):
    """
    Check a detected position against chess rules.

    Args:
        board_matrix: 8x8 list of piece types ('Pawn', 'empty', ...)
        color_matrix: 8x8 list of 'white'/'black'/None

    Returns:
        list of human-readable warning strings (empty if all OK).
    """
    warnings = []

    # Count pieces per color
    counts = {'white': {}, 'black': {}}
    for row in range(8):
        for col in range(8):
            piece = board_matrix[row][col]
            if piece == 'empty':
                continue
            color = color_matrix[row][col]
            if color not in counts:
                continue
            counts[color][piece] = counts[color].get(piece, 0) + 1

    # Rule 1: exactly one king per color
    for color in ['white', 'black']:
        kings = counts[color].get('King', 0)
        if kings != 1:
            warnings.append(
                f"{color.capitalize()} has {kings} kings (should be 1)"
            )

    # Rule 2: piece count limits
    for color in ['white', 'black']:
        for piece, maxn in MAX_COUNTS.items():
            n = counts[color].get(piece, 0)
            if n > maxn:
                warnings.append(
                    f"{color.capitalize()} has {n} {piece}s "
                    f"(max {maxn})"
                )

    # Rule 3: no pawns on rank 1 (row 7) or rank 8 (row 0)
    for col in range(8):
        if board_matrix[0][col] == 'Pawn':
            warnings.append(f"Pawn on rank 8 ({chr(97+col)}8) - impossible")
        if board_matrix[7][col] == 'Pawn':
            warnings.append(f"Pawn on rank 1 ({chr(97+col)}1) - impossible")

    return warnings


def correct_position(board_matrix, color_matrix, conf_matrix):
    """
    Apply automatic corrections to obvious errors.

    Currently fixes:
      - Pawns on rank 1 or rank 8 -> reclassified as Rook
        (most common back-rank piece) since a pawn there
        is impossible.

    Args:
        board_matrix: 8x8 piece types (modified in place is avoided;
                      we return a new corrected matrix)
        color_matrix: 8x8 colors
        conf_matrix: 8x8 confidence scores

    Returns:
        (corrected_board_matrix, list_of_corrections_made)
    """
    # Make a copy so we don't modify the original
    corrected = [row[:] for row in board_matrix]
    corrections = []

    # Fix: pawns on the back ranks are impossible
    for col in range(8):
        # Rank 8 (row 0)
        if corrected[0][col] == 'Pawn':
            corrected[0][col] = 'Rook'
            corrections.append(
                f"{chr(97+col)}8: Pawn -> Rook (pawn cannot be on rank 8)"
            )
        # Rank 1 (row 7)
        if corrected[7][col] == 'Pawn':
            corrected[7][col] = 'Rook'
            corrections.append(
                f"{chr(97+col)}1: Pawn -> Rook (pawn cannot be on rank 1)"
            )

    return corrected, corrections


# ---------- TEST BLOCK ----------
if __name__ == "__main__":
    # A fake board with deliberate errors for testing
    test_board = [
        ['Rook', 'Knight', 'Bishop', 'Queen', 'King', 'Bishop', 'Pawn', 'Rook'],
        ['Pawn'] * 8,
        [['empty'] * 8][0][:],
        ['empty'] * 8,
        ['empty'] * 8,
        ['empty'] * 8,
        ['Pawn'] * 8,
        ['Rook', 'Knight', 'Bishop', 'Queen', 'King', 'Bishop', 'Knight', 'Pawn'],
    ]
    test_color = [['black'] * 8 for _ in range(2)] + \
                 [['white'] * 8 if r >= 6 else [None] * 8 for r in range(2, 8)]
    test_conf = [[0.9] * 8 for _ in range(8)]

    print("🧪 Testing rule checker...\n")
    warnings = check_position(test_board, test_color)
    print(f"Warnings found: {len(warnings)}")
    for w in warnings:
        print(f"  ⚠️  {w}")

    corrected, fixes = correct_position(test_board, test_color, test_conf)
    print(f"\nCorrections applied: {len(fixes)}")
    for f in fixes:
        print(f"  🔧 {f}")
