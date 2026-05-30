"""
Streamlit app for VICA, the visual chess analyzer.

Run with: streamlit run src/app.py
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

from board_analyzer import BoardAnalyzer
from fen_builder import build_fen, build_color_matrix
from chess_advisor import ChessAdvisor
from rule_checker import check_position, correct_position

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_PATH = PROJECT_ROOT / "models" / "chess_classifier_digital.pth"
ENGINE_PATH = PROJECT_ROOT / "engine" / "stockfish.exe"


# Helpers
def flip_board(board_matrix):
    return [row[::-1] for row in board_matrix[::-1]]


def fen_to_grid(fen):
    placement = fen.split()[0]
    grid = []
    for row in placement.split('/'):
        line = []
        for ch in row:
            if ch.isdigit():
                line.extend(['.'] * int(ch))
            else:
                line.append(ch)
        grid.append(line)
    return grid


@st.cache_resource
def load_analyzer():
    return BoardAnalyzer(str(MODEL_PATH))


@st.cache_resource
def load_advisor():
    return ChessAdvisor(str(ENGINE_PATH), think_time=1.0)


# Page setup
st.set_page_config(page_title="VICA - Visual Chess Analyzer",
                   page_icon="🌿", layout="centered")

# Simple custom styling
st.markdown("""
<style>
    .stApp { background-color: #0e1a0e; }
    h1 { color: #66bb6a !important; }
    h2, h3 { color: #81c784 !important; }
    .stButton button {
        background-color: #2e7d32; color: white;
        border-radius: 10px; border: none; font-weight: 600;
        padding: 0.5rem 1.5rem;
    }
    .stButton button:hover { background-color: #1b5e20; color: white; }
   div[data-testid="stMetric"] {
        background-color: #1a2e1a; border-radius: 12px;
        padding: 12px; border: 1px solid #2e4e2e;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🌿 VICA")
st.subheader("Visual Chess Analyzer")
st.markdown(
    "Upload a screenshot of a digital chess board. VICA detects the "
    "position, builds the FEN notation, and uses the **Stockfish** engine "
    "to suggest the best move.")
st.divider()

# Step 1: upload
st.markdown("### 1 · Upload your board")
uploaded = st.file_uploader(
    "Choose a chess board screenshot (PNG or JPG)",
    type=['png', 'jpg', 'jpeg'])

if uploaded is not None:
    pil_image = Image.open(uploaded).convert("RGB")
    st.image(pil_image, caption="Uploaded board", width=300)

    # Step 2: game settings
    st.markdown("### 2 · Game settings")
    col_a, col_b = st.columns(2)
    with col_a:
        turn_choice = st.selectbox(
            "Whose turn is it to move?", ["White", "Black"])
    with col_b:
        orientation = st.selectbox(
            "Which colour is at the bottom of the image?",
            ["White (standard)", "Black (flipped)"])

    side = 'w' if turn_choice == "White" else 'b'
    black_at_bottom = (orientation == "Black (flipped)")

    # Step 3: analyze
    st.markdown("### 3 · Run the analysis")
    if st.button("🌿 Analyse position", type="primary"):

        board_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        with st.spinner("Analysing the board, please wait..."):
            analyzer = load_analyzer()
            board_matrix, conf_matrix = analyzer.analyze_board(board_img)

            squares = analyzer.split_into_squares(board_img)
            square_grid = [[None] * 8 for _ in range(8)]
            for sq in squares:
                square_grid[sq['row']][sq['col']] = sq['image']

            if black_at_bottom:
                board_matrix = flip_board(board_matrix)
                square_grid = [row[::-1] for row in square_grid[::-1]]

            # Debug output
            print("\n=== DEBUG board_matrix ===")
            for r in range(8):
                row_str = f"{8-r}: "
                for c in range(8):
                    p = board_matrix[r][c]
                    row_str += (p[0] if p != 'empty' else '.') + " "
                print(row_str)
            print("   a b c d e f g h\n")

            color_matrix = build_color_matrix(board_matrix, square_grid)
            warnings = check_position(board_matrix, color_matrix)
            board_matrix, corrections = correct_position(
                board_matrix, color_matrix, conf_matrix)

            fen = build_fen(board_matrix, square_grid, side_to_move=side)

            advisor = load_advisor()
            result = advisor.analyze(fen)

        # Results
        st.divider()
        st.markdown("### 🪵 Detected Position")

        grid = fen_to_grid(fen)
        board_text = ""
        for i, row in enumerate(grid):
            board_text += f"{8-i}  " + "  ".join(row) + "\n"
        board_text += "   a  b  c  d  e  f  g  h"
        st.code(board_text, language=None)
        st.caption("UPPERCASE = White pieces · lowercase = black pieces")

        st.text_input("FEN notation", value=fen)

        if warnings:
            st.warning("Rule check: " + "; ".join(warnings))
        if corrections:
            st.info("Auto-corrections: " + "; ".join(corrections))

        # Engine result
        st.divider()
        st.markdown("### 🌳 Engine Analysis")

        if not result['valid']:
            st.error(f"Could not analyse: {result['error']}")
            st.markdown(
                "This usually means a piece was misclassified. "
                "Try a clearer screenshot, cropped tightly to the board.")
        else:
            st.metric("Turn", result['turn'])
            st.metric("Evaluation", result['eval_text'])
            st.metric("Best move", result['best_move_san'])
            st.success(f"🌿 Recommended move: **{result['best_move_san']}** "
                       f"({result['best_move']})")

            # ----- Top alternative moves -----
            top = result.get('top_moves', [])
            if len(top) > 1:
                st.markdown("### 🌳 Alternative moves")
                for i, m in enumerate(top, 1):
                    label = "Best" if i == 1 else f"Option {i}"
                    pv_text = " → ".join(m['pv']) if m['pv'] else m['move_san']
                    st.markdown(
                        f"**{label}: {m['move_san']}** "
                        f"(eval `{m['evaluation']}`)  \n"
                        f"Continuation: `{pv_text}`")

        st.caption("Tip: paste the FEN into lichess.org/analysis to verify.")

else:
    st.info("⬆️ Upload a chess board screenshot above to begin.")

# Footer
st.divider()
st.caption(
    "🌱 VICA is intended for post-game analysis and study only. "
    "Using engine assistance during live rated games violates the "
    "rules of chess platforms and federations.")
