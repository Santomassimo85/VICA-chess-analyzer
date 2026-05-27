"""
chess_advisor.py
Takes a FEN string, sends it to the Stockfish engine,
and returns a human-friendly analysis: who's winning
and what the best move is.
"""

import chess
import chess.engine
from pathlib import Path


class ChessAdvisor:
    """Analyzes a chess position using the Stockfish engine."""

    def __init__(self, engine_path, think_time=1.0):
        """
        Args:
            engine_path: path to the stockfish.exe file
            think_time: how many seconds Stockfish thinks per move
        """
        self.engine_path = str(engine_path)
        self.think_time = think_time

        # Verify the engine file exists
        if not Path(engine_path).exists():
            raise FileNotFoundError(
                f"Stockfish engine not found at: {engine_path}"
            )
        print(f"🐟 Stockfish engine: {engine_path}")

    def analyze(self, fen):
        """
        Analyze a position given as a FEN string.

        Returns a dict with:
            'valid': True/False (is the FEN a legal position?)
            'best_move': the best move (e.g. 'e2e4')
            'best_move_san': human-readable move (e.g. 'e4')
            'evaluation': score (positive = white better)
            'eval_text': friendly description
            'turn': 'White' or 'Black'
        """
        # Step 1: validate the FEN
        try:
            board = chess.Board(fen)
        except ValueError as e:
            return {
                'valid': False,
                'error': f"Invalid FEN: {e}"
            }

        # Check the position is actually legal
        if not board.is_valid():
            return {
                'valid': False,
                'error': "The detected position is not a legal "
                         "chess position (likely a classification error)."
            }

        # Step 2: ask Stockfish
        with chess.engine.SimpleEngine.popen_uci(self.engine_path) as engine:
            # Get the best move
            result = engine.play(
                board, chess.engine.Limit(time=self.think_time)
            )
            best_move = result.move

            # Get the evaluation score
            info = engine.analyse(
                board, chess.engine.Limit(time=self.think_time)
            )
            score = info['score'].white()

        # Step 3: translate into friendly output
        best_move_san = board.san(best_move)
        turn = 'White' if board.turn == chess.WHITE else 'Black'

        # Build evaluation text
        if score.is_mate():
            mate_in = score.mate()
            eval_value = f"Mate in {abs(mate_in)}"
            if mate_in > 0:
                eval_text = f"White has forced mate in {abs(mate_in)}!"
            else:
                eval_text = f"Black has forced mate in {abs(mate_in)}!"
        else:
            # Score is in centipawns (100 = 1 pawn advantage)
            cp = score.score()
            eval_value = cp / 100.0
            if abs(cp) < 50:
                eval_text = "The position is roughly equal."
            elif cp > 0:
                eval_text = f"White is better (+{eval_value:.1f})."
            else:
                eval_text = f"Black is better ({eval_value:.1f})."

        return {
            'valid': True,
            'best_move': best_move.uci(),
            'best_move_san': best_move_san,
            'evaluation': eval_value,
            'eval_text': eval_text,
            'turn': turn
        }


# ---------- TEST BLOCK ----------
if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent

    ENGINE_PATH = PROJECT_ROOT / "engine" / "stockfish.exe"

    print("🧪 Testing ChessAdvisor...\n")
    advisor = ChessAdvisor(ENGINE_PATH, think_time=1.0)

    # Test 1: starting position
    test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    print(f"\n📋 Test position (starting position):")
    print(f"   FEN: {test_fen}\n")

    result = advisor.analyze(test_fen)

    if result['valid']:
        print(f"   ✅ Turn: {result['turn']} to move")
        print(f"   📊 {result['eval_text']}")
        print(f"   💡 Best move: {result['best_move_san']} "
              f"({result['best_move']})")
    else:
        print(f"   ❌ {result['error']}")