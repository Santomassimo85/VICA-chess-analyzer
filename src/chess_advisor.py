"""
ChessAdvisor Module

This module provides a wrapper around the Stockfish chess engine. Given a chess
position in FEN (Forsyth-Edwards Notation) format, it queries Stockfish for
analysis and returns the evaluation and best move in a user-friendly format.

The ChessAdvisor class handles engine initialization, position validation, and
result formatting. It abstracts away the complexity of communicating with the
chess engine, making it easy to analyze positions in a larger application.
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
        Returns top moves + principal variation.
        """
        # Step 1: validate the FEN
        try:
            board = chess.Board(fen)
        except ValueError as e:
            return {'valid': False, 'error': f"Invalid FEN: {e}"}

        # Relaxed check: just require both kings
        if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
            return {'valid': False,
                    'error': "Missing king(s) - classification error."}

        # Step 2: ask Stockfish for TOP 3 moves (MultiPV)
        with chess.engine.SimpleEngine.popen_uci(self.engine_path) as engine:
            infos = engine.analyse(
                board,
                chess.engine.Limit(time=self.think_time),
                multipv=3)

        # Step 3: parse top moves
        top_moves = []
        for info in infos:
            pv = info.get('pv', [])
            if not pv:
                continue
            move = pv[0]
            score = info['score'].white()

            # Format the evaluation
            if score.is_mate():
                eval_str = f"Mate in {abs(score.mate())}"
            else:
                cp = score.score()
                eval_str = f"{cp/100:+.2f}"

            # Build principal variation (up to 5 moves)
            pv_board = board.copy()
            pv_san = []
            for pv_move in pv[:5]:
                pv_san.append(pv_board.san(pv_move))
                pv_board.push(pv_move)

            top_moves.append({
                'move': move.uci(),
                'move_san': board.san(move),
                'evaluation': eval_str,
                'pv': pv_san,
            })

        if not top_moves:
            return {'valid': False, 'error': "Engine returned no moves."}

        # Build a friendly summary text for the best move
        best = top_moves[0]
        turn = 'White' if board.turn == chess.WHITE else 'Black'

        # Evaluation summary
        first_score = infos[0]['score'].white()
        if first_score.is_mate():
            m = first_score.mate()
            if m > 0:
                eval_text = f"White has forced mate in {abs(m)}!"
            else:
                eval_text = f"Black has forced mate in {abs(m)}!"
        else:
            cp = first_score.score()
            v = cp / 100.0
            if abs(cp) < 50:
                eval_text = "The position is roughly equal."
            elif cp > 0:
                eval_text = f"White is better (+{v:.1f})."
            else:
                eval_text = f"Black is better ({v:.1f})."

        return {
            'valid': True,
            'best_move': best['move'],
            'best_move_san': best['move_san'],
            'evaluation': best['evaluation'],
            'eval_text': eval_text,
            'turn': turn,
            'top_moves': top_moves,
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
