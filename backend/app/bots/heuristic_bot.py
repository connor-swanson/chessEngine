from __future__ import annotations

import random
from typing import Optional

import chess

from .base import SyncBot

_MATERIAL = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def _evaluate(board: chess.Board) -> float:
    score = 0.0
    for piece in board.piece_map().values():
        v = _MATERIAL[piece.piece_type]
        score += v if piece.color == chess.WHITE else -v
    return score


class HeuristicBot(SyncBot):
    """Depth-1 material bot.  Takes mate-in-1; otherwise picks uniformly at
    random among the moves that maximise material after the move."""

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def select_move(self, board: chess.Board) -> chess.Move:
        board = board.copy()
        sign = 1 if board.turn == chess.WHITE else -1
        best_score = float("-inf")
        best_moves = []
        for move in list(board.legal_moves):
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move
            score = sign * _evaluate(board)
            board.pop()
            if score > best_score:
                best_score, best_moves = score, [move]
            elif score == best_score:
                best_moves.append(move)
        return self._rng.choice(best_moves)
