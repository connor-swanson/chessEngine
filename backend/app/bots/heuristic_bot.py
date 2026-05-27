import asyncio
import chess
from .base import Bot

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
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            v = _MATERIAL[piece.piece_type]
            score += v if piece.color == chess.WHITE else -v
    return score


def _best_move(board: chess.Board) -> chess.Move:
    maximizing = board.turn == chess.WHITE
    best_move = None
    best_score = float("-inf") if maximizing else float("inf")

    for move in board.legal_moves:
        board.push(move)
        if board.is_checkmate():
            board.pop()
            return move
        score = _evaluate(board)
        board.pop()
        if maximizing and score > best_score:
            best_score, best_move = score, move
        elif not maximizing and score < best_score:
            best_score, best_move = score, move

    return best_move or next(iter(board.legal_moves))


class HeuristicBot(Bot):
    async def choose_move(self, board: chess.Board) -> chess.Move:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _best_move, board.copy())
