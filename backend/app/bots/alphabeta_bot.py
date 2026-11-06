from __future__ import annotations

import random
from typing import List, Optional

import chess

from .base import SyncBot

# ---------------------------------------------------------------------------
# Evaluation tables (Tomasz Michniewski's "Simplified Evaluation Function").
# Tables are written from White's point of view, rank 8 first (as printed).
# ---------------------------------------------------------------------------
_PIECE_VALUE = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0,
}

_PST_RAW = {
    chess.PAWN: [
         0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
         5,  5, 10, 25, 25, 10,  5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5, -5,-10,  0,  0,-10, -5,  5,
         5, 10, 10,-20,-20, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0],
    chess.KNIGHT: [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50],
    chess.BISHOP: [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20],
    chess.ROOK: [
          0,  0,  0,  0,  0,  0,  0,  0,
          5, 10, 10, 10, 10, 10, 10,  5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
          0,  0,  0,  5,  5,  0,  0,  0],
    chess.QUEEN: [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
         -5,  0,  5,  5,  5,  5,  0, -5,
          0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20],
}
_KING_MG = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20]
_KING_EG = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50]


def _build(raw: List[int], value: int):
    """Return (white_table, black_table) indexed by python-chess square,
    material included.  Black uses the vertically mirrored table."""
    white = [value + raw[sq ^ 56] for sq in range(64)]
    black = [value + raw[sq] for sq in range(64)]
    return white, black


_PST = {pt: _build(_PST_RAW[pt], _PIECE_VALUE[pt]) for pt in _PST_RAW}
_KING_TABLES_MG = _build(_KING_MG, 0)
_KING_TABLES_EG = _build(_KING_EG, 0)

MATE = 100000
_MATE_BOUND = MATE - 1000
INF = 10 ** 7
_QS_MAX_DEPTH = 8
# Combined non-pawn, non-king material (both sides) at/below which we use the
# endgame king table.
_ENDGAME_MATERIAL = 2600

_NON_PAWN_VALUE = {chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900}
_MVV = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
        chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
_LVA = {chess.PAWN: 6, chess.KNIGHT: 5, chess.BISHOP: 4,
        chess.ROOK: 3, chess.QUEEN: 2, chess.KING: 1}

_RANK7 = chess.BB_RANK_7
_RANK2 = chess.BB_RANK_2


def evaluate(board: chess.Board) -> int:
    """Static evaluation in centipawns from the side-to-move's perspective."""
    score = 0
    npm = 0
    for pt, (wt, bt) in _PST.items():
        wbb = board.pieces_mask(pt, chess.WHITE)
        bbb = board.pieces_mask(pt, chess.BLACK)
        for sq in chess.scan_forward(wbb):
            score += wt[sq]
        for sq in chess.scan_forward(bbb):
            score -= bt[sq]
        if pt in _NON_PAWN_VALUE:
            npm += _NON_PAWN_VALUE[pt] * (chess.popcount(wbb) + chess.popcount(bbb))
    wt, bt = _KING_TABLES_EG if npm <= _ENDGAME_MATERIAL else _KING_TABLES_MG
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    if wk is not None:
        score += wt[wk]
    if bk is not None:
        score -= bt[bk]
    return score if board.turn == chess.WHITE else -score


class AlphaBetaBot(SyncBot):
    """Negamax alpha-beta with quiescence search, MVV-LVA ordering and a
    material + piece-square-table evaluation.

    noise_cp: at the root, pick uniformly at random among all moves whose score
    is within `noise_cp` centipawns of the best (0 -> random among exact ties).
    """

    def __init__(self, depth: int = 2, seed: Optional[int] = None, noise_cp: int = 0):
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.depth = depth
        self.noise_cp = noise_cp
        self._rng = random.Random(seed)
        self.nodes = 0

    # -- move ordering -----------------------------------------------------
    @staticmethod
    def _order_key(board: chess.Board, move: chess.Move) -> int:
        score = 0
        if board.is_capture(move):
            if board.is_en_passant(move):
                victim = chess.PAWN
            else:
                victim = board.piece_type_at(move.to_square)
            attacker = board.piece_type_at(move.from_square)
            score = 10000 + _MVV[victim] * 10 + _LVA[attacker]
        if move.promotion:
            score += 9000 + _MVV[move.promotion]
        return score

    def _ordered(self, board: chess.Board) -> List[chess.Move]:
        moves = list(board.legal_moves)
        moves.sort(key=lambda m: self._order_key(board, m), reverse=True)
        return moves

    # -- search ------------------------------------------------------------
    def _quiesce(self, board: chess.Board, alpha: int, beta: int, qdepth: int) -> int:
        self.nodes += 1
        stand = evaluate(board)
        if qdepth >= _QS_MAX_DEPTH:
            return stand
        if stand >= beta:
            return stand
        if stand > alpha:
            alpha = stand
        best = stand

        moves = list(board.generate_legal_captures())
        # Non-capturing promotions.
        promo_mask = _RANK7 if board.turn == chess.WHITE else _RANK2
        pawns = board.pawns & board.occupied_co[board.turn] & promo_mask
        if pawns:
            for m in board.generate_legal_moves(from_mask=pawns):
                if m.promotion and not board.is_capture(m):
                    moves.append(m)
        if not moves:
            return stand
        moves.sort(key=lambda m: self._order_key(board, m), reverse=True)
        for move in moves:
            board.push(move)
            score = -self._quiesce(board, -beta, -alpha, qdepth + 1)
            board.pop()
            if score > best:
                best = score
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        break
        return best

    def _negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        moves = self._ordered(board)
        if not moves:
            return -(MATE - ply) if board.is_check() else 0
        # Draw detection (checkmate was handled above).
        if board.halfmove_clock >= 100 or board.is_insufficient_material():
            return 0
        if board.halfmove_clock >= 4 and board.is_repetition(2):
            return 0
        if depth <= 0:
            return self._quiesce(board, alpha, beta, 0)
        self.nodes += 1
        best = -INF
        for move in moves:
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            if score > best:
                best = score
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        break
        return best

    # -- root --------------------------------------------------------------
    def select_move(self, board: chess.Board) -> chess.Move:
        board = board.copy()  # never touch the caller's board
        moves = self._ordered(board)
        if len(moves) == 1:
            return moves[0]
        self.nodes = 0

        scores = {m: 0 for m in moves}
        # Iterative deepening only to improve root move ordering.
        for d in range(1, self.depth):
            ordered = sorted(moves, key=lambda m: scores[m], reverse=True)
            alpha = -INF
            for m in ordered:
                board.push(m)
                s = -self._negamax(board, d - 1, -INF, -alpha, 1)
                board.pop()
                scores[m] = s
                if s > alpha:
                    alpha = s
            moves = ordered

        moves = sorted(moves, key=lambda m: scores[m], reverse=True)
        best = -INF
        results = {}
        margin = self.noise_cp
        for m in moves:
            # Lower bound alpha: anything <= best - margin - 1 is irrelevant.
            lo = -INF if best == -INF else best - margin - 1
            board.push(m)
            s = -self._negamax(board, self.depth - 1, -INF, -lo, 1)
            board.pop()
            results[m] = s
            if s > best:
                best = s
        candidates = [m for m, s in results.items() if s >= best - margin and s > (-INF)]
        # Moves whose search failed low report a bound <= lo < best - margin,
        # so they are excluded above.
        return self._rng.choice(candidates)
