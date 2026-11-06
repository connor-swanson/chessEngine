"""Board and move encodings shared by every notebook.

Board → planes
--------------
The network always sees the board from the point of view of the side to move
("us"): if Black is to move, every square is mirrored top-to-bottom (a1 <-> a8)
and the colours are swapped.  One network therefore learns one game ("I am
moving up the board"), instead of learning White-chess and Black-chess
separately.

=====  =============================================================
plane  meaning
=====  =============================================================
0-5    our   pawns, knights, bishops, rooks, queens, king
6-11   their pawns, knights, bishops, rooks, queens, king
12     this position has occurred before (repetition count >= 2)
13     ... occurred at least twice before  (count >= 3, a draw can be claimed)
14-17  castling rights: our O-O, our O-O-O, their O-O, their O-O-O (full planes)
18     en-passant target square (only if an en-passant capture is legal)
19     half-move clock / 100  (the 50-move-rule counter)
20     all ones  (lets convolutions tell the board edge from empty squares)
=====  =============================================================

Planes are stored compactly as ``uint8`` (plane 19 holds the raw clock) and
converted to float with :func:`planes_to_float` right before the network.

Move → action index
-------------------
Also from the side-to-move's point of view (squares mirrored for Black):

* ``from * 64 + to`` for every normal move, including queen promotions
  (4096 slots);
* ``4096 + piece*24 + direction*8 + from_file`` for under-promotions to a
  knight / bishop / rook (piece 0-2), direction 0 = capture towards the a-file,
  1 = straight, 2 = capture towards the h-file (72 slots).

``ACTION_SIZE = 4168``.  Every legal chess move maps to a distinct index.
"""
from __future__ import annotations

from typing import List, Tuple

import chess
import numpy as np
import torch

NUM_PLANES = 21
HALFMOVE_PLANE = 19
ACTION_SIZE = 64 * 64 + 72

_PIECES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)
_UNDERPROMO = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}


def _masks_to_planes(masks: List[int]) -> np.ndarray:
    """List of 64-bit square masks -> (len, 8, 8) uint8 planes, [rank][file]."""
    arr = np.array(masks, dtype="<u8").view(np.uint8)
    return np.unpackbits(arr, bitorder="little").reshape(len(masks), 8, 8)


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode ``board`` as a (21, 8, 8) uint8 array (see module docstring).

    The board's move stack is used for the repetition planes, so pass the real
    game board (not ``chess.Board(fen)``) whenever history matters.
    """
    us, them = board.turn, not board.turn
    flip = us == chess.BLACK
    masks = []
    for color in (us, them):
        for pt in _PIECES:
            m = board.pieces_mask(pt, color)
            masks.append(chess.flip_vertical(m) if flip else m)

    ep = 0
    if board.ep_square is not None and board.has_legal_en_passant():
        sq = board.ep_square ^ 56 if flip else board.ep_square
        ep = chess.BB_SQUARES[sq]
    masks.append(ep)

    out = np.zeros((NUM_PLANES, 8, 8), dtype=np.uint8)
    planes = _masks_to_planes(masks)
    out[0:12] = planes[0:12]
    out[18] = planes[12]
    if board.is_repetition(2):
        out[12] = 1
        if board.is_repetition(3):
            out[13] = 1
    out[14] = board.has_kingside_castling_rights(us)
    out[15] = board.has_queenside_castling_rights(us)
    out[16] = board.has_kingside_castling_rights(them)
    out[17] = board.has_queenside_castling_rights(them)
    out[HALFMOVE_PLANE] = min(board.halfmove_clock, 255)
    out[20] = 1
    return out


def planes_to_float(x) -> torch.Tensor:
    """uint8 planes (numpy or tensor, (..., 21, 8, 8)) -> float32 network input."""
    t = torch.as_tensor(x)
    t = t.float()
    t[..., HALFMOVE_PLANE, :, :] /= 100.0
    return t


def move_to_index(move: chess.Move, turn: chess.Color) -> int:
    """Action index of ``move`` played by side ``turn`` (see module docstring)."""
    f, t = move.from_square, move.to_square
    if turn == chess.BLACK:
        f ^= 56
        t ^= 56
    if move.promotion is not None and move.promotion != chess.QUEEN:
        direction = chess.square_file(t) - chess.square_file(f) + 1
        return 4096 + _UNDERPROMO[move.promotion] * 24 + direction * 8 + chess.square_file(f)
    return f * 64 + t


def legal_moves_and_indices(board: chess.Board) -> Tuple[List[chess.Move], np.ndarray]:
    """All legal moves and their action indices (same order)."""
    moves = list(board.legal_moves)
    turn = board.turn
    idx = np.fromiter((move_to_index(m, turn) for m in moves), dtype=np.int64, count=len(moves))
    return moves, idx


def legal_mask(board: chess.Board) -> np.ndarray:
    """Boolean (ACTION_SIZE,) mask of legal action indices."""
    mask = np.zeros(ACTION_SIZE, dtype=bool)
    mask[legal_moves_and_indices(board)[1]] = True
    return mask


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Inverse of :func:`move_to_index` for a legal move of ``board``."""
    moves, idx = legal_moves_and_indices(board)
    hit = np.nonzero(idx == index)[0]
    if len(hit) == 0:
        raise ValueError(f"action {index} is not legal in {board.fen()}")
    return moves[int(hit[0])]
