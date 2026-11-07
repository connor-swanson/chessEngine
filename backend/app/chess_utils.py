from __future__ import annotations

from typing import Optional, Tuple

import chess
import numpy as np

# Maps (piece_type, color) to a plane index in the (14, 8, 8) tensor
_PIECE_PLANE: dict[tuple[int, bool], int] = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}


def fen_to_tensor(fen: str) -> np.ndarray:
    """Convert a FEN string to a (14, 8, 8) float32 numpy array.

    Planes 0-11: one per piece type × color (white P,N,B,R,Q,K then black).
    Plane 12:    side to move (all 1s = white, all 0s = black).
    Plane 13:    castling rights (KQkq encoded in top-left 4 cells).
    """
    board = chess.Board(fen)
    tensor = np.zeros((14, 8, 8), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            plane = _PIECE_PLANE[(piece.piece_type, piece.color)]
            tensor[plane, chess.square_rank(square), chess.square_file(square)] = 1.0

    if board.turn == chess.WHITE:
        tensor[12] = 1.0

    for i, has_right in enumerate([
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    ]):
        if has_right:
            tensor[13, 0, i] = 1.0

    return tensor


_TERMINATION = {
    chess.Termination.CHECKMATE: "checkmate",
    chess.Termination.STALEMATE: "stalemate",
    chess.Termination.INSUFFICIENT_MATERIAL: "insufficient material",
    chess.Termination.SEVENTYFIVE_MOVES: "75-move rule",
    chess.Termination.FIVEFOLD_REPETITION: "repetition",
    chess.Termination.FIFTY_MOVES: "50-move rule",
    chess.Termination.THREEFOLD_REPETITION: "repetition",
}


def board_outcome(board: chess.Board) -> Optional[Tuple[str, str]]:
    """(result, termination) if the game is over on the board, else None.

    Threefold repetition and the 50-move rule end the game automatically
    (as on most online sites) so the UI and the bots agree on when it's over.
    """
    outcome = board.outcome()
    if outcome is not None:
        return outcome.result(), _TERMINATION.get(outcome.termination, outcome.termination.name.lower())
    # outcome(claim_draw=True) would also end the game when a repetition is merely
    # *available* to the side to move; only end it once it has actually happened.
    if board.is_repetition(3):
        return "1/2-1/2", "repetition"
    if board.halfmove_clock >= 100:
        return "1/2-1/2", "50-move rule"
    return None


def board_to_state(board: chess.Board, game_id: int, ending: Optional[Tuple[str, str]] = None) -> dict:
    """Serialize a board to a dict matching GameStateResponse.

    `ending` is a (result, termination) that overrides the board, e.g. a
    resignation or a loss on time.
    """
    over = ending or board_outcome(board)
    return {
        "id": game_id,
        "fen": board.fen(),
        "legal_moves": [] if over else [m.uci() for m in board.legal_moves],
        "turn": "white" if board.turn == chess.WHITE else "black",
        "is_game_over": over is not None,
        "result": over[0] if over else None,
        "termination": over[1] if over else None,
        "move_history": [m.uci() for m in board.move_stack],
    }
