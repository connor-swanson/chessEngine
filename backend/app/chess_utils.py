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


def board_to_state(board: chess.Board, game_id: int) -> dict:
    """Serialize a board to a dict matching GameStateResponse."""
    return {
        "id": game_id,
        "fen": board.fen(),
        "legal_moves": [m.uci() for m in board.legal_moves],
        "turn": "white" if board.turn == chess.WHITE else "black",
        "is_game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else None,
        "move_history": [m.uci() for m in board.move_stack],
    }
