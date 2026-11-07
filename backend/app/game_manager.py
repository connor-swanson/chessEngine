from __future__ import annotations
from typing import Optional, Tuple
import chess


class GameManager:
    def __init__(self):
        self._boards: dict[int, chess.Board] = {}
        # game_id -> (result, termination) for games ended off the board
        # (resignation, timeout).  Board-decided endings are computed live.
        self._endings: dict[int, Tuple[str, str]] = {}

    def create(self, game_id: int) -> chess.Board:
        board = chess.Board()
        self._boards[game_id] = board
        return board

    def get(self, game_id: int) -> Optional[chess.Board]:
        return self._boards.get(game_id)

    def restore(self, game_id: int, moves: list[str]) -> chess.Board:
        """Rebuild a board from a list of UCI moves and cache it."""
        board = chess.Board()
        for uci in moves:
            board.push(chess.Move.from_uci(uci))
        self._boards[game_id] = board
        return board

    def push_move(self, game_id: int, move_uci: str) -> chess.Move:
        board = self._boards[game_id]
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            raise ValueError(f"Illegal move: {move_uci}")
        board.push(move)
        return move

    def undo(self, game_id: int, plies: int) -> int:
        """Take back up to `plies` half-moves; returns how many were undone."""
        board = self._boards[game_id]
        n = min(plies, len(board.move_stack))
        for _ in range(n):
            board.pop()
        self._endings.pop(game_id, None)
        return n

    def set_ending(self, game_id: int, result: str, termination: str):
        self._endings[game_id] = (result, termination)

    def get_ending(self, game_id: int) -> Optional[Tuple[str, str]]:
        return self._endings.get(game_id)

    def remove(self, game_id: int):
        self._boards.pop(game_id, None)
        self._endings.pop(game_id, None)


game_manager = GameManager()
