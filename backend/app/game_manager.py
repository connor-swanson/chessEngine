from __future__ import annotations
from typing import Optional
import chess


class GameManager:
    def __init__(self):
        self._boards: dict[int, chess.Board] = {}

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

    def remove(self, game_id: int):
        self._boards.pop(game_id, None)


game_manager = GameManager()
